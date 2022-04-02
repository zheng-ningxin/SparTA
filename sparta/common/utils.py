from multiprocessing.sharedctypes import Value
import os
import copy
import json
from array import array
from pickle import NONE
from shutil import copy
import torch
from torch._C import HalfStorageBase, dtype
import numpy as np
import time
import onnx
import onnx.numpy_helper
from nni.compression.pytorch.utils import get_module_by_name



class LayernameModuleWrapper(torch.nn.Module):
    def __init__(self, module, module_bits) -> None:
        """
        Parameters
        ----------
        module : torch.nn.Module
            Layer module of pytorch model
        module_bits : int
            Bits width setting for module
        """
        super().__init__()
        self.module = module
        self.module_bits = module_bits

    def forward(self, inputs):
        inputs = inputs*self.module_bits
        inputs = self.module(inputs)
        return inputs


def measure_time(model, dummy_input, runtimes=200):
    times = []
    with torch.no_grad():
        for runtime in range(runtimes):
            torch.cuda.synchronize()
            start = time.time()
            out=model(*dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end-start)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean*1000, std*1000

def unwrapper(model_onnx):
    """
    Fill onnx config and remove wrapper node in onnx

    Parameters
    ----------
    model_onnx : onnx model
        Onnx model which is converted from pytorch model

    Returns
    -------
    onnx model
        Onnx model which is converted from pytorch model
    dict
        The configuration of onnx model layers and calibration parameters
    """
    # Support Gemm, Conv, Relu, Clip(Relu6) and Maxpool
    # support_op = ['Gemm', 'Conv', 'MatM']
    support_op = ['Gemm', 'Conv', 'MatM']
    op_names = []
    idx = 0
    onnx_config = {}
    valid_count = 0
    const_output_list = []
    mul_input_list = []
    mul_output_list = []
    nd_input_list = []
    while idx < len(model_onnx.graph.node):
        nd = model_onnx.graph.node[idx]
        op_names.append(nd.name)
        if nd.name[0:4] in support_op and  idx > 1:
            # Grad constant node and multiply node
            const_nd = model_onnx.graph.node[idx-2]
            mul_nd = model_onnx.graph.node[idx-1]

            if const_nd.name[0:8] != "Constant" or mul_nd.name[0:3] != "Mul":
                idx += 1
                continue

            valid_count += 1
            # Get index number which is transferred by constant node
            index = int(onnx.numpy_helper.to_array(const_nd.attribute[0].t))
            if index != -1:
                onnx_config[nd.name] = index
        
            const_output_list.append(const_nd.output[0])
            mul_input_list.append(mul_nd.input[1])
            nd_input_list.append(nd.input[0])
            mul_output_list.append(mul_nd.output[0])
            
            mul_output_name = mul_nd.output[0]
            for input_idx in range(len(nd.input)):
                input_name = nd.input[input_idx]
                if input_name == mul_output_name:
                    nd.input[input_idx] = mul_nd.input[0]
            # nd.input[0] = mul_nd.input[0]
            # Remove constant node and multiply node
            
            model_onnx.graph.node.remove(const_nd)
            model_onnx.graph.node.remove(mul_nd)
            idx = idx-2
        idx = idx+1
    # import ipdb; ipdb.set_trace()
    return model_onnx, onnx_config




def serialize_tesa(tesa):
    serialized = {}
    for tesaid in tesa:
        serialized[tesaid] = {}
        for key in tesa[tesaid]:
            if tesa[tesaid][key] is None:
                continue
            serialized[tesaid][key] = tesa[tesaid][key].tolist()
    return serialized

def export_tesa(model, dummy_input, export_dir, tesa=None):
    """
    Export the model to onnx along with its Tesa Attribute
    writing into the onnx file.
    Parameters
    ----------
    model: torch.nn.Module
        The target model to export the onnx and tesa attribute
    dummy_input: torch.Tensor/tuple of Tensors
        The dummy input of the target model
    export_dir: str
        path to export the onnx
    tesa: dict or None
        The tesa attribute of the target model, {layer_name: {tensor_name: tensor}}
    """
    os.makedirs(export_dir, exist_ok=True)
    support_op = (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d)
    assert isinstance(model, torch.nn.Module)
    torch.save(model.state_dict(), os.path.join(export_dir, 'state_dict.pth'))
    if tesa is None:
        tesa = {}
        # All values are remained
        for name, module in model.named_modules():
            tesa[name] = {}
            if isinstance(module, support_op):
                if hasattr(module, 'weight') and module.weight is not None:
                    tesa[name]['weight'] = torch.ones_like(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    tesa[name]['bias'] = torch.ones_like(module.bias)
    # apply tesa
    for name in tesa:
        # import pdb; pdb.set_trace()
        _, leaf_module = get_module_by_name(model, name)
        masks = tesa[name]
        for key in masks:
            _tensor = getattr(leaf_module, key)
            if _tensor is None:
                continue
            # apply the mask for the onnx values
            _tensor.data = _tensor.data * tesa[name][key].data.to(_tensor.device)
    ori_onnx_path = os.path.join(export_dir, 'model_no_tesa.onnx')
    torch.onnx.export(model, dummy_input, ori_onnx_path, opset_version=10)

    uid = 1
    name2uid = {}
    exported_tesa = {}
    for name, module in model.named_modules():
        if name in tesa and isinstance(module, support_op):
            name2uid[name] = uid
            exported_tesa[uid] = tesa[name] 
            wrapper_module = LayernameModuleWrapper(module, uid)
            father_m, leaf_m = get_module_by_name(model, name)
            setattr(father_m, name.split('.')[-1], wrapper_module)
            uid += 1
    
    onnx_path = os.path.join(export_dir, 'tmp.onnx')
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=10)
    model_onnx = onnx.load(onnx_path)
    model_onnx, tesa_onnx_map = unwrapper(model_onnx)
    
    for node in model_onnx.graph.node:
        node_name = node.name
        if node_name in tesa_onnx_map:
            new_attr = onnx.helper.make_attribute("tesa_id", tesa_onnx_map[node_name])
            node.attribute.append(new_attr)

    onnx.save(model_onnx, os.path.join(export_dir, 'model_tesa.onnx'))
    # onnx.checker.check_model(model_onnx)
    torch.save(exported_tesa, os.path.join(export_dir, 'tesa'))
    json_tesa = serialize_tesa(exported_tesa)
    # with open(os.path.join(export_dir, 'tesa.json'), 'w') as f:
    #     json.dump(json_tesa, f)
    tesaid2name = {}
    for name, tesaid in name2uid.items():
        tesaid2name[tesaid] = [name]
    # import pdb; pdb.set_trace()
    for onnx_node, tesaid in tesa_onnx_map.items():
        tesaid2name[tesaid].append(onnx_node)

    torch.save(tesaid2name, os.path.join(export_dir, 'tesaid_2_names'))
    if os.path.exists(onnx_path):
        os.remove(onnx_path)

def _setattr(model, name, module):
    """
    Parameters
    ----------
    model : pytorch model
        The model to speedup by quantization
    name : str
        name of pytorch module
    module : torch.nn.Module
        Layer module of pytorch model
    """
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def export_tesa_debug(model, dummy_input, export_dir, tesa=None):
    """
    Export the model to onnx along with its Tesa Attribute
    writing into the onnx file.
    Parameters
    ----------
    model: torch.nn.Module
        The target model to export the onnx and tesa attribute
    dummy_input: torch.Tensor/tuple of Tensors
        The dummy input of the target model
    export_dir: str
        path to export the onnx
    tesa: dict or None
        The tesa attribute of the target model, {layer_name: {tensor_name: tensor}}
    """
    os.makedirs(export_dir, exist_ok=True)
    support_op = (torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d)
    assert isinstance(model, torch.nn.Module)
    torch.save(model.state_dict(), os.path.join(export_dir, 'state_dict.pth'))
    if tesa is None:
        tesa = {}
        # All values are remained
        for name, module in model.named_modules():
            tesa[name] = {}
            if isinstance(module, support_op):
                if hasattr(module, 'weight') and module.weight is not None:
                    tesa[name]['weight'] = torch.ones_like(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    tesa[name]['bias'] = torch.ones_like(module.bias)
    # apply tesa
    for name in tesa:
        # import pdb; pdb.set_trace()
        _, leaf_module = get_module_by_name(model, name)
        masks = tesa[name]
        for key in masks:
            _tensor = getattr(leaf_module, key)
            if _tensor is None:
                continue
            # apply the mask for the onnx values
            _tensor.data = _tensor.data * tesa[name][key].data.to(_tensor.device)

    torch.onnx.export(model, dummy_input, os.path.join(export_dir, "before_wrap_model.onnx"), opset_version=10)
    uid = 1
    name2uid = {}
    exported_tesa = {}
    for name, module in model.named_modules():
        if name in tesa and isinstance(module, support_op):
            name2uid[name] = uid
            exported_tesa[uid] = tesa[name] 
            wrapper_module = LayernameModuleWrapper(module, uid)
            #father_m, leaf_m = get_module_by_name(model, name)
            #setattr(father_m, name.split('.')[-1], wrapper_module)
            _setattr(model, name, wrapper_module)
            uid += 1
    onnx_path = os.path.join(export_dir, 'model.onnx')
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=10)
    model_onnx = onnx.load(onnx_path)
    model_onnx, tesa_onnx_map = unwrapper(model_onnx)
    # for node in model_onnx.graph.node:
    #     node_name = node.name
    #     if node_name in tesa_onnx_map:
    #         new_attr = onnx.helper.make_attribute("tesa_id", tesa_onnx_map[node_name])
    #         node.attribute.append(new_attr)
    onnx.save(model_onnx, os.path.join(export_dir, "model_resave.onnx"))
    # onnx.save(model_onnx, onnx_path)
    # onnx.checker.check_model(model_onnx)
    # torch.save(exported_tesa, os.path.join(export_dir, 'tesa'))
    # json_tesa = serialize_tesa(exported_tesa)
    # # with open(os.path.join(export_dir, 'tesa.json'), 'w') as f:
    # #     json.dump(json_tesa, f)
    # tesaid2name = {}
    # for name, tesaid in name2uid.items():
    #     tesaid2name[tesaid] = [name]
    # # import pdb; pdb.set_trace()
    # for onnx_node, tesaid in tesa_onnx_map.items():
    #     tesaid2name[tesaid].append(onnx_node)

    # torch.save(tesaid2name, os.path.join(export_dir, 'tesaid_2_names'))

def convert_to_block_csr_bin(m_tensor, v_tensor, block_h, block_w):
    assert len(m_tensor.size()) == 2
    size_h, size_w = m_tensor.size()
    if size_h % block_h != 0 or size_w % block_w != 0:
        return None, None, None
    rows = []
    cols = []
    values = []
    for _i in range(size_w//block_w):
        rows.append(len(cols))
        for _j in range(size_h//block_h):
            i_start = _i * block_w
            i_end = (_i+1) * block_w
            j_start = _j * block_h
            j_end = (_j+1) * block_h
            if torch.sum(m_tensor[j_start:j_end, i_start:i_end]) > 0:
                cols.append(_j)
                values.extend(v_tensor[j_start:j_end, i_start:i_end].flatten().tolist())
    rows.append(len(cols))
    return rows, cols, values

def convert_to_block_csr(m_tensor, v_tensor, block_h, block_w):
    raise NotImplementedError
    assert len(m_tensor.size()) == 2
    size_h, size_w = m_tensor.size()
    if size_h % block_h != 0 or size_w % block_w != 0:
        return None, None, None
    rows = []
    cols = []
    values = []
    for _i in range(size_h//block_h):
        rows.append(len(cols))
        for _j in range(size_w//block_w):
            i_start = _i * block_h
            i_end = (_i+1) * block_h
            j_start = _j * block_w
            j_end = (_j+1) * block_w
            if torch.sum(m_tensor[i_start:i_end, j_start:j_end]) > 0:
                cols.append(_j)
                values.extend(v_tensor[i_start:i_end,j_start:j_end].flatten().tolist())
    rows.append(len(cols))
    return rows, cols, values


def convert_to_csr(m_tensor, v_tensor):
    assert len(m_tensor.size()) == 2
    
    with torch.no_grad():
        sparsity_pos = m_tensor == 0
        row_idx = []
        col_idx = []
        values = []
        H, W = sparsity_pos.size()
        for i in range(H):
            row_idx.append(len(values))
            for j in range(W):
                if sparsity_pos[i][j] == True:
                    continue
                col_idx.append(j)
                values.append(v_tensor.data[i][j])
        row_idx.append(len(values))
    return row_idx, col_idx, values

def write_array(data, file_path, dtype="i"):
    array_data = array(dtype, data)
    with open(file_path, 'wb') as f:
        array_data.tofile(f)

def generate_block_sparse_cfg(tesa_path, state_path, id_map_path, out_dir, block_h=32, block_w=32, sparse_block_cfg=None):
    os.makedirs(out_dir, exist_ok=True)
    if sparse_block_cfg is None:
        sparse_block_cfg = {}
    tesa = torch.load(tesa_path, map_location='cpu')
    cfg_path = os.path.join(out_dir, 'config')
    state_dict = torch.load(state_path, map_location='cpu')
    id_maps = torch.load(id_map_path, map_location='cpu')
    with open(cfg_path, 'w') as f:
        for tesaid in tesa:
            print(f"Dump the {tesaid}-th block index")
            # assert hasattr(tesa[tesaid], 'weight')
            torch_name = id_maps[tesaid][0]
            # import pdb; pdb.set_trace()
            sparse_ratio = torch.sum(tesa[tesaid]['weight']) / tesa[tesaid]['weight'].numel()
            if sparse_ratio > 0.5:
                # too few sparsity
                continue
            if tesaid in sparse_block_cfg:
                _block_h, _block_w = sparse_block_cfg[tesaid]
            else:
                _block_h, _block_w = block_h, block_w
            print(f"Tesa-{tesaid} Convering with block size: ", _block_h, _block_w)
            row_d, col_d, value_d = convert_to_block_csr_bin(tesa[tesaid]['weight'].t(), state_dict[torch_name+'.weight'].t(), block_h=_block_h, block_w=_block_w)
            bias_d, bias_f = None, ""
            if torch_name + '.bias' in state_dict:
                # matmul has bias
                bias_f = f"bias_{tesaid}.bin"
                bias_d = state_dict[torch_name + '.bias'].tolist()
            if row_d is None:
                # cannot convert to the block sparse
                continue
            # write the data in the binary format
            row_f, col_f, value_f = f"row_{tesaid}.bin", f"col_{tesaid}.bin", f"value_{tesaid}.bin"
            write_array(row_d, os.path.join(out_dir, row_f))
            write_array(col_d, os.path.join(out_dir, col_f))
            write_array(value_d, os.path.join(out_dir, value_f), 'f')
            if bias_d is not None:
                write_array(bias_d, os.path.join(out_dir, bias_f), 'f')
            f.write(f"{tesaid} BlockSparse kernel_{tesaid} {row_f} {col_f} {value_f} {bias_f}\n")

def generate_random(count, d_type, start, end):
    if d_type == 'i':
        re = np.random.randint(start, end, size=count)
        return list(re)
    elif d_type == 'b':
        re = np.random.randint(start, end, size=count, dtype=np.int8)
        return list(re)
    else:
        raise Exception('Not supported')
    

def generate_mobilenet_quantize_cfg(tesa_path, state_path, id_map_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tesa = torch.load(tesa_path, map_location='cpu')
    cfg_path = os.path.join(out_dir, 'config')
    # state_dict = torch.load(state_path)
    id_maps = torch.load(id_map_path, map_location='cpu')
    # import pdb; pdb.set_trace()
    with open(cfg_path, 'w') as f:
        for tesaid in tesa:
            print(f"Dump the {tesaid}-th block index")
            # assert hasattr(tesa[tesaid], 'weight')
            # torch_name = id_maps[tesaid][0]
            weight_d = generate_random(tesa[tesaid]['weight'].numel(), 'b', 0, 10)
            weight_f =  f"weight_{tesaid}.bin"
            write_array(weight_d, os.path.join(out_dir,weight_f), 'b')
            scale_integer_d = generate_random(1, 'i', 0, 10)
            scale_integer_f = f"scale_integer_{tesaid}.bin"
            write_array(scale_integer_d, os.path.join(out_dir, scale_integer_f))
            scale_shift_d = generate_random(1, 'i', 0, 10)
            scale_shift_f = f"scale_shift_{tesaid}.bin"
            write_array(scale_shift_d, os.path.join(out_dir, scale_shift_f))
            bias_data_d = generate_random(tesa[tesaid]['weight'].numel(), 'i', 0, 10)
            bias_data_f = f"bias_{tesaid}.bin"
            write_array(bias_data_d, os.path.join(out_dir, bias_data_f))
            f.write(f"{tesaid} Quantize kernel_{tesaid} 8 8 {weight_f} {scale_integer_f} {scale_shift_f} {bias_data_f}\n")

def generate_quantize_dot_cfg(tesa_path, state_path, id_map_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tesa = torch.load(tesa_path, map_location='cpu')
    cfg_path = os.path.join(out_dir, 'config')
    # state_dict = torch.load(state_path)
    id_maps = torch.load(id_map_path, map_location='cpu')
    with open(cfg_path, 'w') as f:
        for tesaid in tesa:
            print(f"Dump the {tesaid}-th block index")
            # assert hasattr(tesa[tesaid], 'weight')
            # torch_name = id_maps[tesaid][0]
            weight_d = generate_random(tesa[tesaid]['weight'].numel(), 'i', 0, 10)
            weight_f =  f"weight_{tesaid}.bin"
            write_array(weight_d, os.path.join(out_dir,weight_f))
            scale_integer_d = generate_random(1, 'i', 0, 10)
            scale_integer_f = f"scale_integer_{tesaid}.bin"
            write_array(scale_integer_d, os.path.join(out_dir, scale_integer_f))
            scale_shift_d = generate_random(1, 'i', 0, 10)
            scale_shift_f = f"scale_shift_{tesaid}.bin"
            write_array(scale_shift_d, os.path.join(out_dir, scale_shift_f))
            bias_data_d = generate_random(tesa[tesaid]['weight'].numel(), 'i', 0, 10)
            bias_data_f = f"bias_{tesaid}.bin"
            write_array(bias_data_d, os.path.join(out_dir, bias_data_f))
            f.write(f"{tesaid} Quantize kernel_{tesaid} 8 8 {weight_f} {scale_integer_f} {scale_shift_f} {bias_data_f}\n")


def fake_quantize(value):
    count = len(value)
    return generate_random(count, 'b', 0, 32)

def generate_block_quantize_cfg(tesa_path, state_path, id_map_path,out_dir, block_h=32, block_w=32, sparse_block_cfg=None, sparsity_threshold=0.5):
    os.makedirs(out_dir, exist_ok=True)
    if sparse_block_cfg is None:
        sparse_block_cfg = {}
    tesa = torch.load(tesa_path, map_location='cpu')
    cfg_path = os.path.join(out_dir, 'config')
    state_dict = torch.load(state_path, map_location='cpu')
    id_maps = torch.load(id_map_path, map_location='cpu')
    with open(cfg_path, 'w') as f:
        for tesaid in tesa:
            print(f"Dump the {tesaid}-th block index")
            # assert hasattr(tesa[tesaid], 'weight')
            torch_name = id_maps[tesaid][0]
            sparse_ratio = torch.sum(tesa[tesaid]['weight']) / tesa[tesaid]['weight'].numel()
            if sparse_ratio > sparsity_threshold:
                # too few sparsity


                print(f"Dump the {tesaid}-th block index")
                # assert hasattr(tesa[tesaid], 'weight')
                # torch_name = id_maps[tesaid][0]
                weight_d = generate_random(tesa[tesaid]['weight'].numel(), 'i', 0, 10)
                weight_f =  f"weight_{tesaid}.bin"
                write_array(weight_d, os.path.join(out_dir,weight_f))
                scale_integer_d = generate_random(1, 'i', 0, 10)
                scale_integer_f = f"scale_integer_{tesaid}.bin"
                write_array(scale_integer_d, os.path.join(out_dir, scale_integer_f))
                scale_shift_d = generate_random(1, 'i', 0, 10)
                scale_shift_f = f"scale_shift_{tesaid}.bin"
                write_array(scale_shift_d, os.path.join(out_dir, scale_shift_f))
                bias_data_d = generate_random(tesa[tesaid]['weight'].numel(), 'i', 0, 10)
                bias_data_f = f"bias_{tesaid}.bin"
                write_array(bias_data_d, os.path.join(out_dir, bias_data_f))
                f.write(f"{tesaid} Quantize kernel_{tesaid} 8 8 {weight_f} {scale_integer_f} {scale_shift_f} {bias_data_f}\n")

            else:
                if tesaid in sparse_block_cfg:
                    _block_h, _block_w = sparse_block_cfg[tesaid]
                else:
                    _block_h, _block_w = block_h, block_w
                _tmp_weight_shape = tesa[tesaid]['weight'].size()
                print(f'{tesaid} Covering with block: {_block_h}x{_block_w} Weight Shape: {_tmp_weight_shape}')
                if tesa[tesaid]['weight'].size(0) % _block_h !=0 or tesa[tesaid]['weight'].size(1) % _block_w !=0:
                    continue
                # import pdb; pdb.set_trace()
                row_d, col_d, value_d = convert_to_block_csr(tesa[tesaid]['weight'], state_dict[torch_name+'.weight'], block_h=_block_h, block_w=_block_w)
                value_d = fake_quantize(value_d)
                bias_d, bias_f = None, ""
                if torch_name + '.bias' in state_dict:
                    # matmul has bias
                    bias_f = f"bias_{tesaid}.bin"
                    bias_d = state_dict[torch_name + '.bias'].tolist()
                if row_d is None:
                    # cannot convert to the block sparse
                    continue
                # write the data in the binary format
                row_f, col_f, value_f = f"row_{tesaid}.bin", f"col_{tesaid}.bin", f"value_{tesaid}.bin"
                scale_integer_f, scale_shift_f = f"scale_integer_{tesaid}.bin", f"scale_shift_{tesaid}.bin"
                write_array(row_d, os.path.join(out_dir, row_f))
                write_array(col_d, os.path.join(out_dir, col_f))
                write_array(value_d, os.path.join(out_dir, value_f), 'b')
                # TODO : use the write value
                # write a temp scale int value
                write_array([1], os.path.join(out_dir, scale_integer_f))
                write_array([1], os.path.join(out_dir, scale_shift_f))

                if bias_d is not None:
                    write_array(bias_d, os.path.join(out_dir, bias_f), 'f')
                f.write(f"{tesaid} BlockQuantize kernel_{tesaid} 8 8 {row_f} {col_f} {value_f} {scale_integer_f} {scale_shift_f} {bias_f}\n")



def generate_sputnik_sparse_cfg(tesa_path, state_path, id_map_path, out_dir):

    def sort_row_swizzle(n_row, row_index):
        assert len(row_index) == n_row + 1
        row_count = [(row_index[i+1]-row_index[i], i) for i in range(n_row)]
        swizzle = sorted(row_count, reverse=True)
        return [x[1] for x in swizzle]
    os.makedirs(out_dir, exist_ok=True)

    tesa = torch.load(tesa_path, map_location='cpu')
    cfg_path = os.path.join(out_dir, 'config')
    state_dict = torch.load(state_path, map_location='cpu')
    id_maps = torch.load(id_map_path, map_location='cpu')
    with open(cfg_path, 'w') as f:
        for tesaid in tesa:
            f.write(f"{tesaid} Sputnik\n")
            continue
            print(f"Dump the {tesaid}-th block index")
            # assert hasattr(tesa[tesaid], 'weight')
            torch_name = id_maps[tesaid][0]
            # import pdb; pdb.set_trace()
            sparse_ratio = torch.sum(tesa[tesaid]['weight']) / tesa[tesaid]['weight'].numel()
            # if sparse_ratio > 0.5:
            #     # too few sparsity
            #     continue
            # sputnik only support sparse * dense 
            row_d, col_d, value_d = convert_to_csr(tesa[tesaid]['weight'], state_dict[torch_name+'.weight'])
            swizzle_d = sort_row_swizzle(len(row_d)-1, row_d)

            bias_d, bias_f = None, ""
            if torch_name + '.bias' in state_dict:
                # matmul has bias
                bias_f = f"bias_{tesaid}.bin"
                bias_d = state_dict[torch_name + '.bias'].tolist()
            if row_d is None:
                # cannot convert to the block sparse
                continue
            # write the data in the binary format
            row_f, col_f, value_f, swizzle_f = f"row_{tesaid}.bin", f"col_{tesaid}.bin", f"value_{tesaid}.bin", f"swizzle_{tesaid}.bin"
            write_array(row_d, os.path.join(out_dir, row_f))
            write_array(col_d, os.path.join(out_dir, col_f))
            write_array(value_d, os.path.join(out_dir, value_f), 'f')
            write_array(swizzle_d, os.path.join(out_dir, swizzle_f))
            if bias_d is not None:
                write_array(bias_d, os.path.join(out_dir, bias_f), 'f')
            f.write(f"{tesaid} BlockSparse kernel_{tesaid} {row_f} {col_f} {value_f} {swizzle_f}\n")


def generate_hipsparse_sparse_cfg(tesa_path, state_path, id_map_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    tesa = torch.load(tesa_path, map_location='cpu')
    cfg_path = os.path.join(out_dir, 'config')
    state_dict = torch.load(state_path, map_location='cpu')
    id_maps = torch.load(id_map_path, map_location='cpu')
    with open(cfg_path, 'w') as f:
        for tesaid in tesa:
            f.write(f"{tesaid} HipSparse\n")
            continue

def inject_kernel(template_path, kernel_json, op_type, id_map_path, out_dir):
    nnfusion_home = os.getenv('NNFUSION_HOME')
    assert nnfusion_home is not None
    os.makedirs(out_dir, exist_ok=True)
    with open(template_path, 'r') as f:
        template = json.load(f)
        template = template[0]
    id_maps = torch.load(id_map_path)
    name_2_tid= {}
    id_2_name = {}
    for tid, names in id_maps.items():
        id_2_name[tid] = names[0]
        name_2_tid[names[0]] =tid
    
    with open(kernel_json, 'r') as f:
        kernels = json.load(f)
    for kernel_name in kernels:
        tesa_id = name_2_tid[kernel_name]
        code = kernels[kernel_name]['code']
        code = code.replace('COMMENT_TAG', 'TESAID:{}'.format(tesa_id))
        template['code'] = code + tesa_id * ' '
        template['kernel_identifier'] = 'kernel_{}'.format(tesa_id)
        template['op_type'] = op_type
        grid_dim = kernels[kernel_name]['launch_config']['dimGrid']+ [1]
        template['gridDim'] = grid_dim
        block_dim = kernels[kernel_name]['launch_config']['dimBlock'] + [1]
        template['blockDim'] = block_dim
        f_path =  os.path.join(out_dir, f"{tesa_id}.json")
        print(f_path)
        with open(f_path, 'w') as f:
            json.dump(template, f)
        os.system(f"python {nnfusion_home}/src/tools/nnfusion/kernel_db/convert_external_spargen.py {f_path} CUDA_GPU")