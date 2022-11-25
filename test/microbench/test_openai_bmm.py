import torch
import time
import sparta
from sparta.opset.bcsr_converter import BcsrConverter
from sparta.common.utils import convert_bcsr
import openai_bmm_cpp

def convert_to_full_mask(block_layout, block_size):
    full_mask = block_layout.repeat_interleave(block_size[0], dim=0)
    full_mask = full_mask.repeat_interleave(block_size[1], dim=1)
    return full_mask


if __name__ == '__main__':
    batchsize = 1
    # M = 2048
    # K = 2048
    # N = 2048
    # M = 4096
    # K = 4096
    # N = 4096
    for M, K, N in [(4096, 4096, 4096), (4096, 768, 4096)]:
        block_h = 32
        block_w = 64
        RUNTIME = 2000
        
        for sparsity_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
            with open('stil.log', 'a') as f:
                t_re = (t_end-t_start)*1000/RUNTIME)
                f.write(f'MKN: {M} {K} {N} sparsity:{sparsity_ratio} \n')
        
            ########################################################################
            # measure the dense baseline time
            A = torch.rand(batchsize, M, K).cuda()
            B = torch.rand(K, N).cuda()
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(RUNTIME):
                C = torch.matmul(A, B)
            torch.cuda.synchronize()
            t_end = time.time()
            with open('stil.log', 'a') as f:
                t_re = (t_end-t_start)*1000/RUNTIME)
                f.write(f'Dense time baseline latency(ms):{t_re} \n')
            
            #######################################################################
            # original openai sparse kernel
            A = torch.rand(batchsize, M, K).cuda()
            A_copy = A.clone().detach()
            B = torch.rand(batchsize, K, N).cuda()
            mask = torch.ones(M, K).cuda()
            block_wise_weight = torch.rand(M//block_h, K//block_w, dtype=torch.float32).cuda()
            block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
            # print('Block-wise sparsity ratio:', torch.sum(block_mask)/block_mask.numel())
            full_mask = convert_to_full_mask(block_mask, (block_h, block_w))
            A *= full_mask
            ref_out = torch.einsum('bmk,bkn->bmn',A, B)
            converter_1 = BcsrConverter()
            row_ptr, col_idx, row_pos, vals = converter_1(full_mask, A, block_h, block_w)
            block_nnz = row_ptr[M//block_h]
            out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)
            if not torch.allclose(out, ref_out, rtol=1e-04, atol=1e-03):
                import ipdb; ipdb.set_trace()
            assert torch.allclose(out, ref_out, rtol=1e-04, atol=1e-03)
            # measure the latency of the original openai kernel
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(RUNTIME):
                out = openai_bmm_cpp.forward(row_ptr, col_idx, vals, B, M, K, N, batchsize, block_nnz)
            torch.cuda.synchronize()
            t_end = time.time()
            with open('stil.log', 'a') as f:
                t_re = (t_end-t_start)*1000/RUNTIME)
                f.write(f'Original openai bmm latency(ms):{t_re} \n')

            ###########################################################################
            # following is the condense matmul computation
            B = torch.rand(batchsize, K, N).cuda()
            A = torch.rand(batchsize, M, K).cuda()    
            new_block_h = block_h
            new_block_w = 1
            converter_2 = BcsrConverter(True)
            t_block_wise_weight = torch.rand(M//new_block_h, K//new_block_w, dtype=torch.float32).cuda()
            t_block_mask = (t_block_wise_weight > sparsity_ratio).to(torch.int32)
            # print("Block-wise sparsity ratio:", torch.sum(t_block_mask)/t_block_mask.numel())
            t_full_mask = convert_to_full_mask(t_block_mask, (new_block_h, new_block_w))
            A *= t_full_mask
            ref_out = torch.einsum('bmk,bkn->bmn',A, B)
            # print(torch.squeeze(A).size())
            t_row_ptr, t_col_idx, t_row_pos, t_vals = converter_2(t_full_mask, torch.squeeze(A), new_block_h, new_block_w)
            t_block_nnz = t_row_ptr[M//new_block_h]
            condense_out = openai_bmm_cpp.forward_condense(t_row_ptr, t_col_idx, t_vals, B, M, K, N, new_block_h, new_block_w, batchsize, t_block_nnz)
            assert torch.allclose(condense_out, ref_out, rtol=1e-03, atol=1e-03)
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(RUNTIME):
                condense_out = openai_bmm_cpp.forward_condense(t_row_ptr, t_col_idx, t_vals, B, M, K, N, new_block_h, new_block_w, batchsize, t_block_nnz)
            torch.cuda.synchronize()
            t_end = time.time()
            with open('stil.log', 'a') as f:
                t_re = (t_end-t_start)*1000/RUNTIME)
                f.write(f'Condense openai bmm latency(ms): {t_re} \n')

            
            #############################################################################
            # following is the condense module on the M dimension
            A = torch.rand(batchsize, M, K).cuda()
            B = torch.rand(batchsize, K, N).cuda()
            # A[:,:32] = 1
            new_block_h = 1
            new_block_w = 64
            block_wise_weight = torch.rand(M//new_block_h, K//new_block_w, dtype=torch.float32).cuda()
            block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
            # block_mask[:] = 1
            # block_mask[:32] = 1
            # block_mask[1:10] = 0
            # print("Block-wise sparsity ratio:", torch.sum(block_mask)/block_mask.numel())
            # import ipdb; ipdb.set_trace()
            full_mask = convert_to_full_mask(block_mask, (new_block_h, new_block_w))
            A *= full_mask
            ref_out = torch.einsum('bmk,bkn->bmn',A, B)
            m_csr_row, m_csr_col, m_csr_val = convert_bcsr(full_mask.t(), torch.squeeze(A).t(), new_block_w, new_block_h)
            m_csr_row, m_csr_col, m_csr_val = m_csr_row.cuda(), m_csr_col.cuda(), m_csr_val.cuda()
            m_block_nnz = m_csr_row[K//new_block_w].item()
            # print(m_block_nnz)
            # import ipdb; ipdb.set_trace()
            condense_out_m = openai_bmm_cpp.forward_condense_m(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)
            flag = torch.allclose(condense_out_m, ref_out, rtol=1e-04, atol=1e-03)
            if not flag:
                import ipdb; ipdb.set_trace()
                print("Correctness Failed!")
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(RUNTIME):
                condense_out_m = openai_bmm_cpp.forward_condense_m(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)
            torch.cuda.synchronize()
            t_end = time.time()
            with open('stil.log', 'a') as f:
                t_re = (t_end-t_start)*1000/RUNTIME)
                f.write(f'Condense openai bmm on dim m latency(ms): {t_re} \n')


            # print('Condense openai bmm on dim m latency(ms):', (t_end-t_start)*1000/RUNTIME)
            #########################################################################
            A = torch.rand(batchsize, M, K).cuda()
            B = torch.rand(batchsize, K, N).cuda()
            # A[:,:32] = 1
            new_block_h = 1
            new_block_w = 64
            block_wise_weight = torch.rand(M//new_block_h, K//new_block_w, dtype=torch.float32).cuda()
            block_mask = (block_wise_weight > sparsity_ratio).to(torch.int32)
            # block_mask[:] = 1
            # block_mask[:32] = 1
            # block_mask[1:10] = 0
            # print("Block-wise sparsity ratio:", torch.sum(block_mask)/block_mask.numel())
            # import ipdb; ipdb.set_trace()
            full_mask = convert_to_full_mask(block_mask, (new_block_h, new_block_w))
            A *= full_mask
            ref_out = torch.einsum('bmk,bkn->bmn',A, B)
            m_csr_row, m_csr_col, m_csr_val = convert_bcsr(full_mask.t(), torch.squeeze(A).t(), new_block_w, new_block_h)
            m_csr_row, m_csr_col, m_csr_val = m_csr_row.cuda(), m_csr_col.cuda(), m_csr_val.cuda()
            m_block_nnz = m_csr_row[K//new_block_w].item()
            # print(m_block_nnz)
            # import ipdb; ipdb.set_trace()
            condense_out_m = openai_bmm_cpp.forward_condense_m_v2(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)
            flag = torch.allclose(condense_out_m, ref_out, rtol=1e-04, atol=1e-03)
            if not flag:
                import ipdb; ipdb.set_trace()
                print("Correctness Failed!")
            torch.cuda.synchronize()
            t_start = time.time()
            for _ in range(RUNTIME):
                condense_out_m = openai_bmm_cpp.forward_condense_m_v2(m_csr_row, m_csr_col, m_csr_val, B, M, K, N, new_block_h, new_block_w, batchsize, m_block_nnz)
            torch.cuda.synchronize()
            t_end = time.time()
            with open('stil.log', 'a') as f:
                t_re = (t_end-t_start)*1000/RUNTIME)
                f.write(f'Condense openai bmm on dim m latency v2(ms): {t_re} \n')
                f.write("##################################\n\n")
            # print('Condense openai bmm on dim m latency v2(ms):', (t_end-t_start)*1000/RUNTIME)
            # print('##############################\n\n')