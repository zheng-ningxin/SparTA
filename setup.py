
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

rootdir = os.path.dirname(os.path.realpath(__file__))

version = "0.0.1"


def find_template_files():
    template_dir = os.path.join(rootdir, 'sparta/codegen/template')
    files = os.listdir(template_dir)
    templates = []
    for file in files:
        if os.path.splitext(file)[1] in ['.cu', '.cpp', '.json', '.c', '.txt']:
            templates.append(os.path.join(template_dir, file))
    return templates

def _setup():
    ext_modules = []
    if torch.cuda.is_available():
        # Add the extral sparse kernel module
        # cusparse need cuda version higher than 11.1
        cuda_version = torch.version.cuda
        if cuda_version > '11.5':
            cusparse_linear_ext = CUDAExtension(name='cusparse_linear', sources=[
                                    'csrc/cusparse_linear_forward.cpp', 'csrc/cusparse_linear_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-lcusparse', '-O3'])
            ext_modules.append(cusparse_linear_ext)
            cusparse_csr_ext =  CUDAExtension(name='cusparse_csr_cpp', sources=[
                                    'csrc/cusparse_csr_convert_forward.cpp', 'csrc/cusparse_csr_convert_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-lcusparse', '-O3'])
            ext_modules.append(cusparse_csr_ext)
        # the bcsr convert kernel
        bcsr_blockwise_ext = CUDAExtension(name='convert_bcsr_blockwise_cpp', sources=['csrc/convert_bcsr_forward_blockwise.cpp',
                                                            'csrc/convert_bcsr_forward_blockwise_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(bcsr_blockwise_ext)
        
        bcsr_ext = CUDAExtension(name='convert_bcsr_cpp', sources=['csrc/convert_bcsr_forward.cpp',
                                                            'csrc/convert_bcsr_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(bcsr_ext)
        csr_ext = CUDAExtension(name='convert_csr_cpp', sources=['csrc/convert_csr_forward.cpp',
                                                            'csrc/convert_csr_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(csr_ext)
        bcsr_trans_ext = CUDAExtension(name='convert_bcsr_transpose_cpp', sources=['csrc/convert_bcsr_transpose_forward.cpp',
                                                            'csrc/convert_bcsr_transpose_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(bcsr_trans_ext)
        dynamic_attention_ext = CUDAExtension(name='dynamic_sparse_attention_cpp', sources=['csrc/dynamic_sparse_attention_forward.cpp',
                                                            'csrc/dynamic_sparse_attention_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(dynamic_attention_ext)
        dynamic_linear_ext = CUDAExtension(name='dynamic_sparse_linear_cpp', sources=['csrc/dynamic_sparse_linear_forward.cpp',
                                                                'csrc/dynamic_sparse_linear_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(dynamic_linear_ext)

        seqlen_dynamic_attention_ext = CUDAExtension(name='seqlen_dynamic_sparse_attention_cpp', sources=['csrc/seqlen_dynamic_sparse_attention_forward.cpp',
                                                                'csrc/seqlen_dynamic_sparse_attention_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3',  '-lineinfo', "-U__CUDA_NO_HALF_OPERATORS__",
                                                        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF_CONVERSIONS__"])
        ext_modules.append(seqlen_dynamic_attention_ext)
        seqlen_dynamic_linear_ext = CUDAExtension(name='seqlen_dynamic_sparse_linear_cpp', sources=['csrc/seqlen_dynamic_sparse_linear_forward.cpp',
                                                                'csrc/seqlen_dynamic_sparse_linear_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3',  '-lineinfo', "-U__CUDA_NO_HALF_OPERATORS__",
                                                        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF_CONVERSIONS__"])
        ext_modules.append(seqlen_dynamic_linear_ext)

        longformer_dynamic_attention_ext = CUDAExtension(name='longformer_dynamic_attention_cpp', sources=['csrc/longformer_dynamic_sparse_attention_forward.cpp',
                                                                'csrc/longformer_dynamic_sparse_attention_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(longformer_dynamic_attention_ext)
        in_out_elastic_linear_ext = CUDAExtension(name='in_out_elastic_linear_cpp', sources=['csrc/elastic_linear_forward.cpp',\
                                                                'csrc/elastic_linear_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(in_out_elastic_linear_ext)
        openai_bmm_ext = CUDAExtension(name='openai_bmm_cpp', sources=['csrc/openai_bmm_forward.cpp',\
                                                                'csrc/openai_bmm_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(openai_bmm_ext)
        # cusparse_ext = CUDAExtension(name='our_sparse_attention', sources=[
        #                             'csrc/sparse_attention.cpp', 'csrc/sparse_attention_kernel.cu'],
        #                             extra_compile_args=['-std=c++14', '-O3'])
        # ext_modules.append(cusparse_ext)
        blockwise_sparse_linear_ext = CUDAExtension(name='blockwise_sparse_linear_cpp', sources=['csrc/blockwise_dynamic_sparse_linear.cpp',\
                                                                'csrc/blockwise_dynamic_sparse_linear_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(blockwise_sparse_linear_ext)
        dim_dynamic_sparse_linear_ext = CUDAExtension(name='dim_dynamic_sparse_linear_cpp', sources=['csrc/dim_dynamic_sparse_linear_forward.cpp', \
                                                                'csrc/dim_dynamic_sparse_linear_forward_kernel.cu'])
        ext_modules.append(dim_dynamic_sparse_linear_ext)
        condense_sparse_linear_ext = CUDAExtension(name='condense_sparse_linear_cpp', sources=['csrc/dynamic_sparse_linear_condense_forward.cpp',\
                                                                'csrc/dynamic_sparse_linear_condense_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(condense_sparse_linear_ext)
        condense_sparse_linear_ext_v2 = CUDAExtension(name='condense_sparse_linear_v2_cpp', sources=['csrc/dynamic_sparse_linear_condense_forward_v2.cpp',\
                                                                'csrc/dynamic_sparse_linear_condense_forward_v2_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(condense_sparse_linear_ext_v2)
        moe_ext = CUDAExtension(name='sparse_moe', sources=['csrc/moe_sparse_forward.cpp',\
                                                                'csrc/moe_sparse_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3',  '-lineinfo', "-U__CUDA_NO_HALF_OPERATORS__",
                                                        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF_CONVERSIONS__"])
        ext_modules.append(moe_ext)

        cache_atten_ext = CUDAExtension(name='sparse_cache_atten', sources=['csrc/dynamic_sparse_cache_attention_forward.cpp',\
                                                                'csrc/dynamic_sparse_cache_attention_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3',  '-lineinfo', "-U__CUDA_NO_HALF_OPERATORS__",
                                                        "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF_CONVERSIONS__"])
        ext_modules.append(cache_atten_ext)

        sparse_act_convert_ext = CUDAExtension(name='sparse_act_convert', sources=['csrc/sparse_act_convert.cpp', 'csrc/sparse_act_convert_kernel.cu'],
                                               extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(sparse_act_convert_ext)
        
    print(rootdir)
    setup(
        name='SparTA',
        version=version,
        description='Deployment tool',
        author='MSRA',
        author_email="Ningxin.Zheng@microsoft.com",
        packages=find_packages(),
        include_package_data=True,
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension}
    )

if __name__ == '__main__':
    _setup()