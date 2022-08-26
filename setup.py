
import os
from tkinter.tix import FileSelectBox
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
        if cuda_version > '11.1':
            cusparse_ext = CUDAExtension(name='cusparse_linear', sources=[
                                    'csrc/cusparse_linear_forward.cpp', 'csrc/cusparse_linear_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-lcusparse', '-O3'])
            ext_modules.append(cusparse_ext)
        # the bcsr convert kernel
        bcsr_ext = CUDAExtension(name='convert_bcsr_cpp', sources=['csrc/convert_bcsr_forward.cpp',
                                                            'csrc/convert_bcsr_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(bcsr_ext)
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
                                    extra_compile_args=['-std=c++14', '-O3'])
        ext_modules.append(seqlen_dynamic_attention_ext)
        seqlen_dynamic_linear_ext = CUDAExtension(name='seqlen_dynamic_sparse_linear_cpp', sources=['csrc/seqlen_dynamic_sparse_linear_forward.cpp',
                                                                'csrc/seqlen_dynamic_sparse_linear_forward_kernel.cu'],
                                    extra_compile_args=['-std=c++14', '-O3'])
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