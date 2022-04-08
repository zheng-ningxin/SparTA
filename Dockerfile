FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04
    
# Install tools and dependencies.
RUN apt-get -y update --fix-missing
RUN apt-get install -y \
  emacs \
  git \
  wget \
  libgoogle-glog-dev 

# Setup to install the latest version of cmake.
RUN apt-get install -y software-properties-common && \
    apt-get update && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && apt-get install -y cmake
# Set the working directory.
WORKDIR /root

#install sputnik
RUN git clone --recurse-submodules https://github.com/zheng-ningxin/sputnik.git && \
    cd sputnik && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DCUDA_ARCHS="70;75" && \
    make -j && cp sputnik/libspmm.so /usr/local/lib/ && cp -r /root/sputnik/third_party/abseil-cpp/absl /usr/local/include/

# install nnfusion
RUN git clone https://github.com/zheng-ningxin/nnfusion.git && cd nnfusion && git checkout hubert_antares && \
    ./maint/script/install_dependency.sh && mkdir build && cd build && cmake .. && make -j

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh && \
    bash Anaconda3-2021.11-Linux-x86_64.sh -b -p /root/anaconda && \
    eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda create -n artifact python=3.8 -y && \
    conda activate artifact && pip install torch==1.7.0 torchvision==0.8.0

# install nni
RUN git clone https://github.com/zheng-ningxin/nni.git && cd nni && git checkout artifact && \
    eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && pip install -U -r dependencies/setup.txt && \
    pip install -r dependencies/develop.txt && python setup.py develop && pip install tensorboard transformers==3.5.0 onnxruntime graphviz onnx

# install antares
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && \
    pip install antares==0.3.12.1

# install tensorrt
RUN  eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && \
    pip install pycuda==2020.1 && python3 -m pip install --upgrade setuptools pip && \
    python3 -m pip install nvidia-pyindex && python3 -m pip install --upgrade nvidia-tensorrt==8.4.0.6 && \
    pip install six

# install tvm
RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz && \
    tar -xvf clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-16.04.tar.xz
RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && conda activate artifact && \
    git clone --recursive https://github.com/linbinskn/tvm.git tvm && cd tvm && git checkout cuda && \
    apt-get update && apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev && \
    cd build && cmake .. && make -j4 && echo "export PYTHONPATH=/root/tvm/python:${PYTHONPATH}" >> /root/.bashrc && \
    pip install decorator

# configure the bashrc
RUN echo 'export NNFUSION_HOME=/root/nnfusion \n\
export PATH=$NNFUSION_HOME/build/src/tools/nnfusion:$PATH \n\
export CUDA_HOME=/usr/local/cuda \n\
' >> /root/.bashrc
