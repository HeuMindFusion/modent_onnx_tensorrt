# modent_onnx_tensorrt

# 1. https://github.com/jkjung-avt/tensorrt_demos

# create engine
$ cd ${HOME}/project/tensorrt_demos/ssd
$ ./install_pycuda.sh

$ cd ${HOME}/project/tensorrt_demos/modnet
### check out the "onnx-tensorrt" submodule
$ git submodule update --init --recursive
### patch CMakeLists.txt
$ sed -i '21s/cmake_minimum_required(VERSION 3.13)/#cmake_minimum_required(VERSION 3.13)/' \
      onnx-tensorrt/CMakeLists.txt
### build onnx-tensorrt
$ mkdir -p onnx-tensorrt/build
$ cd onnx-tensorrt/build
$ cmake -DCMAKE_CXX_FLAGS=-I/usr/local/cuda/targets/aarch64-linux/include \
        -DONNX_NAMESPACE=onnx2trt_onnx ..
$ make -j4
### finally, we could build the TensorRT (FP16) engine
$ cd ${HOME}/project/tensorrt_demos/modnet
$ LD_LIBRARY_PATH=$(pwd)/onnx-tensorrt/build \
      onnx-tensorrt/build/onnx2trt modnet.onnx -o modnet.engine \
                                   -d 16 -v
