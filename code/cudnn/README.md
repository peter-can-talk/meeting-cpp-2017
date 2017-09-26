# cudnn

Example of doing an edge-detection convolution using NVIDIA cuDNN.

## Building

Prerequisites:

0. A GPU and the whole CUDA stack, including the `nvcc` compiler,
1. Install NVIDIA cuDNN for your system: https://developer.nvidia.com/rdp/cudnn-download,
2. OpenCV2 (consult your package manager).

Set the `CUDNN_PATH` environment variable and `make`, e.g.:

```shell
$ CUDNN_PATH=/opt/cudnn make
```

## Running

The binary expects the path to an image, e.g. for the `cppcon-logo.png` image
that's already there:

```sh
$ ./conv cppcon-logo.png
```

It then generates an image called `cudnn-out.png`.
