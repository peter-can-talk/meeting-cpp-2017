# mxnet

Demo of creating a deep neural network with MXNet's C++ api.

## Building

### Prerequisites

1. Clone MXNet: `git clone https://github.com/apache/incubator-mxnet`,
2. Build with CMake, using something like this: `make -j4 USE_BLAS=apple USE_OPENCV=1 USE_CPP_PACKAGE=1 USE_OPENMP=0` (the important part is `USE_CPP_PACKAGE`, the rest may differ for you),
5. OpenCV2 (consult your package manager).

### Build the Neural Network

The `lenet.cpp` file contains code for a neural network using MXNet. You can
build it with the Makefile in this folder. For this, you need to set the
`MXNET_PATH` environment variable to point to your MXNet library path prefix,
e.g. for me:

```sh
MXNET_PATH=~/Documents/Libraries/mxnet make
```

### Build the Demo

The demo GUI uses Qt (5.7 or newer). You will need to download it. Then generate
a Makefile using `qmake` and make:

```sh
$ cd demo
$ qmake
$ make
```

## Running

First run the binary produced from `lenet.cpp`, which will train the neural
network. You can optionally pass a number of epochs to train as a command line
argument. Anywhere between 1 and 10 is sensible. The default is two epochs,
which gets you to around 98% accuracy for the task (handwritten digit
classification). Once it's done training, it will start listening on a socket
for prediction requests. At this point, launch the demo app binary, which will
connect to the server, allowing you to request predictions. Like so:

```sh
$ LD_LIBRARY_PATH=/path/to/mxnet/lib ./lenet MNIST_data
$ demo/<demo binary>
```

where `<demo binary>` is `demo.app/Contents/MacOS/demo` for example. Differs on
Linux or Windows.
