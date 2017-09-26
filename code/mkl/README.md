# mkl

Example of doing an edge-detection convolution using Intel MKL.

## Building

Prerequisites:

1. Install Intel MKL for your system: https://software.intel.com/en-us/mkl,
2. OpenCV2 (consult your package manager).

Then just `make`. If your MKL did not end up under `/opt/intel/mkl`, change the
Makefile or set the `MKL_PATH` environment variable before the `make`
invocation.

## Running

The binary expects the path to an image, e.g. for the `cppcon-logo.png` image
that's already there:

```sh
$ LD_LIBRARY_PATH=/path/to/mkl/libs ./conv cppcon-logo.png
```

It then generates an image called `mkl-out.png`.
