# tf-kernel

Example of creating a custom TensorFlow operator.

## Building

Prerequisites:

1. Install Python (preferrably 3) and `pip install -r requirements.txt` found in the `code/` root folder,

Then `make`.

## Running

The `test.py` script gives an example of loading the custom op in Python.

The `cpu/` folder can be run exclusively on a CPU, the `cpu+gpu` stuff requires
a GPU to run (at least to run `test.py`, the kernel itself is available on
both).
