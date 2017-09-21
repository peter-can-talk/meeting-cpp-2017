import os
import tensorflow as tf

_kernel_path = os.environ.get('CPP_CON_KERNEL_PATH', './kernel.so')
_module = tf.load_op_library(_kernel_path)

cpp_con_sigmoid = _module.cpp_con_sigmoid
