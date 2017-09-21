import tensorflow as tf
import cpp_con_sigmoid

with tf.device('/gpu:0'):
    x = tf.constant([[5.0, 0.0], [0.0, -5.0]])
    y = cpp_con_sigmoid.cpp_con_sigmoid(x)

config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=False)
config.graph_options.optimizer_options.opt_level = -1

with tf.Session(config=config) as session:
    print(session.run(y))
