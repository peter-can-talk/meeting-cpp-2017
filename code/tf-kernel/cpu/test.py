import tensorflow as tf
import cpp_con_sigmoid

x = tf.constant([[1.0, 0.0], [0.0, -1.0]])
y = cpp_con_sigmoid.cpp_con_sigmoid(x)

with tf.Session() as session:
    print(session.run(y))
