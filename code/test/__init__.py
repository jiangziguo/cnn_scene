import tensorflow as tf

var = tf.Variable(tf.truncated_normal([5, 5], stddev=0.1))
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(var.eval(sess))
