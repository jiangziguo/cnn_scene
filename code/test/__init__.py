import tensorflow as tf

var = tf.Variable(tf.truncated_normal([5, 5], stddev=0.1))
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(var.eval(sess))

a = [1, 2, 3, 4]
print(a[1:a.__len__()])
print(a[1: 4])

b = [[1, 2, 3], [4, 5, 6]]
b[1] = a
print(b)