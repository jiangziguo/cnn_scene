import random

import tensorflow as tf

# var = tf.Variable(tf.truncated_normal([5, 5], stddev=0.1))
# a = [[1, 2], [2, 4]]
# b = [[3, 5], [4, 6]]
# ab = tf.concat([a, b], 0)
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# out = sess.run(ab)
# tf.Print(ab)
# print(out.eval(sess))
# print(sess.run(ab))
# print(var.eval(sess))
#
# a = [1, 2, 3, 4]
# print(a[1:a.__len__()])
# print(a[1: 4])
#
# b = [[1, 2, 3], [4, 5, 6]]
# b[1] = a
# print(b)
#
for i in range(3000):
    print(random.randint(0, 50))
#
# a = []
# for i in range(100):
#     a.append(i)
# i = 0
# for num in a:
#     if i == num:
#         i += 1
#         continue
#     print(num)
