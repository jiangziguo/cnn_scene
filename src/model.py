import os
import tensorflow as tf

from src import data_util


class Model(object):
    def __init__(self, config):
        self.config = config

    # 定义一个函数，用于初始化所有的权值
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 定义一个函数，用于初始化所有的偏置项 b
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 定义一个函数，用于构建卷积层
    def conv2d(self, x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')

    # 定义一个函数，用于构建池化层
    def max_pool(self, x, pool_height):
        return tf.nn.max_pool(x, ksize=[1, pool_height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

    def train(self):
        config = self.config
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, config.sentence_length, config.word2vec_dimension])
        label_data = tf.placeholder(dtype=tf.float32, shape=[None, config.scene_num])
        drop_out_prob = tf.placeholder("float")

        # 构建网络
        # 转换输入数据shape,以便于用于网络中
        with tf.name_scope("input"):
            x_word = tf.reshape(input_data, [-1, config.sentence_length, config.word2vec_dimension, 1])

        with tf.name_scope("conv_1"):
            W_conv1 = self.weight_variable(
                [config.filter_height_1, config.word2vec_dimension, 1, config.filter_output_channel])
            b_conv1 = self.bias_variable([config.filter_output_channel])
            h_conv1 = tf.nn.relu(self.conv2d(x_word, W_conv1) + b_conv1)  # 第一个卷积层
            tf.summary.histogram('weight', W_conv1)
            tf.summary.histogram('bias', b_conv1)
            tf.summary.histogram('activations', h_conv1)
        with tf.name_scope("max_pool_1"):
            h_pool1 = self.max_pool(h_conv1, config.sentence_length - config.filter_height_1 + 1)  # 第一个池化层

        with tf.name_scope("conv_2"):
            W_conv2 = self.weight_variable(
                [config.filter_height_2, config.word2vec_dimension, 1, config.filter_output_channel])
            b_conv2 = self.bias_variable([config.filter_output_channel])
            h_conv2 = tf.nn.relu(self.conv2d(x_word, W_conv2) + b_conv2)  # 第二个卷积层
            tf.summary.histogram('weight', W_conv2)
            tf.summary.histogram('bias', b_conv2)
            tf.summary.histogram('activations', h_conv2)
        with tf.name_scope("max_pool_2"):
            h_pool2 = self.max_pool(h_conv2, config.sentence_length - config.filter_height_2 + 1)  # 第一个池化层

        with tf.name_scope("conv_3"):
            W_conv3 = self.weight_variable(
                [config.filter_height_3, config.word2vec_dimension, 1, config.filter_output_channel])
            b_conv3 = self.bias_variable([config.filter_output_channel])
            h_conv3 = tf.nn.relu(self.conv2d(x_word, W_conv3) + b_conv3)  # 第三个卷积层
            tf.summary.histogram('weight', W_conv3)
            tf.summary.histogram('bias', b_conv3)
            tf.summary.histogram('activations', h_conv3)
        with tf.name_scope("max_pool_3"):
            h_pool3 = self.max_pool(h_conv3, config.sentence_length - config.filter_height_3 + 1)  # 第一个池化层

        with tf.name_scope("concat"):
            contact_all_pool = tf.concat([h_pool1, h_pool2, h_pool3], 3)

        with tf.name_scope("fc_1"):
            W_fc1 = self.weight_variable([300, config.hidden_unit])
            b_fc1 = self.bias_variable([config.hidden_unit])
            h_pool_flat = tf.reshape(contact_all_pool, [-1, 300])  # reshape成向量
            h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)  # 第一个全连接层
            tf.summary.histogram('weight', W_fc1)
            tf.summary.histogram('bias', b_fc1)
            tf.summary.histogram('activations', h_fc1)

        with tf.name_scope("dropout"):
            h_fc1_drop = tf.nn.dropout(h_fc1, drop_out_prob)  # dropout层

        with tf.name_scope("fc_2"):
            W_fc2 = self.weight_variable([config.hidden_unit, config.scene_num])
            b_fc2 = self.bias_variable([config.scene_num])
            tf.summary.histogram('weight', W_fc1)
            tf.summary.histogram('bias', b_fc1)

        with tf.name_scope("y_predict"):
            y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层
            tf.summary.histogram('y_predict', y_predict)

        with tf.name_scope('loss'):
            cross_entropy = -tf.reduce_sum(label_data * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))  # 交叉熵
            tf.summary.scalar('loss', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(label_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算
            tf.summary.scalar('train_accuracy', accuracy)

        merged = tf.summary.merge_all()
        word2vec_map = data_util.get_word2vec_map(config.word2vec_data_dir)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(config.train_log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for i in range(1, config.train_step):
                batch = data_util.get_batch_data(config, word2vec_map)
                if i % config.accuracy_step == 0:
                    # eval_accuracy = accuracy.eval(
                    #     feed_dict={input_data: batch[0], label_data: batch[1], drop_out_prob: 1.0})
                    merge_summary, eval_accuracy = sess.run([merged, accuracy], feed_dict={input_data: batch[0], label_data: batch[1], drop_out_prob: 1.0})
                    # accuracy_sum = tf.Summary(
                    #     value=[tf.Summary.Value(tag="model/accuracy", simple_value=eval_accuracy), ])
                    # print('step', i, 'training accuracy', eval_accuracy)
                    writer.add_summary(merge_summary, global_step=i)
                    writer.flush()
                    continue
                # train_step.run(feed_dict={input_data: batch[0], label_data: batch[1], drop_out_prob: 0.5})
                merge_summary, train_accuracy = sess.run([merged, train_step],
                                                        feed_dict={input_data: batch[0], label_data: batch[1],
                                                                   drop_out_prob: 0.5})
                writer.add_summary(merge_summary, global_step=i)
                writer.flush()
                model_file_name = os.path.join(config.model_save_dir, "model.ckpt")
                saver.save(sess, model_file_name)

    def test(self):
        config = self.config
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, config.sentence_length, config.word2vec_dimension])
        label_data = tf.placeholder(dtype=tf.float32, shape=[None, config.scene_num])
        drop_out_prob = tf.placeholder("float")

        # 构建网络
        # 转换输入数据shape,以便于用于网络中
        with tf.name_scope("input"):
            x_word = tf.reshape(input_data, [-1, config.sentence_length, config.word2vec_dimension, 1])

        with tf.name_scope("conv_1"):
            W_conv1 = self.weight_variable(
                [config.filter_height_1, config.word2vec_dimension, 1, config.filter_output_channel])
            b_conv1 = self.bias_variable([config.filter_output_channel])
            h_conv1 = tf.nn.relu(self.conv2d(x_word, W_conv1) + b_conv1)  # 第一个卷积层
            tf.summary.histogram('weight', W_conv1)
            tf.summary.histogram('bias', b_conv1)
            tf.summary.histogram('activations', h_conv1)
        with tf.name_scope("max_pool_1"):
            h_pool1 = self.max_pool(h_conv1, config.sentence_length - config.filter_height_1 + 1)  # 第一个池化层

        with tf.name_scope("conv_2"):
            W_conv2 = self.weight_variable(
                [config.filter_height_2, config.word2vec_dimension, 1, config.filter_output_channel])
            b_conv2 = self.bias_variable([config.filter_output_channel])
            h_conv2 = tf.nn.relu(self.conv2d(x_word, W_conv2) + b_conv2)  # 第二个卷积层
            tf.summary.histogram('weight', W_conv2)
            tf.summary.histogram('bias', b_conv2)
            tf.summary.histogram('activations', h_conv2)
        with tf.name_scope("max_pool_2"):
            h_pool2 = self.max_pool(h_conv2, config.sentence_length - config.filter_height_2 + 1)  # 第一个池化层

        with tf.name_scope("conv_3"):
            W_conv3 = self.weight_variable(
                [config.filter_height_3, config.word2vec_dimension, 1, config.filter_output_channel])
            b_conv3 = self.bias_variable([config.filter_output_channel])
            h_conv3 = tf.nn.relu(self.conv2d(x_word, W_conv3) + b_conv3)  # 第三个卷积层
            tf.summary.histogram('weight', W_conv3)
            tf.summary.histogram('bias', b_conv3)
            tf.summary.histogram('activations', h_conv3)
        with tf.name_scope("max_pool_3"):
            h_pool3 = self.max_pool(h_conv3, config.sentence_length - config.filter_height_3 + 1)  # 第一个池化层

        with tf.name_scope("concat"):
            contact_all_pool = tf.concat([h_pool1, h_pool2, h_pool3], 3)

        with tf.name_scope("fc_1"):
            W_fc1 = self.weight_variable([300, config.hidden_unit])
            b_fc1 = self.bias_variable([config.hidden_unit])
            h_pool_flat = tf.reshape(contact_all_pool, [-1, 300])  # reshape成向量
            h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)  # 第一个全连接层
            tf.summary.histogram('weight', W_fc1)
            tf.summary.histogram('bias', b_fc1)
            tf.summary.histogram('activations', h_fc1)

        with tf.name_scope("dropout"):
            h_fc1_drop = tf.nn.dropout(h_fc1, drop_out_prob)  # dropout层

        with tf.name_scope("fc_2"):
            W_fc2 = self.weight_variable([config.hidden_unit, config.scene_num])
            b_fc2 = self.bias_variable([config.scene_num])
            tf.summary.histogram('weight', W_fc1)
            tf.summary.histogram('bias', b_fc1)

        with tf.name_scope("y_predict"):
            y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层
            tf.summary.histogram('y_predict', y_predict)

        with tf.name_scope('loss'):
            cross_entropy = -tf.reduce_sum(label_data * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))  # 交叉熵
            tf.summary.scalar('loss', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(label_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算
            tf.summary.scalar('train_accuracy', accuracy)

        merged = tf.summary.merge_all()
        word2vec_map = data_util.get_word2vec_map(config.word2vec_data_dir)

        # with tf.Session() as sess:
        #     writer = tf.summary.FileWriter(config.test_log_dir, sess.graph)
        #     sess.run(tf.global_variables_initializer())
        #     saver = tf.train.Saver()
        #     saver.restore(sess, tf.train.latest_checkpoint(config.model_save_dir))
        #     for i in range(1, config.test_step):
        #         batch = data_util.get_batch_data(config, word2vec_map)
        #         eval_accuracy = accuracy.eval(
        #             feed_dict={input_data: batch[0], label_data: batch[1], drop_out_prob: 1.0})
        #         accuracy_sum = tf.Summary(
        #             value=[tf.Summary.Value(tag="model/accuracy", simple_value=eval_accuracy), ])
        #         print('step', i, 'test accuracy', eval_accuracy)
        #         writer.add_summary(accuracy_sum)
        #         writer.flush()

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(config.test_log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.model_save_dir))
            all_accuracy_sum = 0.0
            for item in data_util.scene_to_num.items():
                x, y = data_util.get_test_data(config, word2vec_map, item)
                eval_accuracy = accuracy.eval(
                    feed_dict={input_data: x, label_data: y, drop_out_prob: 1.0})
                accuracy_sum = tf.Summary(
                    value=[tf.Summary.Value(tag="model/accuracy", simple_value=eval_accuracy), ])
                all_accuracy_sum += eval_accuracy
                print('scene', item[1], 'test accuracy: ', eval_accuracy)
                writer.add_summary(accuracy_sum)
                writer.flush()
            print('all accuracy', all_accuracy_sum/14)
