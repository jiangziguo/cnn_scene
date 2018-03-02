import random

import os
import tensorflow as tf

scene_to_num = {0: "爱情",
                1: "熬夜",
                2: "比赛",
                3: "工作",
                4: "婚姻"
                # 5: "考试",
                # 6: "旅游",
                # 7: "梦想",
                # 8: "人生",
                # 9: "散步",
                # 10: "童年",
                # 11: "学校",
                # 12: "演唱会",
                # 13: "其他"
                }


def load_data(file_name):
    """
    加载数据,并将分词后的句子转换为字典：key=自增整数序号，value=句子对应的词列表
    :param file_name:
    :return:
    """
    sentence_map = {}
    i = 0
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            words = [word for word in line.split(' ')]
            # 去掉最后的\n符号
            sentence_map[i] = words[0:words.__len__() - 1]
            i += 1
    file.close()
    return sentence_map


def get_word2vec_map(file_name):
    """
    将词向量转换为Map,以方便创建句子对应的矩阵，key=词，value=词向量（1X200）
    :param file_name:
    :return:
    """
    vec_map = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line_split = line.split(" ")
            word = line_split[0]
            word_vec_list = [float(i) for i in line_split[1:line_split.__len__()]]
            vec_map[word] = word_vec_list
    file.close()
    return vec_map


def get_sentence_vec(words, word_vec_map, vec_length, word2vec_dimension):
    """
    将分词后的句子转换为长度为vec_length的矩阵，长度不够的，用零补全
    :param words:
    :param word_vec_map:
    :param vec_length:
    :param word2vec_dimension:
    :return:
    """
    sentence_vec = []

    for i in range(vec_length):
        if i > list(words).__len__() - 1 or not word_vec_map.__contains__(words[i]):
            sentence_vec.append([0 for i in range(word2vec_dimension)])
            continue
        sentence_vec.append(word_vec_map[words[i]])
    return sentence_vec


def get_one_scene_data(item, vec_map):
    """
    获取一个场景的数据数据
    :param vec_map: 词向量map
    :param item: value=文件名，key=文件名对应的序号
    :return:
    """
    all_vec = []
    label_vector = [0 for i in range(5)]
    label_vector[item[0]] = 1
    sentence_map = load_data('E:\场景\场景评论分词\\' + item[1] + '.txt')
    # vec_map = get_word2vec_map("D:\hifive\HanLP\data\\test\word2vec_ikaNoDic.txt")
    for key, value in sentence_map.items():
        all_vec.append(get_sentence_vec(value, vec_map, 200, 200))
    return all_vec, label_vector


def get_train_data():
    """
    获取训练数据
    :param config:
    :return:
    """
    vec_map = get_word2vec_map("D:\hifive\HanLP\data\\test\word2vec_ikaNoDic.txt")
    all_data = {}
    for item in scene_to_num.items():
        all_data[item[0]] = get_one_scene_data(item, vec_map)
    return all_data


# def get_next_batch(data, data_num):
#     vec = data['学校'][0]
#     label = data['学校'][1]
#     x = []
#     y = [label for i in range(data_num)]
#     size = list(data['学校'][0]).__len__() - 1
#     for i in range(data_num):
#         index = random.randint(0, size)
#         try:
#             x.append(vec[index])
#         except Exception as error:
#             print(index)
#     return x, y

def get_next_batch(data, batch_size):
    vec = []
    label = []
    for i in range(batch_size):
        scene_data = data[random.randint(0, 4)]
        sentence_vec = list(scene_data[0])
        one_sentence_vec = sentence_vec[random.randint(0, sentence_vec.__len__() - 1)]
        vec.append(one_sentence_vec)
        label.append(scene_data[1])
    return vec, label


# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 2, 1], padding='SAME')


input_data = tf.placeholder(dtype=tf.float32, shape=[None, 200, 200])
label_data = tf.placeholder(dtype=tf.float32, shape=[None, 5])
drop_out_prob = tf.placeholder("float")

# 构建网络
x_word = tf.reshape(input_data, [-1, 200, 200, 1])  # 转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([5, 200, 1, 128])
b_conv1 = bias_variable([128])
h_conv1 = tf.nn.relu(conv2d(x_word, W_conv1) + b_conv1)  # 第一个卷积层
h_pool1 = max_pool(h_conv1)  # 第一个池化层

W_conv2 = weight_variable([3, 1, 128, 256])
b_conv2 = bias_variable([256])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二个卷积层
h_pool2 = max_pool(h_conv2)  # 第二个池化层

W_conv3 = weight_variable([3, 1, 256, 512])
b_conv3 = bias_variable([512])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # 第三个卷积层
h_pool3 = max_pool(h_conv3)  # 第二个池化层

# W_fc1 = weight_variable([75 * 50 * 20, 1024])
W_fc1 = weight_variable([23 * 1 * 512, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 23 * 1 * 512])  # reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # 第一个全连接层

h_fc1_drop = tf.nn.dropout(h_fc1, drop_out_prob)  # dropout层

W_fc2 = weight_variable([1024, 5])
b_fc2 = bias_variable([5])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层

cross_entropy = -tf.reduce_sum(label_data * tf.log(tf.clip_by_value(y_predict, 1e-10, 1.0)))  # 交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(label_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

data = get_train_data()
print("data load success.")
for i in range(1, 10000):
    print("train time: " + str(i))
    batch = get_next_batch(data, 50)
    if i % 5 == 0:  # 训练100次，验证一次
        train_acc = accuracy.eval(feed_dict={input_data: batch[0], label_data: batch[1], drop_out_prob: 1.0})
        print('step', i, 'training accuracy', train_acc)
        continue
    # try:
    train_step.run(feed_dict={input_data: batch[0], label_data: batch[1], drop_out_prob: 0.5})
    # except Exception as error:
    #     print(input_data)
        # y = y_predict.eval(feed_dict={input_data: batch[0], label_data: batch[1], drop_out_prob: 1.0})
        # print('predict:' + str(y))

# test_acc = accuracy.eval(feed_dict={input_data: mnist.test.images, label_data: mnist.test.labels, drop_out_prob: 1.0})
# print("test accuracy", test_acc)
