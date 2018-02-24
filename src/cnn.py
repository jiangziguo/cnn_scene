import random

import tensorflow as tf

scene_to_num = {"学校": 11}


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
            sentence_map[i] = words[0:words.__len__()-1]
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
    sentence_vec = [[0 for i in range(word2vec_dimension)] for i in range(vec_length)]
    if vec_length < list(words).__len__():
        return 'vec_length is too small'
    for i in range(list(words).__len__()):
        if not word_vec_map.__contains__(words[i]):
            sentence_vec[i] = [0 for i in range(word2vec_dimension)]
            continue
        sentence_vec[i] = word_vec_map[words[i]]
    return sentence_vec


def get_one_scene_data(file_name):
    """
    获取训练数据
    :param config:
    :param file_name:
    :return:
    """
    all_vec = []
    label_vector = [0 for i in range(14)]
    label_vector[scene_to_num[file_name]] = 1
    sentence_map = load_data('E:\场景\场景评论分词\\'+file_name+'.txt')
    vec_map = get_word2vec_map("D:\hifive\HanLP\data\\test\word2vec_ikaNoDic.txt")
    for key, value in sentence_map.items():
        all_vec.append(get_sentence_vec(value, vec_map, 2000, 200))
    return all_vec, label_vector


def get_train_data():
    """
    获取训练数据
    :param config:
    :return:
    """
    all_data = {}
    for key in scene_to_num.keys():
        all_data[key] = get_one_scene_data(key)
    return all_data


def get_next_batch(data, data_num):
    vec = data['学校'][0]
    label = data['学校'][1]
    x = []
    y = [label for i in range(data_num)]
    size = list(data['学校'][0]).__len__() - 1
    for i in range(data_num):
        index = random.randint(0, size)
        x[i] = vec[index]
    return x, y


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
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


input_data = tf.placeholder(dtype=tf.float32, shape=[None, 2000*200])
label_data = tf.placeholder(dtype=tf.float32, shape=[None, 14])
drop_out_prob = tf.placeholder("float")

# 构建网络
x_word = tf.reshape(input_data, [-1, 2000, 200, 1])  # 转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([5, 200, 1, 600])
b_conv1 = bias_variable([600])
h_conv1 = tf.nn.relu(conv2d(x_word, W_conv1) + b_conv1)  # 第一个卷积层
h_pool1 = max_pool(h_conv1)  # 第一个池化层

W_conv2 = weight_variable([1, 3, 600, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二个卷积层
h_pool2 = max_pool(h_conv2)  # 第二个池化层

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层

h_fc1_drop = tf.nn.dropout(h_fc1, drop_out_prob)  # dropout层

W_fc2 = weight_variable([1024, 14])
b_fc2 = bias_variable([14])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层

cross_entropy = -tf.reduce_sum(label_data * tf.log(y_predict))  # 交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(label_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

data = get_train_data()
for i in range(200):
    print("train time: "+i)
    batch = get_next_batch(data, 10)
    if i % 10 == 0:  # 训练100次，验证一次
        train_acc = accuracy.eval(feed_dict={input_data: batch[0], label_data: batch[1], drop_out_prob: 1.0})
        print('step', i, 'training accuracy', train_acc)
    train_step.run(feed_dict={input_data: batch[0], label_data: batch[1], drop_out_prob: 0.5})

# test_acc = accuracy.eval(feed_dict={input_data: mnist.test.images, label_data: mnist.test.labels, drop_out_prob: 1.0})
# print("test accuracy", test_acc)
