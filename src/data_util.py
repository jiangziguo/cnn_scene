import random

scene_to_num = {0: "爱情",
                1: "熬夜",
                2: "比赛",
                3: "工作",
                4: "婚姻",
                5: "考试",
                6: "旅游",
                7: "梦想",
                8: "人生",
                9: "散步",
                10: "童年",
                11: "学校",
                12: "演唱会",
                13: "其他"
                }


def load_data(file_name, data_total_num):
    """
    加载数据,并将分词后的句子转换为字典,从中随机选择10条数据返回：key=自增整数序号，value=句子对应的词列表
    :param file_name:场景文件名
    :param data_total_num:每个场景返回数据条数
    :return:
    """
    sentence_map = {}
    sentences = []
    i = 0
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            words = [word for word in line.split(' ')]
            # 去掉最后的\n符号
            sentence_map[i] = words[0:words.__len__() - 1]
            i += 1
    file.close()
    for j in range(data_total_num):
        sentences.append(sentence_map[random.randint(0, i - 1)])
    return sentences


def get_word2vec_map(file_name):
    """
    将词向量转换为Map,以方便创建句子对应的矩阵，key=词，value=词向量（1X200）
    :param file_name:词向量文件位置
    :return:
    """
    word2vec_map = {}
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            line_split = line.split(" ")
            word = line_split[0]
            word_vec_list = [float(i) for i in line_split[1:line_split.__len__()]]
            word2vec_map[word] = word_vec_list
    file.close()
    return word2vec_map


def get_sentence_vec(words, word2vec_map, config):
    """
    将分词后的句子转换为长度为sentence_length的矩阵（长度不够，用零补全；句子太长，截断）
    :param words:句子分词之后的map，key代表词在句子中的位置
    :param word2vec_map:词向量map
    :param config: 配置信息
    :return:
    """
    sentence_vec = []
    for i in range(config.sentence_length):
        if i > list(words).__len__() - 1 or not word2vec_map.__contains__(words[i]):
            sentence_vec.append([0 for i in range(config.word2vec_dimension)])
            continue
        sentence_vec.append(word2vec_map[words[i]])
    return sentence_vec


def get_one_scene_data(item, word2vec_map, scene_bath_size, config):
    """
    获取一个场景的数据数据
    :param word2vec_map: 词向量map
    :param item: value=文件名，key=文件名对应的序号
    :param scene_bath_size: 每一个场景获取的数据条数
    :param config: 配置信息
    :return:
    """
    all_vec = []
    label_vector = [0 for i in range(config.scene_num)]
    label_vector[item[0]] = 1

    if config.mode == 'test':
        sentence_map = load_data(config.test_data_dir + item[1] + '.txt', scene_bath_size)
    else:
        sentence_map = load_data(config.train_data_dir + item[1] + '.txt', scene_bath_size)

    for sentence in sentence_map:
        all_vec.append(get_sentence_vec(sentence, word2vec_map, config))
    return all_vec, label_vector


def get_batch_data(config, word2vec_map):
    """
    获取批量数据
    :param config: 配置信息
    :param word2vec_map: 词向量map
    :return:
    """
    batch_data = {}
    scene_batch_size = int(config.batch_size / config.scene_num)
    for item in scene_to_num.items():
        if item[0] == config.scene_num:
            last_scene_batch_size = scene_batch_size + config.batch_size % config.scene_num
            batch_data[item[0]] = get_one_scene_data(item, word2vec_map, last_scene_batch_size, config)
            continue
        batch_data[item[0]] = get_one_scene_data(item, word2vec_map, scene_batch_size, config)

    vec = []
    label = []
    for scene_vec in batch_data.values():
        for x in scene_vec[0]:
            vec.append(x)
            label.append(scene_vec[1])
    return vec, label


def get_scene_data_size(file_name):
    """
    获取测试数据数量
    :param file_name:
    :return:
    """
    total_num = 0
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            total_num += 1
    file.close()
    return total_num


def get_test_data(config, word2vec_map, item):
    """
    获取场景的测试数据
    :param config:
    :param word2vec_map:
    :param item: 场景
    :return:
    """
    sentence_batch_size = get_scene_data_size(config.test_data_dir + item[1] + '.txt')
    all_sentence_vector, label_vec = get_one_scene_data(item, word2vec_map, sentence_batch_size, config)
    all_label_vector = [label_vec for i in range(sentence_batch_size)]
    return all_sentence_vector, all_label_vector
