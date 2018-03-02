# scene_to_num = {"爱情": 0,
#                 "熬夜": 1,
#                 "比赛": 2,
#                 "工作": 3,
#                 "婚姻": 4,
#                 "考试": 5,
#                 "旅游": 6,
#                 "梦想": 7,
#                 "人生": 8,
#                 "散步": 9,
#                 "童年": 10,
#                 "学校": 11,
#                 "演唱会": 12,
#                 "其他": 13}

scene_to_num = {"学校": 11}


def load_data(file_name):
    """
    加载数据,并将分词后的句子转换为字典：key=自增整数序号，value=句子对应的词列表
    :param file_name:
    :return:
    """
    sentence_map = {}
    i = 0
    with open(file_name, 'rb') as file:
        for line in file:
            sentence_map[i] = [word for word in line.split(' ')]
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
    with open(file_name, 'rb') as file:
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
        sentence_vec[i] = word_vec_map[words[i]]
    return sentence_vec


def get_one_scene_data(config, file_name):
    """
    获取训练数据
    :param config:
    :param file_name:
    :return:
    """
    all_vec = []
    label_vector = [0 for i in range(config.scene_num)]
    label_vector[scene_to_num[file_name]] = 1
    sentence_map = load_data(file_name)
    vec_map = get_word2vec_map(config.word2vec_file)
    for key, value in sentence_map.items():
        all_vec[int(key)] = get_sentence_vec(value, vec_map, config.sentence_vector_length,
                                             config.word2vec_dimension)
    return all_vec, label_vector


def get_train_data(config):
    """
    获取训练数据
    :param config:
    :return:
    """
    all_data = {}
    for key in scene_to_num.keys():
        all_data[key] = get_one_scene_data(config, key)
    return all_data
