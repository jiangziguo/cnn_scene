import os
import tensorflow as tf
from src.model import Model

flags = tf.flags

flags.DEFINE_string("mode", "train", "启动模式 train/debug/test")

train_data_dir = "E:\scene\\train\\k1\\"
test_data_dir = "E:\scene\\test\\k5\\"
word2vec_data_dir = "D:\hifive\HanLP\data\\test\word2vec_ikaNoDic.txt"
model_save_dir = "E:\scene\model\k1"
train_log_dir = "E:\scene\log\\train\k1"
test_log_dir = "E:\scene\log\\test\k5"

flags.DEFINE_string("train_data_dir", train_data_dir, "训练数据文件目录")
flags.DEFINE_string("test_data_dir", test_data_dir, "测试数据文件目录")
flags.DEFINE_string("word2vec_data_dir", word2vec_data_dir, "词向量文件路径")
flags.DEFINE_string("model_save_dir", model_save_dir, "模型保存路径")
flags.DEFINE_string("train_log_dir", train_log_dir, "训练日志文件目录")
flags.DEFINE_string("test_log_dir", test_log_dir, "测试日志文件目录")

flags.DEFINE_integer("train_step", 50000, "训练次数")
flags.DEFINE_integer("test_step", 2000, "测试次数")
flags.DEFINE_integer("accuracy_step", 5, "验证间隔")

flags.DEFINE_integer("scene_num", 14, "场景数量")
flags.DEFINE_integer("batch_size", 140, "数据分批大小")
flags.DEFINE_integer("train_keep_prob", 0.5, "训练时保留概率")
flags.DEFINE_integer("accuracy_keep_prob", 1.0, "验证时保留概率")
flags.DEFINE_integer("word2vec_dimension", 200, "词向量维度")
flags.DEFINE_integer("sentence_length", 100, "句子长度")
flags.DEFINE_integer("filter_output_channel", 100, "卷积输出通道大小")
flags.DEFINE_integer("filter_height_1", 3, "第1个卷机器的高度")
flags.DEFINE_integer("filter_height_2", 4, "第2个卷机器的高度")
flags.DEFINE_integer("filter_height_3", 5, "第3个卷机器的高度")
flags.DEFINE_integer("hidden_unit", 1024, "隐藏神经元数量")


def main(_):
    config = flags.FLAGS
    if config.mode == "train":
        Model(config).train()
    if config.mode == "test":
        Model(config).test()


if __name__ == "__main__":
    tf.app.run()
