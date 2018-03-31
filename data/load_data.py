#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : load_data.py
# @Author: JohnHuiWB
# @Date  : 2018/3/31 0031
# @Desc  :
# @Contact : huiwenbin199822@gmail.com
# @Software : PyCharm

import cv2
import numpy as np
from os import path
import tensorflow as tf


FILENAME = path.join(
    path.dirname(
        path.realpath(__file__)),
    'dataset.tfrecords')
NUM = 2200
PIXELS = (512, 512)


def _load_data():
    """
    读取data目录中的所有数据，并返回
    :return:
    """

    print('读取数据中')

    xs = []
    ys = []
    for i in range(NUM):
        x_path = path.join(path.dirname(path.realpath(__file__)),
                           'Label/Label' + str(i + 1) + '.png')
        x = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
        xs.append(x)
        y_path = path.join(path.dirname(path.realpath(
            __file__)), 'Image/IM' + str(i + 1) + '.png')
        y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        ys.append(y)

    print('读取完成')

    return xs, ys


def _bytes_feature(value):
    """
    生成字符串型数据
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_and_save(filename=FILENAME):
    """
    将数据保存为TFRecord文件
    :return:
    """
    # 创建一个writer来写TFRecord文件
    xs, ys = _load_data()
    writer = tf.python_io.TFRecordWriter(filename)

    print('处理数据中')

    for i in range(len(xs)):
        # 将图像矩阵转化成为一个字符串
        image_raw = xs[i].tostring()
        label_raw = ys[i].tostring()
        # 将一个样例转化为Example Protocol Buffer，并将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'label_raw': _bytes_feature(label_raw)
        }))
        # 将一个Example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()

    print('处理完成')


def load(filename=FILENAME):
    """

    :param filename:
    :return:
    """

    print('加载数据中')

    # 创建一个reader来读取TFRecord文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表，
    filename_queue = tf.train.string_input_producer([filename])
    # 从文件中读出一个样例，也可以使用read_up_to函数一次性读出多个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析读入的一个样例，如果需要解析多个样例，可以使用parse_example函数
    features = tf.parse_single_example(
        serialized_example,
        {
            # TensorFlow提供两种不同的属性解析方法。一种方法是tf.FixedLenFeature,
            # 这种方法解析的结果为一个Tensor。另一种方法是tf.VarLenFeature，这种方法
            # 得到的解析结果为SparseTensor，用于处理稀疏数据。这里解析数据的格式需要和
            # 上面程序写入数据的格式一致。
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([], tf.string)
        }
    )

    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    images = tf.decode_raw(features['image_raw'], tf.uint8)
    labels = tf.decode_raw(features['label_raw'], tf.uint8)

    sess = tf.Session()
    # 启动多线程处理输入数据。
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 每次运行可以读取TFRecord文件中的一个样例。当所有样例读取完之后，在此样例中程序
    # 会在重头读取。
    xs = []
    ys = []
    for i in range(NUM):
        # 读取一组x和y
        x, y = sess.run([images, labels])
        # reshape
        xs.append(np.reshape(x, PIXELS))
        ys.append(np.reshape(x, PIXELS))

    print('加载完成')

    return xs, ys


if __name__ == '__main__':
    # create_and_save()
    # load()
    pass
