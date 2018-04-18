

import numpy as np
import tensorflow as tf
import cv2
import glob
from os import path
FILENAME = 'train.tfrecords'
FILENAME_V ='validation.tfrecords'
FILENAME_T = 'test.tfrecords'
NUM_ALL = 4000
NUM_TRAIN_RATIO = 0.6
NUM_VALIDATION_RATIO = 0.1
NUM_TEST_RATIO = 0.3
NUM_TRAIN = int(NUM_ALL * NUM_TRAIN_RATIO)
NUM_VALIDATION = int(NUM_ALL * NUM_VALIDATION_RATIO)
NUM_TEST = int(NUM_ALL * NUM_TEST_RATIO)
def _load_data():
    """
    读取data目录中的所有数据，并返回
    
    """
    image_path = ( 'data/Image/*')
    images = glob.glob(image_path)
    labels = []
    for x in images:
        if(int(x.split('.')[-2].split('\IM')[-1])<=2000):
            labels.append(1)
        else:
            labels.append(0)
    

    paths = [[i, l] for i, l in zip(images, labels)]
    np.random.shuffle(paths)
    paths_train = paths[:NUM_TRAIN]
    paths_validation = paths[NUM_TRAIN:NUM_TRAIN+NUM_VALIDATION]
    paths_test = paths[NUM_TRAIN+NUM_VALIDATION:]

    print('Reading train data...')
    xs = []
    ys = []
    for x_path, y in paths_train:
        x = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x[100:400],(224,224))
        xs.append(x)
        ys.append(y)
    print('Done')

    print('Reading validation data...')
    xs_v = []
    ys_v = []
    for x_path, y in paths_validation:
        x = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x[100:400],(224,224))
        xs_v.append(x)
        ys_v.append(y)
    print('Done')

    print('Reading test data...')
    xs_t = []
    ys_t = []
    for x_path, y in paths_test:
        x = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x[100:400],(224,224))
        xs_t.append(x)
        ys_t.append(y)
    print('Done')

    return xs, ys, xs_v, ys_v, xs_t, ys_t


def _create_and_save(xs, ys, filename):
    """
    将数据保存为TFRecord文件
    :return:
    """
    # 创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)

    print('Creating data')

    for i in range(len(xs)):
        # 将图像矩阵转化成为一个字符串
        image_raw = xs[i].tostring()
        label = ys[i]
        # 将一个样例转化为Example Protocol Buffer，并将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            'label_raw':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        # 将一个Example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()

    print('Done')


def is_exists():
    if not path.exists(FILENAME) or not path.exists(FILENAME_V) or not path.exists(FILENAME_T):
        print('File does not exist.')
        xs, ys, xs_v, ys_v, xs_t, ys_t = _load_data()
        _create_and_save(xs, ys, FILENAME)
        _create_and_save(xs_v, ys_v, FILENAME_V)
        _create_and_save(xs_t, ys_t, FILENAME_T)


def generate_arrays_from_file(filename, batch_size=1):
    """
    迭代器方法generate_arrays_from_file
    :param filename:
    :return:
    """
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
            'label_raw': tf.FixedLenFeature([],tf.int64)
        }
    )

    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    images = tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8), [224,224,1])
    labels = features['label_raw']

    sess = tf.Session()
    # 启动多线程处理输入数据
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 每次运行可以读取TFRecord文件中的一个样例
    while 1:
        xs, ys = [], []
        for i in range(batch_size):
            # 读取一组数据
            x, y = sess.run([images, labels])
            xs.append(x)
            ys.append(y)
            
        # 转换为数组，归一化
        xs= (np.array(xs, dtype=np.float32))/255.
        ys=np.array(ys)
        yield (xs, ys)


if __name__ == '__main__':
    is_exists()
    
    


