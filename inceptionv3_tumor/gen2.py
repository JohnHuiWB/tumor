import numpy as np
import tensorflow as tf
import cv2
import glob
from os import path
from tqdm import tqdm
FILENAME = 'train.tfrecords'
FILENAME_V ='validation.tfrecords'
FILENAME_T = 'test.tfrecords'
NUM_ALL = 2200
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
    images=[]
    images2=[]
    labels=[]
    for i in tqdm(range(1,2001)):
        images.append('../data/Image/IM'+str(i)+'.png')
        images2.append('../data/Label2/Label'+str(i)+'.png')
        pic=np.array(cv2.imread('../data/Label/Label'+str(i)+'.png'))
        if np.max(pic) == 255 :
            labels.append(1)
        else:
            labels.append(0)
    

    paths = [[i, l, k] for i, l ,k in zip(images, images2,labels)]
    np.random.shuffle(paths)
    paths_train = paths[:NUM_TRAIN]
    paths_validation = paths[NUM_TRAIN:NUM_TRAIN+NUM_VALIDATION]
    paths_test = paths[NUM_TRAIN+NUM_VALIDATION:]

    print('Reading train data...')
    xs1 = []
    xs2 = []
    ys = []
    for x_path,x_path2, y in paths_train:
        x1 = cv2.imread(x_path)
        x1 = cv2.resize(x1,(299,299))
        x2 = cv2.imread(x_path2,0)
        x2 = cv2.resize(x2,(299,299))
        xs1.append(x1)
        xs2.append(x2)
        ys.append(y)
    print('Done')

    print('Reading validation data...')
    xs1_v = []
    xs2_v = []
    ys_v = []
    for x_path,x_path2, y in paths_validation:
        x1 = cv2.imread(x_path)
        x1 = cv2.resize(x1,(299,299))
        x2 = cv2.imread(x_path2,0)
        x2 = cv2.resize(x2,(299,299))
        xs1_v.append(x1)
        xs2_v.append(x2)
        ys_v.append(y)
    print('Done')

    print('Reading test data...')
    xs1_t = []
    xs2_t = []
    ys_t = []
    for x_path,x_path2, y in paths_test:
        x1 = cv2.imread(x_path)
        x1 = cv2.resize(x1,(299,299))
        x2 = cv2.imread(x_path2,0)
        x2 = cv2.resize(x2,(299,299))
        xs1_t.append(x1)
        xs2_t.append(x2)
        ys_t.append(y)
    print('Done')

    return xs1,xs2, ys, xs1_v, xs2_v, ys_v, xs1_t,xs2_t, ys_t


def _create_and_save(xs1,xs2,ys, filename):
    """
    将数据保存为TFRecord文件
    :return:
    """
    # 创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)

    print('Creating data')

    for i in range(len(xs1)):
        # 将图像矩阵转化成为一个字符串
        image_raw = xs1[i].tostring()
        image_raw2 = xs2[i].tostring()
        label = ys[i]
        # 将一个样例转化为Example Protocol Buffer，并将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
            'image_raw2':tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw2])),
            'label_raw':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        # 将一个Example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()

    print('Done')


def is_exists():
    if not path.exists(FILENAME) or not path.exists(FILENAME_V) or not path.exists(FILENAME_T):
        print('File does not exist.')
        xs1, xs2 , ys, xs1_v, xs2_v, ys_v, xs1_t, xs2_t, ys_t = _load_data()
        _create_and_save(xs1, xs2,ys, FILENAME)
        _create_and_save(xs1_v, xs2_v,ys_v, FILENAME_V)
        _create_and_save(xs1_t, xs2_t,ys_t, FILENAME_T)


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
            'image_raw2': tf.FixedLenFeature([], tf.string),
            'label_raw': tf.FixedLenFeature([],tf.int64)
        }
    )

    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    images = tf.reshape(tf.decode_raw(features['image_raw'], tf.uint8), [299,299,3])
    images2 = tf.reshape(tf.decode_raw(features['image_raw2'], tf.uint8), [299,299,1])
    labels = features['label_raw']

    sess = tf.Session()
    # 启动多线程处理输入数据
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 每次运行可以读取TFRecord文件中的一个样例
    while 1:
        xs1, xs2, ys = [], [],[]
        for i in range(batch_size):
            # 读取一组数据
            x1,x2, y = sess.run([images,images2, labels])
            xs1.append(x1)
            xs2.append(x2)
            ys.append(y)
            
        # 转换为数组，归一化
        xs1= (np.array(xs1, dtype=np.float32))/255.
        xs2= (np.array(xs2, dtype=np.float32))/255.
        ys=np.array(ys)
        yield ([xs1,xs2],[ys])


if __name__ == '__main__':
    is_exists()
     
    
    


