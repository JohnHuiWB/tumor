#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : test_data_pkg.py
# @Author: JohnHuiWB
# @Date  : 2018/3/31 0031
# @Desc  : 
# @Contact : huiwenbin199822@gmail.com 
# @Software : PyCharm

import tensorflow as tf

if __name__ == '__main__':
    import tumor.data as data
    image_batch, label_batch = data.load()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
        # 启动QueueRunner, 此时文件名队列已经进队。
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        image, label = sess.run([image_batch, label_batch])
        print(image.shape)
        coord.request_stop()
        coord.join(threads)
