#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : test_unet_unet.py
# @Author: JohnHuiWB
# @Date  : 2018/3/31 0031
# @Desc  : 
# @Contact : huiwenbin199822@gmail.com 
# @Software : PyCharm


from tumor.unet.unet import *
import numpy as np
import cv2


def test_predict(u):
    from tumor.seg_data import generate_arrays_from_file, FILENAME_T
    g = generate_arrays_from_file(FILENAME_T)
    x, y = g.__next__()
    x, y = g.__next__()
    x, y = g.__next__()
    x, y = g.__next__()



    r = u.predict(x)

    # 二值化
    # r[r >= 0.5] = 1
    # r[r < 0.5] = 0

    # 还原为灰度级
    x *= 255
    y *= 255
    r *= 255

    # 还原为uint8类型
    x = x.astype('uint8')
    y = y.astype('uint8')
    r = r.astype('uint8')

    # 删掉维数为1的维度，保留(512, 512)的矩阵
    x = np.squeeze(x)
    y = np.squeeze(y)
    r = np.squeeze(r)

    cv2.imshow('x', x)
    cv2.imshow('y', y)
    cv2.imshow('r', r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_draw_model(u):
    u.draw_model()


def test_train(u, batch_size, samples_per_epoch, epochs):
    u.train(batch_size=batch_size,
            samples_per_epoch=samples_per_epoch,
            epochs=epochs)


def test_continue_train(u, batch_size, samples_per_epoch, epochs):
    u.continue_train(batch_size=batch_size,
            samples_per_epoch=samples_per_epoch,
            epochs=epochs)

def test_eval(u):
    result = u.eval()
    print(result)

if __name__ == '__main__':
    u = Unet()
    # test_draw_model(u)
    # test_train(u, batch_size=6, samples_per_epoch=1000, epochs=4)
    test_predict(u)
    # test_continue_train(u, batch_size=6, samples_per_epoch=1000, epochs=8)
    # test_eval(u)
