#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : test_data_pkg.py
# @Author: JohnHuiWB
# @Date  : 2018/3/31 0031
# @Desc  : 
# @Contact : huiwenbin199822@gmail.com 
# @Software : PyCharm


import tumor.seg_data as data
import numpy as np
import cv2

def test_generate_arrays_from_file():
    x, y = data.generate_arrays_from_file(data.FILENAME, 1).__next__()
    x *= 255
    x = x.astype('uint8')
    x = np.squeeze(x)
    print(x.shape)
    im = cv2.imshow('x', x)
    im.show()


if __name__ == '__main__':
    test_generate_arrays_from_file()
