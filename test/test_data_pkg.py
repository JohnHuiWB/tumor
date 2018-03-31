#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : test_data_pkg.py
# @Author: JohnHuiWB
# @Date  : 2018/3/31 0031
# @Desc  : 
# @Contact : huiwenbin199822@gmail.com 
# @Software : PyCharm

if __name__ == '__main__':
    import tumor.data as data
    xs, ys = data.load()
    print(len(xs))
    print(ys[0].shape)