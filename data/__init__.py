#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : __init__.py
# @Author: JohnHuiWB
# @Date  : 2018/3/31 0031
# @Desc  : 
# @Contact : huiwenbin199822@gmail.com 
# @Software : PyCharm

from tumor.data.load_data import FILENAME
import os

if not os.path.exists(FILENAME):
    print('文件不存在：', FILENAME)
    from tumor.data.load_data import create_and_save
    create_and_save()
    print('读取文件：', FILENAME)
else:
    print('读取文件：', FILENAME)

from tumor.data.load_data import load
load = load
