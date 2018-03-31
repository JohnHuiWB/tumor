#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : __init__.py
# @Author: JohnHuiWB
# @Date  : 2018/3/31 0031
# @Desc  : 
# @Contact : huiwenbin199822@gmail.com 
# @Software : PyCharm

from tumor.data.load_data import *
import os

if not os.path.exists(FILENAME):
    create_and_save()

load = load
