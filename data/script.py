#!/usr/bin/python
# -*- coding: utf-8 -*-

# @File  : script.py
# @Author: JohnHuiWB
# @Date  : 2018/3/30 0030
# @Desc  :
# @Contact : huiwenbin199822@gmail.com
# @Software : PyCharm

import cv2
import numpy as np

NUM = 920  # 提取的图片编号

img = cv2.imread('./Label/Label'+str(NUM)+'.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 彩色转灰度

# 二值化处理，低于阈值的像素点灰度值置为0；高于阈值的值置为参数3
ret, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)  # 初始化一个3*3的kernel
dilation = cv2.dilate(img_gray, kernel, iterations=1)  # 膨胀
black_hat = img_gray - dilation  # 黑帽

# 检索模式为树形cv2.RETR_CCOMP，
# RETR_CCOMP：提取所有轮廓，并且将其组织为双边结构。
# 意思就是第一层为它的外围轮廓，第二层为内围轮廓，利用这个参数可以筛选出来一个独立边缘只对应一个轮廓。
# 轮廓存储模式为简单模式cv2.CHAIN_APPROX_SIMPLE
_, contours, hierarchy = cv2.findContours(
    black_hat, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# https://blog.csdn.net/liumangmao1314/article/details/68922647
# https://blog.csdn.net/jjddss/article/details/73527990
# RETR_CCOMP模式下，hierarchy[i][0]–hierarchy[i][3]依次表示轮廓的后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓，
# 如果没有对应项，设置为-1。
# 利用这个特性，可以找到所有的最内侧的轮廓。
_, x, _ = np.shape(hierarchy)
w = []
h = []
for i in range(x):
    # 确定此轮廓是最内侧的轮廓（没有内嵌轮廓），且不独立存在（既没有父轮廓，又没有内嵌轮廓）
    if hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1:
        w.append(contours[i])
        h.append(hierarchy[0][i])
for x in h:
    print(x)
print(np.size(w))


im = cv2.imread('./Image/IM'+str(NUM)+'.png')
# 绘制轮廓，第三个参数是轮廓的索引（在绘制独立轮廓是很有用，当设置为 -1 时绘制所有轮廓），
# 接下来的参数是轮廓的颜色和厚度等，此时是将轮廓绘制到了原始图像上
im_with_contour = cv2.drawContours(im, w, -1, (0, 0, 255), 1)

cv2.imshow("test", im_with_contour)
cv2.waitKey(0)
cv2.destroyAllWindows()

