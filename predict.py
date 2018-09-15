import cv2
import numpy as np
from keras.models import load_model

# 禁止输出tensorflow的log
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# 加载四个模型
print('————————————————————加载模型中————————————————————')
model1 =load_model('inceptionv3_outline/model.h5')
print('进度  1 / 4')
model3 =load_model('inceptionv3_tumor/model5.h5')
print('进度  2 / 4')
model2 =load_model('unet/unet.h5')
print('进度  3 / 4')
model4 =load_model('unet2/unet.h5')
print('进度  4 / 4')
print('————————————————————加载完成————————————————————')


def judge_outline(img):  #3通道
	img=cv2.resize(img[100:400],(299,299))
	img=(np.array(img, dtype=np.float32))/255.
	img=np.reshape(img,[1,299,299,3])
	end=model1.predict(img)
	# print(end)
	if end[0][0]>=0.5:
		return True
	else:
		return False

def gen_outline(img): #1通道
	x=(np.array(img, dtype=np.float32))/255.
	x=np.reshape(x,[1,512,512,1])
	r=model2.predict(x)
	r *= 255
	r=r.astype('uint8')
	r=np.squeeze(r)
	cv2.imshow('outline', r)
	return r

def judge_tumor(img_origion,img_unet):
	img_origion=cv2.resize(img_origion,(299,299))
	img_origion=(np.array(img_origion, dtype=np.float32))/255.
	img_origion=np.reshape(img_origion,[1,299,299,3])

	img_unet=cv2.resize(img_unet,(299,299))
	img_unet=(np.array(img_unet, dtype=np.float32))/255.
	img_unet=np.reshape(img_unet,[1,299,299,1])
	end=model3.predict([img_origion,img_unet])
	# print(end)
	if end[0][0]>=0.5:
		return True
	else:
		return False

def gen_tumor(img_origion,img_unet):
	img_origion=cv2.resize(img_origion,(512,512))
	img_origion=(np.array(img_origion, dtype=np.float32))/255.
	img_origion=np.reshape(img_origion,[1,512,512,1])

	img_unet=cv2.resize(img_unet,(512,512))
	img_unet=(np.array(img_unet, dtype=np.float32))/255.
	img_unet=np.reshape(img_unet,[1,512,512,1])
	r=model4.predict([img_origion,img_unet])
	r *= 255
	r=r.astype('uint8')
	r=np.squeeze(r)
	cv2.imshow('tumor', r)
	return r


def predict(filename):
	img1=cv2.imread(filename,1)
	cv2.imshow('orign', img1)
	img2=cv2.imread(filename,0)
	img3=img1
	img4=img2
	print('\n————————————————————输出结果————————————————————')
	print('是否有膀胱壁：')
	if judge_outline(img1):
		print("\t有膀胱壁")
		outline=gen_outline(img2)
		outline2=outline
		print('是否有肿瘤：')
		if judge_tumor(img3,outline):
			print("\t有肿瘤")
			gen_tumor(img4,outline2)
		else:
			print("\t无肿瘤")
	else:
		print("\t无膀胱壁")
	print('——————————————————————————————————————————————')
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# def predict(filename):
# 	img=cv2.imread(filename,1)
# 	cv2.imshow('original', img)
# 	x=cv2.resize(img[100:400],(299,299))
# 	x=(np.array(x, dtype=np.float32))/255.
# 	x=np.reshape(x,[1,299,299,3])
# 	r=model1.predict(x)
# 	if(r[0][0]>=0.5):
# 		image=cv2.imread(filename,0)
# 		x=(np.array(image, dtype=np.float32))/255.
# 		x=np.reshape(x,[1,512,512,1])
# 		r=model2.predict(x)
# 		r *= 255
# 		r=r.astype('uint8')
# 		r=np.squeeze(r)
# 		cv2.imshow('ending', r)
# 	else:
# 		print("这张图片无肿瘤")
# 	cv2.waitKey(0)
# 	cv2.destroyAllWindows()

predict("IM26.png")
