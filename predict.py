from keras.models import load_model
import cv2
import numpy as np

model1 =load_model('inceptionv3_tranfer/model.h5')
model2 =load_model('unet/unet.h5')

def predict(filename):
	img=cv2.imread(filename,1)
	cv2.imshow('original', img)
	x=cv2.resize(img[100:400],(299,299))
	x=(np.array(x, dtype=np.float32))/255.
	x=np.reshape(x,[1,299,299,3])
	r=model1.predict(x)
	if(r[0][0]>=0.5):
		image=cv2.imread(filename,0)
		x=(np.array(image, dtype=np.float32))/255.
		x=np.reshape(x,[1,512,512,1])
		r=model2.predict(x)
		r *= 255
		r=r.astype('uint8')
		r=np.squeeze(r)
		cv2.imshow('ending', r)
	else:
		print("这张图片无肿瘤")
	cv2.waitKey(0)
	cv2.destroyAllWindows()


predict("IM8.png")