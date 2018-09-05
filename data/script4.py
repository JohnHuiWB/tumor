import cv2
import numpy as np
from tqdm import tqdm

for i in tqdm(range(1,2201)):
	img=cv2.imread('./Label/Label'+str(i)+'.png',0)
	ret,thresh = cv2.threshold(img,1,255,cv2.THRESH_BINARY)

	cv2.imwrite('./Label2/Label'+str(i)+'.png',thresh)
		