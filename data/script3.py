import cv2
import numpy as np
from tqdm import tqdm


with open("label_exist_tumor.txt",'w') as f:
	for i in tqdm(range(1,2001)):
		pic=np.array(cv2.imread('./Label/Label'+str(i)+'.png'))
		if np.max(pic) == 255 :
			f.write('1\n')
		else:
			f.write('0\n')

