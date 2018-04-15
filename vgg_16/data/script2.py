
import glob
import cv2



image_path = ( 'Image/*')
images = glob.glob(image_path)

j=2201         
for x in images:
    i=int(x.split('.')[-2].split('\IM')[-1])
    if(i>2000):
        img=cv2.imread(x)
        k=0.9
        for i in range(9):  #每张负样本生成9张
        	img2=img*k  #调整亮度
        	k-=0.07
        	cv2.imwrite("Image/IM"+str(j)+'.png',img2)
            print("IM"+str(j)+'.png')
        	j+=1
        	



        
