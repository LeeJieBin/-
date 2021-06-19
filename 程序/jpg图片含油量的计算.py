import cv2
import numpy as np
import os
import pandas as pd
def duibi(img):
    img2=cv2.imread('D:/BaiduNetdiskDownload/rock_all_data/test/125.jpg')
    img = cv2.addWeighted(img,2,img2,2,0)   # 调整对比度
    #cv2.imwrite("D:/BaiduNetdiskDownload/rock_all_data/8_5.jpg",img)
    return img
#去掉蓝色背景
def hanyou_1(img_path1,img_path_2):
    #src_1 = cv2.imread("D:/BaiduNetdiskDownload/rock_all_data/Rock_es/326-1.jpg")
    src_1 = cv2.imread(img_path1)    #白光
    #cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)    
    hsv_1 = cv2.cvtColor(src_1, cv2.COLOR_BGR2HSV)
    
    low_hsv2 = np.array([100,43,46])
    high_hsv2 = np.array([124,255,255])
    
    back = cv2.inRange(hsv_1,lowerb=low_hsv2,upperb=high_hsv2)
    back=np.array(back)
    #cv2.imwrite('D:/BaiduNetdiskDownload/rock_all_data/88.jpg',back)
    
    src = cv2.imread(img_path_2)     #荧光
    src = duibi(src)    #调整对比度
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        
    low_hsv1 = np.array([26,43,46])
    high_hsv1 = np.array([34,255,255])
    
    
    low_hsv = np.array([35,43,46])
    high_hsv = np.array([77,255,255])
    
    mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    mask_1=cv2.inRange(hsv,lowerb=low_hsv1,upperb=high_hsv1)
        
    #cv2.imwrite('D:/BaiduNetdiskDownload/you/temp/122.bmp',mask)
    #cv2.imwrite('D:/BaiduNetdiskDownload/you/temp/126.bmp',mask_1)
       
    num_255=0     #油
    S=0       #岩石面积
    
    mask=np.array(mask)
    mask_1=np.array(mask_1)
    
    for i in range(2048):
        for j in range(2448):
            if back[i][j]==255:
                continue
            else:
                S+=1
                if mask[i][j]==255 or mask_1[i][j]==255:
                    num_255+=1
    print("含油量：")
    uio=(num_255)/(S)
    uio=str(round(uio, 4)*100)+'%'
    print(uio)
    return uio

blist=[]
rock_data='D:/BaiduNetdiskDownload/rock_all_data/Rock_es'
files = os.listdir(rock_data)
files.sort(key=lambda x: int(x.split('-')[0]))
ok=0
full_path_1 = 'abc'
full_path_2 = 'abc'
for img in files:
    if ok%2==0:
        full_path_1=rock_data+"/"+img       
        print(full_path_1)
    else:
        full_path_2=rock_data+"/"+img       
        print(full_path_2)
        blist.append(hanyou_1(full_path_1,full_path_2))
    ok+=1


d = pd.DataFrame(blist)
d.to_csv('D:/BaiduNetdiskDownload/rock_all_data/a.csv',index=False,header=None)


