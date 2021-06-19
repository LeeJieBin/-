import cv2
import numpy as np
import os
import pandas as pd
alist=[]

def Liangduguiyi(image):
    m_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Gamma = np.log(128.0 / 255.0) / np.log(cv2.mean(m_gray)[0] / 255.0)
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, Gamma) * 255.0, 0, 255)
    image = cv2.LUT(image, lookUpTable)
    return image

def hanyou(img_path):
    src=cv2.imread(img_path)
    #cv2.namedWindow("input", cv2.WINDOW_AUTOSIZE)
    src=Liangduguiyi(src)
    """
    提取图中的红色部分
    """
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    
    low_hsv1 = np.array([26,43,46])
    high_hsv1 = np.array([34,255,255])
    
    
    low_hsv = np.array([35,43,46])
    high_hsv = np.array([77,255,255])
    
    mask = cv2.inRange(hsv,lowerb=low_hsv,upperb=high_hsv)
    mask_1=cv2.inRange(hsv,lowerb=low_hsv1,upperb=high_hsv1)
    
    
    #cv2.imwrite('D:/BaiduNetdiskDownload/you/temp/122.bmp',mask)
    #cv2.imwrite('D:/BaiduNetdiskDownload/you/temp/126.bmp',mask_1)
    
    num_0=0
    num_255=0
    
    sum_0=0
    sum_255=0
    
    mask=np.array(mask)
    mask_1=np.array(mask_1)
    #print(mask)
    
    
    for han in mask:
        for i in han:
            if i==0:
                num_0+=1
            elif i==255:
                num_255+=1
                
    for gan in mask_1:
        for j in gan:
            if j==0:
                sum_0+=1
            elif j==255:
                sum_255+=1
    '''
    print("黄色：")
    print(num_0)
    print(num_255)
    print("绿色：")
    print(sum_0)
    print(sum_255)
    '''
    print("含油量：")
    uio=(sum_255+num_255)/(3000*4096)
    uio=str(round(uio, 4)*100)+'%'
    print(uio)
    return uio;


path='D:/BaiduNetdiskDownload/you/temp/4-2.bmp'
rock_data='D:/BaiduNetdiskDownload/rock_all_data/Rock'
files = os.listdir(rock_data)
files.sort(key=lambda x: int(x.split('-')[0]))
ok=0
for img in files:
    if ok%2==1:
        full_path=rock_data+"/"+img
        alist.append(hanyou(full_path))
        print(full_path)
    ok+=1



d = pd.DataFrame(alist)
d.to_csv('D:/BaiduNetdiskDownload/rock_all_data/a.csv',index=False,header=None)