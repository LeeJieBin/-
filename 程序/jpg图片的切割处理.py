import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import keras

num = 3468
def pic_cut(pic_path,x,st):
    global num
    cut_width = 224
    cut_length = 224
    pic_target = "D:/BaiduNetdiskDownload/rock_all_data/test_/"
    # 读取要分割的图片，以及其尺寸等数据
    picture = cv2.imread(pic_path)
    (width, length, depth) = picture.shape
    # 预处理生成0矩阵
    pic = np.zeros((cut_width, cut_length, depth))
    # 计算可以划分的横纵的个数
    num_width = int(width / cut_width)
    num_length = int(length / cut_length)
    # for循环迭代生成

    for i in range(3, 7):
        for j in range(3, 6):
            num = num + 1
            pic = picture[i*cut_width : (i+1)*cut_width, j*cut_length : (j+1)*cut_length, :]      
            #result_path = pic_target + '{}_{}.jpg'.format(num,x)
            result_path = pic_target + st + '{}_{}.jpg'.format(num,x)
            cv2.imwrite(result_path, pic)

io="D:/BaiduNetdiskDownload/rock_all_data/rock_label.csv"
data = pd.read_csv(io,encoding='GBK')
alist=data.values[0::,1]
label=[]
labellist={'浅灰色细砂岩':0, '深灰色粉砂质泥岩':1, '黑色煤':2, '灰色泥质粉砂岩':3, '灰色细砂岩':4, '深灰色泥岩':5, '灰黑色泥岩':6}

for i in alist:
    label.append(labellist[i])


print(label)

rock_data='D:/BaiduNetdiskDownload/rock_all_data/Rock_es'
files = os.listdir(rock_data)
files.sort(key=lambda x: int(x.split('-')[0]))
ok=0
t=289
for img in files:
    if ok%2==0:
        full_path=rock_data+"/"+img
        #pic_cut(full_path,label[t])
        pic_cut(full_path,label[t],img[:-4])
        #print(label[t])
        print(full_path)
        t+=1
    ok+=1
print(t)