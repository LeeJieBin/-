import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from pylab import *
from matplotlib.font_manager import FontProperties 

mpl.rcParams['font.sans-serif'] = ['SimHei']
number=0
label=[]
count=[0,0,0,0,0,0,0]
labellist={'浅灰色细砂岩':0, '深灰色粉砂质泥岩':1, '黑色煤':2, '灰色泥质粉砂岩':3, '灰色细砂岩':4, '深灰色泥岩':5, '灰黑色泥岩':6}
name=['浅灰色\n细砂岩', '深灰色粉\n砂质泥岩', '黑色煤', '灰色泥\n质粉砂岩', '灰色\n细砂岩', '深灰\n色泥岩', '灰黑色\n泥岩']
'''
rock_data='D:/BaiduNetdiskDownload/rock_all_data/Data_all'
files = os.listdir(rock_data)
files.sort(key=lambda x: int(x.split('_')[0]))
for img in files:
    full_path=rock_data+"/"+img
    i=eval(full_path[-5])
    label.append(i)
    count[i]+=1
    number+=1
color=['blue','green','yello','red','orange','purple','pink']
'''
io="D:/BaiduNetdiskDownload/rock_all_data/rock_label.csv"
data = pd.read_csv(io,encoding='GBK')
alist=data.values[0::,1]
labellist={'浅灰色细砂岩':0, '深灰色粉砂质泥岩':1, '黑色煤':2, '灰色泥质粉砂岩':3, '灰色细砂岩':4, '深灰色泥岩':5, '灰黑色泥岩':6}
for i in  alist:
    count[labellist[i]]+=1
    number+=1
rects=plt.bar(range(len(count)), count,color='pink',tick_label=name)


for rect in rects:  #rects 是三根柱子的集合
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), size=15, ha='center', va='bottom')
plt.title('数据集rock中各种岩石的数量（白光）')
plt.show() 
print(count)
print(number)