import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
import keras.optimizers
import time
from keras.optimizer_v2 import adam, rmsprop  # 不同版本的keras
import tensorflow
import platform

print("python版本：" + platform.python_version())
print("keras版本" + keras.__version__)
print(tensorflow.__version__)

np.random.seed(520)
io = "D:\\_dataset\\rock_label_1.csv"
data = pd.read_csv(io, encoding='GBK')
alist = data.values[0::, 1]
labellist = {'浅灰色细砂岩': 0, '深灰色粉砂质泥岩': 1, '黑色煤': 2, '灰色泥质粉砂岩': 3, '灰色细砂岩': 4, '深灰色泥岩': 5, '灰黑色泥岩': 6}
# 7分类

r_width, r_height = 112, 112

#读取图片文件
rock_sample_data = 'D:\_dataset\Data_all'
files = os.listdir(rock_sample_data)
images = []
images_norm = []
label = []
count = [0, 0, 0, 0, 0, 0, 0]
number = 0
for img in files:
    full_path = rock_sample_data + "/" + img
    img = cv2.imread(full_path)
    images.append(cv2.resize(img, (r_width, r_height), interpolation=cv2.INTER_NEAREST))  # 缩小图片到r_width*r_height
    i = eval(full_path[-5])
    label.append(i)
    count[i] += 1
    number += 1

X=np.array(images)/255   #对数据除以255，cnn模型对数值小的数处理得比较好
X=X.astype(np.float64)   #转换数据格式,图像格式为np.uint8, 转换成float型，计算机可以计算
Y=np.array(label)
from keras.utils.np_utils import to_categorical

#整理，打乱数据集；分离到训练集和测试集
from pylab import *
from matplotlib.font_manager import FontProperties


def cnt_label(label):
    count = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(label)):
        count[label[i]] += 1
    return count


def print_count(count):
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    labellist = {'浅灰色细砂岩': 0, '深灰色粉砂质泥岩': 1, '黑色煤': 2, '灰色泥质粉砂岩': 3, '灰色细砂岩': 4, '深灰色泥岩': 5, '灰黑色泥岩': 6}
    name = ['浅灰色\n细砂岩', '深灰色粉\n砂质泥岩', '黑色煤', '灰色泥\n质粉砂岩', '灰色\n细砂岩', '深灰\n色泥岩', '灰黑色\n泥岩']
    rects = plt.bar(range(len(count)), count, tick_label=name,
                    color=['darkolivegreen', 'darkkhaki', 'darkgoldenrod', 'darkgray', 'darkgreen', 'coral', 'darkred'])
    for rect in rects:  # rects 是三根柱子的集合
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), size=15, ha='center', va='bottom')

    plt.show()
    print(count)
    print(number)


print_count(count)

num_example = len(images)
print("数量:")
print(num_example)
arr = []

dir_dataset_file = "D:\\_dataset\\trained_models\\dataset_file.txt"
if os.path.exists(dir_dataset_file) == True:  # 读取数据集文件
    with open(dir_dataset_file) as lines:
        for line in lines:
            arr.append(eval(line))
    print("已读取文件上的数据集")
else:
    arr = np.arange(num_example)  # 数据集文件不存在，创建数据集并打乱
    np.random.shuffle(arr)  # 调用函数，打乱顺序

dat = X[arr]
data_label = Y[arr]
ratio = 0.8
s = np.int(num_example * ratio)
ss = np.int(num_example * (ratio + 0.1))
#对数据集的划分为8：1：1
x_train = dat[:s]
y_train = data_label[:s]
x_val = dat[s:ss]
y_val = data_label[s:ss]
x_test = dat[ss:]
y_test = data_label[ss:]

print(X.shape)
print(Y.shape)

y_test=to_categorical(y_test,7)   #多分类问题得在打乱数据前对Y进行one-hot化
y_train=to_categorical(y_train,7)
y_val=to_categorical(y_val,7)

print(x_train.shape)
print("矩阵总大小:")
print(len(x_train)*len(x_train[0])*len(x_train[0][0])*len(x_train[0][0][0]))

#下面是集成学习代码
#1.读取模型
from keras.models import load_model
dir_model=[]
dir_model_base='D:\\_dataset\\trained_models\\'
dir_model.append(dir_model_base+"原始模型(softmax,512epochs).h5")
dir_model.append(dir_model_base+"dense121(256epochs).h5")
dir_model.append(dir_model_base+"densenet169(softmax,256epochs).h5")
dir_model.append(dir_model_base+"VGG16(softmax,256epochs).h5")
#dir_model.append(dir_model_base+"VGG19(softmax,256epochs).h5")

#model1
#model2=define_model()
cnt_models=len(dir_model)
models=[]
for i in range(cnt_models):
    models.append(load_model(dir_model[i]))
print("已读取{}个模型".format(cnt_models))

#2.进行模型预测

pre=[]
for i in range(cnt_models):
    #pre.append(to_categorical(np.argmax(models[i].predict(x_test), axis=-1),7))  #投票法
    pre.append(models[i].predict(x_test))  #概率上的投票法..

#测试集
cnt_test=cnt_label(np.argmax(y_test, axis=-1))
print(len(pre[0])==len(y_test))
y_test_1=np.argmax(y_test, axis=-1)
print(y_test_1) 
#print(pre)

#3.集成学习
#投票法，计算每个模型的预测值之和
if cnt_models>2:
    all_pre=np.sum(pre,axis=0)          #对应位置的概率叠加
    all_pre_spread=all_pre/cnt_models    #取均值后，集成学习后预测概率的分布
    all_pre=np.argmax(all_pre_spread, axis=-1)    #取概率最大对应的标签
else:
    all_pre_spread=np.sum(pre,axis=0)/cnt_models
    all_pre=np.argmax(all_pre_spread, axis=-1)  #如果只有两个模型，那么用预测的概率相加

print(all_pre)

  
print(len(y_test))
print("原始模型(softmax,512epochs):一个样本的预测值之和："+str(sum(pre[0][5])))
print("dense121(256epochs):一个样本的预测值之和："+str(sum(pre[1][5])))
print("densenet169(softmax,256epochs):一个样本的预测值之和："+str(sum(pre[2][5])))
print("VGG16(softmax,256epochs):一个样本的预测值之和："+str(sum(pre[3][5])))


def draw_test_acc(pre,y_test,cnt_test):
    pre_cnt=[0,0,0,0,0,0,0]  #每个类别预测准确率
    for i in range(len(pre)):
        if pre[i]==y_test[i]:
            pre_cnt[pre[i]]+=1
    print(pre_cnt)
    acc=str(sum(pre_cnt)/sum(cnt_test))
    for i in range(len(pre_cnt)):        
        pre_cnt[i]=pre_cnt[i]/cnt_test[i]
    print(pre_cnt)
    print_count(pre_cnt)
    return acc

acc_count=draw_test_acc(all_pre,y_test_1,cnt_test)

from keras import losses
print("集成学习后的测试集准确率："+str(acc_count))
keras_loss=sum(losses.categorical_crossentropy(y_test,all_pre_spread))/len(y_test_1)      #all_pre_spread
print("集成学习后的损失（交叉熵）："+str(keras_loss))


print('\nTesting -------------')

eval_list=[]
for i in range(len(models)):
    eval_list.append(models[i].evaluate(x_test,y_test))

#各个模型的测试
pre_1=np.argmax(models[3].predict(x_test), axis=-1)
#model.predict_classes(x_test)   
acc=draw_test_acc(pre_1,y_test_1,cnt_test)
print("该模型的测试集准确率："+str(acc))

#分别几个模型和集成后的准确率以及损失
acc_list=[eval("%.4f"%(eval(acc_count)))]
loss_list=[eval("%.4f"%(keras_loss))]
for i in range(len(models)):
    loss_list.append(eval("%.4f"%(sum(losses.categorical_crossentropy(y_test,pre[i]))/len(y_test_1))))      #交叉熵损失函数
    acc_list.append(eval("%.4f"%(eval_list[i][1])))
print(acc_list) 
print(loss_list)

#绘画图表，评估集成模型
acc_label=["集成后\n的模型",'原始\n模型','densenet121','dense169','VGG16']
loss_label=acc_label
mpl.rcParams['font.sans-serif'] = ['SimHei']
rects=plt.bar(range(len(acc_label)), np.array(acc_list),tick_label=acc_label,
              color=['darkolivegreen','darkkhaki','darkgoldenrod','darkgray','darkgreen','coral'])
for rect in rects:  
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), size=15, ha='center', va='bottom')
plt.title('各模型准确度')
plt.show() 

rects=plt.bar(range(len(loss_label)),loss_list,tick_label=loss_label,color=['darkolivegreen','darkkhaki','darkgoldenrod','darkgray','darkgreen','coral'])
for rect in rects:  
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), size=15, ha='center', va='bottom')
plt.title('各模型损失')
plt.show() 

