import os

train_file="D:\\_dataset\\trained_models\\模型历史数据\\原始模型(softmax,512epochs)\\"
train_loss=train_file+"loss.txt"
train_acc=train_file+"acc.txt"
train_val_loss=train_file+"val_loss.txt"
train_val_acc=train_file+"val_acc.txt"

train_file_list=[train_loss,train_acc,train_val_loss,train_val_acc]
train_key_list=['loss', 'accuracy', 'val_loss', 'val_accuracy']

history={'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}
#读取模型
for i in range(4):
    with open(train_file_list[i]) as lines:
        for line in lines:
            history[train_key_list[i]].append(eval(line))
print("已读取模型历史数据")
    
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.font_manager import FontProperties 

#########画图
mpl.rcParams['font.sans-serif'] = ['SimHei']
accuracy = history['accuracy']     #获取训练集准确性数据
val_accuracy = history['val_accuracy']    #获取验证集准确性数据
loss = history['loss']          #获取训练集错误值数据
val_loss = history['val_loss']  #获取验证集错误值数据
epochs = range(1,len(accuracy)+1)
plt.plot(epochs,accuracy,label='Trainning acc',color = 'blue')     #以epochs为横坐标，以训练集准确性为纵坐标
plt.plot(epochs,val_accuracy,label='Vaildation cc',color = 'red') #以epochs为横坐标，以验证集准确性为纵坐标
plt.legend()   #绘制图例，即标明图中的线段代表何种含义
plt.title('模型准确率')

plt.figure()   #创建一个新的图表
plt.plot(epochs,loss,label='Trainning loss',color = 'blue')
plt.plot(epochs,val_loss,label='Vaildation loss',color = 'red')
plt.legend()  ##绘制图例，即标明图中的线段代表何种含义
plt.title('模型损失')
plt.show()    #显示图表


