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

#随机访问一个数据并绘图，查看数据处理结果
labellist_=['浅灰色细砂岩', '深灰色粉砂质泥岩', '黑色煤', '灰色泥质粉砂岩', '灰色细砂岩', '深灰色泥岩', '灰黑色泥岩']
#print(len(x_train))
import matplotlib.pyplot as plt
import random
rand=random.randint(0,len(X))

imagedemo = X[rand]
imagedemolabel = label[rand]
plt.imshow(imagedemo)
print("下图显示的是："+labellist_[imagedemolabel])
print(type(imagedemo))
print(imagedemolabel)

#数据增强
'''
keras数据增强有两个办法
1.使用flow产生一个迭代器
2.对文件保存路径使用.flow_from_directory(directory)，可以直接训练
'''
#训练集数据增强处理
datagen = ImageDataGenerator(
    rotation_range = 2,     # 随机旋转度数
    width_shift_range = 0.20, # 随机水平平移
    height_shift_range = 0.20,# 随机竖直平移
    #rescale = 1/255,         # 数据归一化
    shear_range = 10,       # 随机错切变换
    zoom_range = 0.20,        # 随机放大
    horizontal_flip = True,  # 水平翻转
    fill_mode = 'wrap',   # 填充方式
)

#查看处理前后的图片
temp=x_train[0:1]
plt.imshow(temp[0])
print(temp.shape)
for batch in datagen.flow(temp, batch_size=1):
    plt.imshow(batch[0])
    i += 1
    if i > 2:
        break  # otherwise the generator would loop indefinitel

from keras.models import Sequential
from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import models


# 模型的建立
def define_model():
    model = Sequential()

    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     padding='same',
                     input_shape=(r_width, r_height, 3),
                     activation='relu'))  # 卷积层1

    model.add(MaxPooling2D(pool_size=(2, 2)))  # 池化层2
    # model.add(BatchNormalization(axis=-1))  #添加BN层将上一层的数据标准化，利于避开激励函数的饱和区
    CLASS = 7
    DROPOUT_RATE = 0.2

    # 再次构建一个卷积层
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    # 构建一个池化层，提取特征，池化层的池化窗口为2*2，步长为2。
    model.add(MaxPool2D(pool_size=(3, 3), strides=1))
    # 继续构建卷积层和池化层，区别是卷积核数量为64。
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Dropout(DROPOUT_RATE))  # 加入dropout，防止过拟合。
    model.add(MaxPool2D(pool_size=(3, 3), strides=1))

    model.add(Flatten())  # 数据扁平化
    model.add(Dense(128, activation='relu'))  # 构建一个具有128个神经元的全连接层
    model.add(Dense(64, activation='relu'))  # 构建一个具有64个神经元的全连接层
    model.add(Dropout(DROPOUT_RATE))  # 加入dropout，防止过拟合。
    model.add(Dense(CLASS, activation='softmax'))  # 输出层，一共7个神经元，对应7个分类
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


#迁移学习
from keras.applications import vgg19
from keras import layers,optimizers
from keras.applications.densenet import DenseNet169
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 调用GPU
''''''
net = vgg19.VGG19(include_top=False, weights='imagenet', pooling='max')
epochs = 256

# 迁移学习实例化
net.trainable = False
model = keras.Sequential([

    net,
    layers.Dense(512, activation='relu'),
    # layers.BatchNormalization(),
    layers.Dropout(rate=0.25),
    layers.Dense(7, activation='softmax')

])
model.build(input_shape=(r_width, r_height, 3))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.001,
    patience=10
)

# model = define_model()    #训练自己模型时，注释掉迁移学习代码，使用此代码

history = model.fit(datagen.flow(x_train, y_train, batch_size=32),  # 验证集参与
                    validation_data=(x_val, y_val),
                    epochs=epochs)
                    # ,callbacks=[early_stopping])


#画图
mpl.rcParams['font.sans-serif'] = ['SimHei']
accuracy = history.history['accuracy']     #获取训练集准确性数据
val_accuracy = history.history['val_accuracy']    #获取验证集准确性数据
loss = history.history['loss']          #获取训练集错误值数据
val_loss = history.history['val_loss']  #获取验证集错误值数据
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

print('\nTesting -------------')
list1=model.evaluate(x_test,y_test)
print(list1)
pre=np.argmax(model.predict(x_test), axis=-1)
y_test_1=np.argmax(y_test, axis=-1)
#验证集
cnt_test=cnt_label(np.argmax(y_test, axis=-1))

pre_cnt=[0,0,0,0,0,0,0]  #每个类别预测准确率

for i in range(len(pre)):
    if pre[i]==y_test_1[i]:
        pre_cnt[pre[i]]+=1
print(pre_cnt)
for i in range(len(pre_cnt)):
    pre_cnt[i]=pre_cnt[i]/cnt_test[i]
print(pre_cnt)
print_count(pre_cnt)

#保存训练结果
#数据集文件保存
import os
dir_model="D:\\_dataset\\trained_models\\VGG19_2(softmax,256epochs).h5"

if os.path.exists(dir_model)==False:
    model.save(dir_model)
    print("保存模型成功")

train_file="D:\\_dataset\\trained_models\\模型历史数据\\VGG19_2(softmax,256epochs).h5\\"
train_loss=train_file+"loss.txt"
train_acc=train_file+"acc.txt"
train_val_loss=train_file+"val_loss.txt"
train_val_acc=train_file+"val_acc.txt"

if os.path.exists(train_loss)==False:
    np.savetxt(train_loss, history.history['loss'], fmt="%f") 
    print("保存模型训练损失成功")
if os.path.exists(train_acc)==False:
    np.savetxt(train_acc, history.history['accuracy'], fmt="%f") 
    print("保存模型训练准确率成功")
if os.path.exists(train_val_loss)==False:
    np.savetxt(train_val_loss, history.history['val_loss'], fmt="%f") 
    print("保存模型验证损失成功")
if os.path.exists(train_val_acc)==False:
    np.savetxt(train_val_acc, history.history['val_accuracy'], fmt="%f") 
    print("保存模型验证准确率成功")

dir_dataset_file="D:\\_dataset\\trained_models\\dataset_file.txt"
if os.path.exists(dir_dataset_file)==False:
    np.savetxt(dir_dataset_file, arr, fmt="%d") #保存为整数
    print("保存数据集成功")
