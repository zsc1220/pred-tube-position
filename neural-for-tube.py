# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:54:49 2018

@author: Administrator
"""
import numpy as np
import math
#from sklearn import preprocessing
#from sklearn.svm import SVR
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt

Path0 ='resultBlowPipe.txt'
F=open(Path0, 'r')
Lines0 = F.readlines()
F.close()
result = []
for a in Lines0:
    a_array = a.split( '\n' )
    result.append(a_array[0])
result = [x for x in result if x != '']#去掉列表中元素为空格的元素
c=[]
d=[]
for b in result:
    b_array = b.split('   ')
    c.append(b_array[0])
    d.append(b_array[1])
e1,e2,e3=[],[],[]
for item in enumerate(c):    
    if item[1] == '节点11':
        e1.append(float(d[item[0]]))
    if item[1] == '节点14':
        e2.append(float(d[item[0]]))
    if item[1] == '节点22':
        e3.append(float(d[item[0]]))
e11=np.array(e1)[:,np.newaxis]
e21=np.array(e2)[:,np.newaxis]
e31=np.array(e3)[:,np.newaxis]
e=np.concatenate((e11,e21,e31),axis=1)
df=[]
for i in range(len(e)):
    singe=(e[0]-e[i])/e[0]
    df.append(singe)
df=np.array(df)
i=np.array( range(0,900,30))
df=np.delete(df,[i],axis=0)

points=[]
for i in range(5):
    for j in range(6):
        point=[200*i+200,100*j+100]
        points.append(point)
        
dics1,dics2,dics3=[],[],[]
for i in range(len(points)):
    dist1=np.array(points[i])-np.array(points[10])
    dic1=math.hypot(abs(dist1[0]),abs(dist1[1]))
    dics1.append(dic1)
    dist2=np.array(points[i])-np.array(points[13])
    dic2=math.hypot(abs(dist2[0]),abs(dist2[1]))
    dics2.append(dic2)
    dist3=np.array(points[i])-np.array(points[21])
    dic3=math.hypot(abs(dist3[0]),abs(dist3[1]))
    dics3.append(dic3)

d1=np.reshape(np.array(dics1*29),(29,-1)).T
d1=np.reshape(d1,(-1,1))

d2=np.reshape(np.array(dics2*29),(29,-1)).T
d2=np.reshape(d2,(-1,1))

d3=np.reshape(np.array(dics3*29),(29,-1)).T
d3=np.reshape(d3,(-1,1))

arr=[]
for i in range(30):
#    arr0 = np.array([0])
    arr1 = np.array([i+1]*29)
#    arr2 = np.hstack((arr0,arr1))
    arr.append(arr1)
arr=np.array(arr)
arr=np.reshape(arr,(-1,1))

data0=np.concatenate((df,d1,d2,d3,arr),axis=1)
np.random.shuffle(data0)
X_train = data0[:-10, :-4]
y1_train = data0[:-10, -4]
y2_train = data0[:-10, -3]
y3_train = data0[:-10, -2]

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成200个随机点
x_data = X_train
y_data = data0[:-10, -2][:,np.newaxis]
#y_data = np.sqrt(x_data) + noise
#定义两个placeholder
x = tf.placeholder(tf.float32,[None,3])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层1
Weights_L1 = tf.Variable(tf.random_normal([3,20]))
biases_L1 = tf.Variable(tf.zeros([1,20]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)
#L1 = Wx_plus_b_L1

##定义神经网络中间层2
Weights_L2 = tf.Variable(tf.random_normal([20,20]))
biases_L2 = tf.Variable(tf.zeros([1,20]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
L2 = tf.nn.tanh(Wx_plus_b_L2)
#L2 = Wx_plus_b_L2

#定义神经网络输出层
Weights_L3 = tf.Variable(tf.random_normal([20,1]))
biases_L3 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L3 = tf.matmul(L2,Weights_L3) + biases_L3
#prediction = tf.nn.tanh(Wx_plus_b_L3)
prediction = Wx_plus_b_L3

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法训练
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
lost=[]
#Weights_L3_list=[]
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(20000):
       _,loss_= sess.run([train_step,loss],feed_dict={x:x_data,y:y_data})
       print('loss:',loss_)
#       print('Weights_L3:',Weights_L3_)
       lost.append(loss_)
       
        
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:data0[-10:, :-4]})
    #画图
    plt.figure()
    plt.plot(data0[-10:, -2],'ro--', label="true")
    plt.plot(prediction_value,'ko--', label="pred")
    plt.legend(loc=0)
    plt.show()





'''
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

params = {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.2} 
regressor1 = SVR(**params)
regressor1.fit(X_train, y1_train)

regressor2 = SVR(**params)
regressor2.fit(X_train, y2_train)

regressor3 = SVR(**params)
regressor3.fit(X_train, y3_train)

# Cross validation
#import sklearn.metrics as sm
y1_pred = regressor1.predict(data0[-10:-8, :-4])
y2_pred = regressor2.predict(data0[-10:-8, :-4])
y3_pred = regressor3.predict(data0[-10:-8, :-4])
y_label = data0[-10:-8, -1]
print(y1_pred,y2_pred,y3_pred,y_label)

r1=int(y1_pred[0])
r2=int(y2_pred[0])
r3=int(y3_pred[0])

import pygame
pygame.init()
screen=pygame.display.set_caption('hello world')
screen=pygame.display.set_mode([1200,800])
screen.fill([255,255,255])
pygame.draw.circle(screen,[0,0,0],[100,50],7,0) #颜色、坐标、半径、是否填充
points=[]
for i in range(5):
    for j in range(6):
        point=[(i+1)*200,(j+1)*100]
        points.append(point)
        pygame.draw.circle(screen,[0,0,0],point,7,0)
pygame.draw.line(screen,[0,0,0],[100,50],[200,100],3)
for i in range(5):
    pygame.draw.line(screen,[0,0,0],[200*i+200,100],[200*i+200,600],3)
for i in range(6):
    pygame.draw.line(screen,[0,0,0],[200,100*i+100],[1000,100*i+100],3)

pygame.draw.circle(screen,[255,0,0],[400,500],r1,1)
pygame.draw.circle(screen,[255,0,0],[600,200],r2,1) 
pygame.draw.circle(screen,[255,0,0],[800,400],r3,1) 
pygame.display.flip()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
#plt.plot(range(len(y_pred)),y_pred, 'ro--', label="pred")
#plt.plot(range(len(y_test[:10])),y_test[:10], 'ko--', label="true")
#plt.legend(loc=0)
#print ("Mean absolute error =", round(sm.mean_absolute_error(y_test[:10], y_pred), 2))

#import numpy as np
#y1=np.random.randint(10,50,(10,3))
##y1=X.T[:,870:900]
#plt.plot(y1)
'''