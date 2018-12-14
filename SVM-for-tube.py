# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:54:49 2018

@author: Administrator
"""
import numpy as np
#from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
#i=np.array( range(0,900,30))
    
#df=np.delete(df,[i],axis=0)
arr=[]
for i in range(30):
    arr0 = np.array([0])
    arr1 = np.array([i+1]*29)
    arr2 = np.hstack((arr0,arr1))
    arr.append(arr2)
arr=np.array(arr)
arr=np.reshape(arr,(-1,1))
data0=np.concatenate((df,arr),axis=1)
X = data0[:, :-1]
y = data0[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Build SVR
params = {'kernel': 'rbf', 'C': 10.0, 'epsilon': 0.2} 
regressor = SVR(**params)
#regressor.fit(X, y)
regressor.fit(X_train, y_train)

# Cross validation
import sklearn.metrics as sm
y_pred = regressor.predict(X_test)
plt.plot(range(len(y_pred)),y_pred, 'ro--', label="pred")
plt.plot(range(len(y_test)),y_test, 'ko--', label="true")
plt.legend(loc=0)
print ("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2))

#import numpy as np
#y1=np.random.randint(10,50,(10,3))
##y1=X.T[:,870:900]
#plt.plot(y1)