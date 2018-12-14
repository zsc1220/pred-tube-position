# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:10:27 2018

@author: Administrator
"""
#import datetime
#start = datetime.datetime.now()
##import math
##a=math.fabs(1000**2-456**2)
#a=abs(1000**2-456**2)
#
#end = datetime.datetime.now()
#print (end-start)

import numpy as np
import math
p1=np.array([0,0])
p2=np.array([1000,2000])
p3=p2-p1
p4=math.hypot(p3[0],p3[1])
print(p4)


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

d1=np.array(dics1*29)
d1=np.reshape(d1,(29,-1)).T
d1=np.reshape(d1,(-1,1))

'''
#计算两圆的交点坐标
import math
import numpy as np
def insec(p1,r1,p2,r2):
    x = p1[0]
    y = p1[1]
    R = r1
    a = p2[0]
    b = p2[1]
    S = r2
    d = math.sqrt((abs(a-x))**2 + (abs(b-y))**2)
    if d > (R+S) or d < (abs(R-S)):
        print ("Two circles have no intersection")
        return 
    elif d == 0 and R==S :
        print ("Two circles have same center!")
        return
    else:
        A = (R**2 - S**2 + d**2) / (2 * d)
        h = math.sqrt(R**2 - A**2)
        x2 = x + A * (a-x)/d
        y2 = y + A * (b-y)/d
        x3 = round(x2 - h * (b - y) / d,2)
        y3 = round(y2 + h * (a - x) / d,2)
        x4 = round(x2 + h * (b - y) / d,2)
        y4 = round(y2 - h * (a - x) / d,2)
        print (x3, y3)
        print (x4, y4)
        c1=np.array([x3, y3])
        c2=np.array([x4, y4])
        return c1,c2

P1=np.array([-5,0])
R1=10
P2=np.array([5,0])
R2=5
C=insec(P1,R1,P2,R2)
C1=C[0]
C2=C[1]
'''

