# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:04:06 2018

@author: Evan_He
"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl 
import numpy as np

from sklearn import linear_model  #线性回归库
from sklearn.model_selection import train_test_split #数据分离库
import math


mpl.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['figure.figsize'] = (8, 4.0) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = '' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率

data = pd.read_csv('C:/Users/Evan_He/Desktop/scrap_ershoufang/data/clean_data.csv',encoding="utf8",index_col='链家id')

data_area_price_corr = data[['面积','总价(万元)']].groupby(data['区域']).corr()
group_dict = {}
for zone in set(data['区域']):
    group_dict[zone] = data[data['区域']==zone][['面积','总价(万元)']]
#for zone, value in group_dict.items():
#    area = value['面积'].values.reshape(-1,1)
#    price = value['总价(万元)']
#    X_train,X_test, y_train, y_test = train_test_split(area,price,test_size = 0.25,random_state=100)
#    #random_state 相当于随机数种子,相当于random.seed()
#
#    regr = linear_model.LinearRegression()
#    regr.fit(X_train,y_train)
#    a, b = regr.coef_, regr.intercept_
#    
#    plt.scatter(X_train, y_train, color='blue')
#    x = np.linspace(X_train.min()*0.95,X_train.max()*1.05,1000).reshape(-1,1)
#    plt.plot(x,regr.predict(x),color='red', linewidth=4)
#    plt.xlim(X_train.min()*0.9,X_train.max()*1.05)
#    plt.xlabel('面积',FontSize=12)
#    plt.ylabel('总价(万元)',FontSize=12)
#    plt.title(zone+'train_房价-面积线性回归',FontSize=14)
#    plt.savefig('data/pic/LinearRegression/'+zone+'train_房价-面积线性回归.jpg')
#    plt.show()
#    plt.close()
#    
#    X_test0 = X_test[X_test<=200].reshape(-1,1)
#    
#    y_test0 = y_test[(X_test<=200).reshape(-1,)]
#    bb= regr.predict(X_test0)-y_test0
#    MSE = math.sqrt(sum(bb**2)/len(bb))
#    
#    plt.scatter(X_test0, y_test0, color='blue')
#    x = np.linspace(X_test0.min()*0.95,X_test0.max()*1.05,1000).reshape(-1,1)
#    plt.plot(x,regr.predict(x),color='red', linewidth=4)
#    plt.xlim(X_test0.min()*0.9,X_test0.max()*1.05)
#    plt.xlabel('面积',FontSize=12)
#    plt.ylabel('总价(万元)',FontSize=12)
#    
#
#    
#    plt.title(zone+'test_mse:'+str("%.2f") %MSE,FontSize=14)
#    plt.savefig('data/pic/LinearRegression/'+zone+'test_房价-面积线性回归.jpg')
#    plt.show()
#    plt.close()
    
    
"""
比较不同的支持向量回归，不同的核函数的对误差的影响
"""
from sklearn.preprocessing import StandardScaler  #数据归一化库
from sklearn.svm import SVR  #支持向量回归库
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error #回归的不同误差表示
    
    
for zone, value in group_dict.items():
    area = value['面积'].values.reshape(-1,1)
    price = value['总价(万元)'].values.reshape(-1,1)
    X_train,X_test, y_train, y_test = train_test_split(area,price,test_size = 0.25,random_state=100)
    
    X_test0 = X_test[X_test<=200].reshape(-1,1)
    y_test0 = y_test[(X_test<=200)].reshape(-1,1)
    
    ss_X = StandardScaler()
    ss_y = StandardScaler()
    #数据的标准化处理 
    X_train = ss_X.fit_transform(X_train)
    X_test0 = ss_X.transform(X_test0)
    y_train = ss_y.fit_transform(y_train)
    y_test0 = ss_y.transform(y_test0)

    linear_svr = SVR(kernel = 'linear')
    linear_svr.fit(X_train,y_train)

    
    y_predict = linear_svr.predict(X_test0)
    RMSE = math.sqrt(mean_squared_error(ss_y.inverse_transform(y_test0), ss_y.inverse_transform(y_predict)))
    
    plt.scatter(ss_X.inverse_transform(X_test0), ss_y.inverse_transform(y_test0), color='blue')
    

    xx = np.sort(X_test0,axis=0)
    plt.plot(ss_X.inverse_transform(xx),ss_y.inverse_transform(linear_svr.predict(xx)),color='red', linewidth=4)
    plt.xlabel('面积',FontSize=12)
    plt.ylabel('总价(万元)',FontSize=12)
    

    plt.title(zone+'test_mse:'+str("%.2f") %RMSE,FontSize=14)
    plt.savefig('data/pic/SVR/'+zone+'linear_test_回归.jpg')
    plt.show()
    plt.close()
       

#    poly_svr = SVR(kernel = 'poly',degree=2)
#    poly_svr.fit(X_train,y_train)
#
#    
#    y_predict = poly_svr.predict(X_test0)
#    RMSE = math.sqrt(mean_squared_error(ss_y.inverse_transform(y_test0), ss_y.inverse_transform(y_predict)))
#    
#    plt.scatter(ss_X.inverse_transform(X_test0), ss_y.inverse_transform(y_test0), color='blue')
#
#    plt.plot(ss_X.inverse_transform(xx),ss_y.inverse_transform(poly_svr.predict(xx)),color='red', linewidth=4)
#    plt.xlabel('面积',FontSize=12)
#    plt.ylabel('总价(万元)',FontSize=12)
#    
#
#    plt.title(zone+'test_mse:'+str("%.2f") %RMSE,FontSize=14)
#    plt.savefig('data/pic/SVR/'+zone+'poly_test_回归回归.jpg')
#    plt.show()
#    plt.close()
#    
    
    
    rbf_svr = SVR(kernel = 'rbf', gamma=1.5)
    rbf_svr.fit(X_train,y_train)

    
    y_predict = rbf_svr.predict(X_test0)
    RMSE = math.sqrt(mean_squared_error(ss_y.inverse_transform(y_test0), ss_y.inverse_transform(y_predict)))
    
    plt.scatter(ss_X.inverse_transform(X_test0), ss_y.inverse_transform(y_test0), color='blue')

    plt.plot(ss_X.inverse_transform(xx),ss_y.inverse_transform(rbf_svr.predict(xx)),color='red', linewidth=4)
    plt.xlabel('面积',FontSize=12)
    plt.ylabel('总价(万元)',FontSize=12)
    

    plt.title(zone+'test_mse:'+str("%.2f") %RMSE,FontSize=14)
    plt.savefig('data/pic/SVR/'+zone+'rbf_test_回归回归.jpg')
    plt.show()
    plt.close()
#    
#    

    
    
    
    
    
    