# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:25:31 2018

@author: Evan_He
"""
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl 
mpl.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['figure.figsize'] = (13, 8.0) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = '' # 设置 interpolation style
plt.rcParams['image.cmap'] = 'gray' # 设置 颜色 style
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 100 #分辨率



data = pd.read_csv('C:/Users/Evan_He/Desktop/scrap_ershoufang/data/clean_data.csv',encoding="utf8",index_col='链家id')

"""
对楼层进行分块，10层以下记为0, 11-25记为1, 26-40记为2, 41层以上记为3
"""
def floor_to_num(x):
    if x<=10:
        return 0
    elif x<=25:
        return 1
    elif x<=40:
        return 2
    else:
        return 3

data['总楼层']=data['总楼层'].apply(floor_to_num)

def subway_to_num(x):
    if len(x)==1:
        return int(x)
    else:
        return 1

data['地铁']=data['地铁'].apply(subway_to_num)


"""
非别墅区域平均价格
"""
data1 = data[data['别墅']==0]
zone_groups1 = data1['单价'].groupby(data1['区域'])

zone_price_mean = zone_groups1.mean()
zone_price_max = zone_groups1.max()
zone_price_min = zone_groups1.min()
zone_price = pd.concat([zone_price_min,zone_price_mean,zone_price_max],axis=1)
zone_price.columns = ['min','mean','max']


zone_price.plot(kind='bar',color=['k','r','b'],alpha=0.7,figsize=(13,6),title='成都市各区域二手房平均价格',fontsize=16)
plt.savefig('data/pic/static/zone_price_unit.jpg')
plt.ylabel('价格(元/平方米)',FontSize=12)
plt.show()
plt.close()

X = zone_price_mean.index.values
Y = zone_price_mean.values
plt.figure(figsize=(14,6))
plt.bar(X, Y, facecolor='r', edgecolor='k')
for x,y in zip(X,Y):
    plt.text(x,y*1.01, '%d' % y, ha='center', va= 'bottom')
plt.ylim(0,int(Y.max()*1.2))
plt.xlabel('区域',FontSize=12)
plt.ylabel('价格(元/平方米)',FontSize=12)
plt.title('成都市各区域二手房平均价格',FontSize=14)
plt.savefig('data/pic/static/zone_price_unit.jpg')
plt.show()
plt.close()


"""
别墅区域平均价格
"""
data2 = data[data['别墅']==1]
zone_groups2 = data2['单价'].groupby(data2['区域'])

zone_price1_mean = zone_groups2.mean()
zone_price1_max = zone_groups2.max()
zone_price1_min = zone_groups2.min()
zone_price1 = pd.concat([zone_price1_min,zone_price1_mean,zone_price1_max],axis=1)
zone_price1.columns = ['min','mean','max']


zone_price1.plot(kind='bar',color=['k','r','b'],alpha=0.7,figsize=(13,6),title='成都市各区域二手别墅价格',fontsize=16)
plt.savefig('data/pic/static/zone_price_villa_unit.jpg')
plt.ylabel('价格(元/平方米)',FontSize=12)
plt.show()
plt.close()

X = zone_price1_mean.index.values
Y = zone_price1_mean.values
plt.figure(figsize=(14,6))
plt.bar(X, Y, facecolor='r', edgecolor='k')
for x,y in zip(X,Y):
    plt.text(x,y*1.01, '%d' % y, ha='center', va= 'bottom')
plt.ylim(0,int(Y.max()*1.2))
plt.xlabel('区域',FontSize=12)
plt.ylabel('价格(元/平方米)',FontSize=12)
plt.title('成都市各区域二手别墅平均价格',FontSize=14)
plt.savefig('data/pic/static/zone_villa_price_unit.jpg')
plt.show()
plt.close()


X = zone_price1_mean.index.values
Y = zone_price1_mean.values
plt.figure(figsize=(14,6))
plt.bar(X, Y, facecolor='r', edgecolor='k')
for x,y in zip(X,Y):
    plt.text(x,y*1.01, '%d' % y, ha='center', va= 'bottom')
plt.ylim(0,int(Y.max()*1.2))
plt.xlabel('区域',FontSize=12)
plt.ylabel('价格(万元)',FontSize=12)
plt.title('成都市各区域二手别墅总售价',FontSize=14)
plt.savefig('data/pic/static/zone_villa_price_all.jpg')
plt.show()
plt.close()


"""
非别墅区域平均面积
"""
data1 = data[data['别墅']==0]
zone_groups1 = data1['面积'].groupby(data1['区域'])

zone_area_mean = zone_groups1.mean()

X = zone_area_mean.index.values
Y = zone_area_mean.values
plt.figure(figsize=(14,6))
plt.bar(X, Y, facecolor='r', edgecolor='k')
for x,y in zip(X,Y):
    plt.text(x,y*1.01, '%d' % y, ha='center', va= 'bottom')
plt.ylim(0,int(Y.max()*1.2))
plt.xlabel('区域',FontSize=12)
plt.ylabel('面积',FontSize=12)
plt.title('成都市各区域二手房平均面积',FontSize=14)
plt.savefig('data/pic/static/zone_area.jpg')
plt.show()
plt.close()


"""
别墅区域平均面积
"""
data2 = data[data['别墅']==1]
zone_groups2 = data2['面积'].groupby(data2['区域'])
zone_area1_mean = zone_groups2.mean()
X = zone_area1_mean.index.values
Y = zone_area1_mean.values
plt.figure(figsize=(14,6))
plt.bar(X, Y, facecolor='r', edgecolor='k')
for x,y in zip(X,Y):
    plt.text(x,y*1.01, '%d' % y, ha='center', va= 'bottom')
plt.ylim(0,int(Y.max()*1.2))
plt.xlabel('区域',FontSize=12)
plt.ylabel('面积',FontSize=12)
plt.title('成都市各区域二手别墅平均面积',FontSize=14)
plt.savefig('data/pic/static/zone_area_all.jpg')
plt.show()
plt.close()


"""
楼层价格绘图
"""
zone_groups = data['单价'].groupby(data['总楼层'])
zone_floor_mean = zone_groups.mean()
X = ['0-10','11-25','26-40','>40']
Y = zone_floor_mean.values
plt.figure(figsize=(8,4))
plt.bar(X, Y, facecolor='r', edgecolor='k')
for x,y in zip(X,Y):
    plt.text(x,y*1.01, '%d' % y, ha='center', va= 'bottom')
plt.ylim(0,int(Y.max()*1.2))
plt.xlabel('楼层',FontSize=12)
plt.ylabel('价格',FontSize=12)
plt.title('成都市不同楼层平均价格',FontSize=14)
plt.savefig('data/pic/static/zone_floor_all.jpg')
plt.show()
plt.close()

group = data[['单价','总价(万元)']].groupby(data['电梯'])
data_ele = group.mean().T


group =data[['单价','总价(万元)']].groupby(data['地铁'])
data_subway = group.mean().T



group = data[['总价(万元)']].groupby(data['卧室个数'])
data_restroom = group.mean().T


group =data[['总价(万元)']].groupby(data['客厅数'])
data_live = group.mean().T

group = data[['单价','总价(万元)']].groupby(data['小区名称'])
comm_num = group.size()
comm_num  = comm_num[comm_num .values>60]  #获取在售房屋数量超过60的小区及对应售房数量的Series
price_stati_all = group.mean()
price_stati = price_stati_all.loc[comm_num.index.values.tolist()]

stati_1 = price_stati.sort_values('单价')[-10:]['单价']

stati_2 = price_stati.sort_values('总价(万元)')[-10:]['总价(万元)']

stati_1.plot(kind='barh',color=['k','r','b','y','c'],alpha=0.7,figsize=(13,10),fontsize=10)
plt.savefig('data/pic/static/zone_price_unit_count.jpg')
plt.xlabel('单价(元/平方米)',FontSize=15)
plt.ylabel('小区名称',FontSize=15)
plt.title('成都市单价前十小区',FontSize=15)
plt.show()
plt.close()

stati_2.plot(kind='barh',color=['k','r','b','y','c'],alpha=0.7,figsize=(13,10),fontsize=10)
plt.savefig('data/pic/static/zone_price_all_count.jpg')
plt.xlabel('总价(万元)',FontSize=15)
plt.ylabel('小区名称',FontSize=15)
plt.title('成都市总价前十小区',FontSize=15)
plt.show()
plt.close()




