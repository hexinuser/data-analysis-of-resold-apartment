# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 13:07:16 2018

@author: Evan_He
"""

import pandas as pd
import re

#data = pd.read_excel('C:/Users/Evan_He/Desktop/scrap_ershoufang/data/data.xlsx',encoding="utf8",index_col='链家id')
#data.to_csv('C:/Users/Evan_He/Desktop/scrap_ershoufang/data/house_data.csv',encoding="utf8")
data = pd.read_csv('C:/Users/Evan_He/Desktop/scrap_ershoufang/data/house_data.csv',encoding="utf8",index_col='链家id')

data['备注']=data['备注'].fillna(0)
data.dropna(axis=0, how='any', inplace=True)



data['小区名称']=data['小区名称'].apply(lambda x: x.strip())
data['房屋朝向']=data['房屋朝向'].apply(lambda x: x.strip())
data['电梯'] = data['电梯'].apply(lambda x: x.strip()[0])
data['装修'] =data['装修'].apply(lambda x: x.strip())
data['区域'] =data['区域'].apply(lambda x: x.strip())



data['布局']=data['布局'].apply(lambda x: x.strip())
data = data[data['布局']!='车位']
data['卧室个数']=data['布局'].apply(lambda x: x[0])
data['客厅数']=data['布局'].apply(lambda x: x[2])
data.drop('布局',axis=1,inplace=True)

data.rename(columns={'总价':'总价(万元)','备注':'地铁'},inplace =True)
data['单价']=data['单价'].apply(lambda x: int(re.compile('单价(.*?)元').findall(x)[0]))


def bieshu(x):
    if len(x)==1:
        return 0
    else:
        return 1
data['别墅'] = data['别墅'].apply(bieshu)   

data['面积']=data['面积'].apply(lambda x: float(x.strip()[:-2]))
data = data[data['面积']<=600]

def ele_or(x):
    if x=='无':
        return 0
    else:
        return 1
    
data['电梯'] = data['电梯'].apply(ele_or) #0表示无电梯，1表示有电梯

def floor_sum(x):
    try:
        return int(re.compile('共(.*?)层').findall(x)[0])
    except:
        return int(re.compile('(.*?)层').findall(x)[0])

data['总楼层'] = data['地址'].apply(floor_sum)

data.drop(['装修','地址'],axis=1,inplace=True)

data = data[data['单价']<=60000]

data.to_csv('C:/Users/Evan_He/Desktop/scrap_ershoufang/data/clean_data.csv',encoding="utf8")









