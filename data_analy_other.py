# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 21:03:48 2018

@author: Evan_He
"""

import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv('C:/Users/Evan_He/Desktop/scrap_ershoufang/data/clean_data.csv',encoding="utf8",index_col='链家id')

group = data[['总价(万元)','单价']].groupby(data['小区名称'])
group_dict = {}
for zone in set(data['区域']):
    group_dict[zone] = data[data['区域']==zone][['面积','总价(万元)']]

from sklearn.model_selection import train_test_split

area = group_dict['青羊']['面积'].values.reshape(-1,1)
price = group_dict['青羊']['总价(万元)'].values.reshape(-1,1)
X_train,X_test, y_train, y_test = train_test_split(area,price,test_size = 0.25,random_state=100)


batch_size = 50

input_node = 1 
output_node =1 

layer1_node = 6
layer2_node = 3

learning_rate = 0.8   
learning_rate_decay = 0.96  #设定基础学习率，设置其衰减

regularization_rate = 0.001  #损失函数中变量正则化的系数
training_steps = 2000 #训练轮数

moving_average_decay = 0.99 #滑动平均衰减数


def inference(input_tensor,avg_class,weight1,biases1,weight2,biases2,weight3,biases3):
    #返回预测结果
    if avg_class == None:
        #是否使用滑动平均
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
        #利用激活函数ReLu对隐藏层进行前向传播
        layer2 = tf.nn.relu(tf.matmul(layer1,weight2)+biases2)
        return tf.matmul(layer2,weight3)+biases3
    else:
        #含滑动平均的预测
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+\
                            avg_class.average(biases1))
        layer2 = tf.nn.relu(tf.matmul(layer1,avg_class.average(weight2))+\
                            avg_class.average(biases2))
        #利用激活函数ReLu对隐藏层进行前向传播
        return tf.matmul(layer2,avg_class.average(weight3))+avg_class.average(biases3)


def train():
    x = tf.placeholder(tf.float32,[None,input_node],name = 'x_input')
    #对输入值进行向量存放，其每个元素长度为input_node,None表示输入数据元素不限制
    y_label = tf.placeholder(tf.float32,[None,output_node],name = 'y_label')
    
    weight1 = tf.Variable(tf.truncated_normal([input_node,layer1_node],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[layer1_node]))  #会根据输入元素长度进行广播计算
    
    weight2 = tf.Variable(tf.truncated_normal([layer1_node,layer2_node],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[layer2_node]))
    
    weight3 = tf.Variable(tf.truncated_normal([layer2_node,output_node],stddev=0.1))
    biases3 = tf.Variable(tf.constant(0.1,shape=[output_node]))
    y = inference(x,None,weight1,biases1,weight2,biases2,weight3,biases3)
    
    global_step = tf.Variable(0,trainable = False) ##设定训练轮数的变量是不能训练更新的
    
    variable_average = tf.train.ExponentialMovingAverage(moving_average_decay,global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())#对所有训练变量进行滑动平均
    #得到的是对变量更新的滑动平均，
    average_y = inference(x,variable_average,weight1,biases1,weight2,biases2,weight3,biases3)
    
#    #交叉熵的计算
#    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_label)
#    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    mse = tf.reduce_mean(tf.square(y -y_label))
    #损失函数
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    regularization = regularizer(weight1)+regularizer(weight2)
    
    loss = mse + regularization  #加正则项的损失函数
    #学习率进行损失函数优化
    learning_train_rate = tf.train.exponential_decay(learning_rate,global_step ,\
                                                     4000/ batch_size,  ##需要迭代的总次数
                                                     learning_rate_decay) #tf.arg_max(
    
#不同的优化方法优化参数，针对不同的数据优化不同
#    train_step = tf.train.AdamOptimizer(learning_train_rate).minimize(loss,global_step = global_step)
#    train_step = tf.train.AdagradOptimizer(learning_train_rate).minimize(loss,global_step = global_step)
#    train_step = tf.train.AdadeltaOptimizer(learning_train_rate).minimize(loss,global_step = global_step)
#    train_step = tf.train.AdagradDAOptimizer(learning_train_rate).minimize(loss,global_step = global_step)
#    train_step = tf.train.ProximalGradientDescentOptimizer(learning_train_rate).minimize(loss,global_step = global_step)
#    train_step = tf.train.ProximalAdagradOptimizer(learning_train_rate).minimize(loss,global_step = global_step)
#    train_step = tf.train.RMSPropOptimizer(learning_train_rate).minimize(loss,global_step = global_step)
#    train_step = tf.train.FtrlOptimizer(learning_train_rate).minimize(loss,global_step = global_step)


    train_step = tf.train.GradientDescentOptimizer(learning_train_rate).minimize(loss,global_step = global_step)
#    
    #利用下述两条等价代码对反向传播系数进行更新
#    train_op = tf.group([train_step,variable_average_op])
    with tf.control_dependencies([train_step,variable_average_op]):
        train_op = tf.no_op(name = 'train')
    
    accuary_mse = tf.sqrt(tf.reduce_mean(tf.square(average_y -y_label))) 
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        validata_feed = {x: X_train, y_label: y_train}
    
        test_feed = {x: X_test, y_label:y_test}
    
        for i in range(training_steps):
            if i%500 ==0:
                validata_acc = sess.run(accuary_mse,feed_dict = validata_feed)
                print('after %d training steps, the validata accuary is %g' %(i,validata_acc))
            start = (i*batch_size)%(len(X_train))
            end = min(start+batch_size,len(X_train))
            
            sess.run(train_op,feed_dict = {x:X_train[start:end],y_label:y_train[start:end]})
        
        test_acc = sess.run(accuary_mse, feed_dict = test_feed)
        print('the test accuary is %g' %test_acc)
        
        
        
train()        
        
        
        
        
        
        
        
        
        
    