#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


data = pd.read_csv("/kaggle/input/movie-rating/movieRating.csv")
print(data)


# In[3]:


max_num_user = np.max(data["UserID"]) #UserID最大的值
max_num_movie = np.max(data["MovieID"])#MovieID最大的值
print(max_num_user,max_num_movie,len(data))


# In[4]:


# 資料input分佈
the_set = set()
user_count = 0
movie_count = 0
for i in range(len(data)):
    if data.loc[i,"UserID"] not in the_set:
        the_set.add(data.loc[i,"UserID"])
        user_count+=1
    if data.loc[i,"MovieID"] not in the_set:
        the_set.add(data.loc[i,"MovieID"])
        movie_count+=1
        

print(user_count)      
print(movie_count) 


# In[5]:


# 資料隨機分成8:2
from sklearn.model_selection import train_test_split


data_split = train_test_split(data, test_size=0.2, shuffle=True)


# In[6]:


print(len(data_split))
train = data_split[0]
test = data_split[1]
print("train_len",len(train))
print("test_len",len(test))


# In[7]:


print(train.info())
print(train.head())


# In[8]:


from keras import Model
import keras.backend as K
from keras.layers import Embedding,Reshape,Input,Dot


num_user = max_num_user
num_movie = max_num_movie

def Recmand_model(num_user,num_movie,k):
    input_user = Input(shape=[None,],dtype="int32")
    print(type(input_user))
    model_user = Embedding(num_user+1,k,input_length = 2)(input_user) # input轉成向量
    print(model_user.shape)
    model_user = Reshape((k,))(model_user) #轉成一維張量，讓它可以與movie資料點乘。
    
    input_movie = Input(shape=[None,],dtype="int32")
    model_movie  = Embedding(num_movie+1,k,input_length = 2)(input_movie)
    model_movie = Reshape((k,))(model_movie)#轉成一維張量。
    
    out = Dot(1)([model_user,model_movie]) #資料點乘
    model = Model(inputs=[input_user,input_movie], outputs=out)
    model.compile(loss='mse', optimizer='Adam')
    model.summary()
    return model


# In[9]:


# 建立模型
model = Recmand_model(num_user,num_movie,5000)


# In[10]:


# 處理train, test資料
train_user,train_movie = train["UserID"].values, train["MovieID"].values
train_x = [train_user,train_movie]
train_y = train["Rating"].values

test_user,test_movie = test["UserID"].values, test["MovieID"].values
test_x = [test_user,test_movie]
test_y = test["Rating"].values


# In[11]:


# fit model
model.fit(train_x,train_y,batch_size = 1000,epochs =10)


# In[12]:


# 預測測試資料
from sklearn.metrics import mean_absolute_error
predict = model.predict(test_x)
print(len(predict))


# In[13]:


mae = mean_absolute_error(predict, test_y)
print(mae)

