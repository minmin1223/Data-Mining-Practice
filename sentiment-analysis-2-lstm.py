#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


train_df = pd.read_csv("/kaggle/input/Reviews.csv")
train_df = train_df.loc[0:9999,["Score","Text"]]
test_df = pd.read_csv("/kaggle/input/test_2.csv")
print(test_df)
test_x = test_df.loc[:,"Text"]
def score_map(x):
    score = 1 if x >3 else 0
    return score

#注意这里传入的是函数名，不带括号
train_df["Score"] = train_df["Score"].map(score_map)
train_y = train_df["Score"]
train_x = train_df["Text"]
print(train_y)


# In[4]:


import seaborn as sns
import statistics as stat
sum_words = []
max_i = 0
for i in train_x:
    sum_words.append(len(i))
    if len(i) > max_i:
        max_i = len(i)

print("max_i",max_i)
print(len(train_x))

sns.displot(sum_words)
statData = list([9,3,3,1,2,1,1,1,2,4])
statData

min(statData)

max(statData)

sum(statData)


meansNb = stat.mean(sum_words)
print('平均數 : ', meansNb)

modeNb = stat.mode(sum_words)
print('眾數 : ', modeNb)

medianNb = stat.median(sum_words)
print('中位數 : ', medianNb)

stdevNb = stat.stdev(sum_words)
print('標準差 : ', stdevNb)

varianceNb = stat.variance(sum_words)
print('變異數 : ', varianceNb)


# 

# In[5]:


import tqdm
import re
def pre_process_corpus(docs):
    norm_docs = []
    for doc in tqdm.tqdm(docs):
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = doc.lower()
        # lower case and remove special characters\whitespaces
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()  
        norm_docs.append(doc)
  
    return norm_docs


# In[6]:


train_x = pre_process_corpus(train_x)
test_x = pre_process_corpus(test_x)


# In[7]:


from tensorflow.keras.preprocessing.text import Tokenizer
import math
max_words= 5000
tokenizer=Tokenizer(max_words)
tokenizer.fit_on_texts(train_x)

sequence_train=tokenizer.texts_to_sequences(train_x)
sequence_test=tokenizer.texts_to_sequences(test_x)



from tensorflow.keras.preprocessing.sequence import pad_sequences
list_len = math.ceil(meansNb+3*stdevNb)
data_train=pad_sequences(sequence_train,maxlen=list_len)

data_train = pd.DataFrame(data_train)

data_test=pad_sequences(sequence_test,maxlen=list_len)

data_test = pd.DataFrame(data_test)


# 

# In[8]:


import numpy 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense , Dropout , Activation
from keras.layers import LSTM , Embedding
from keras.layers import MaxPooling1D , GlobalMaxPooling1D,Conv1D , Flatten
from keras.datasets import imdb
from keras.preprocessing import text

model = Sequential()
model.add(Embedding(32, 3800, input_length = list_len))

model.add(LSTM(103))
#model.add(Dropout(0.7))

model.add(Dense(48, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())


# In[ ]:





# In[9]:


import datetime
def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today()
    
def plot_performance(history=None, figure_directory=None, ylim_pad=[0, 0]):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    plt.figure(figsize=(20, 5))

    y1 = hist['accuracy']

    min_y = min(y1)-ylim_pad[1]
    max_y = max(y1)+ylim_pad[1]


    plt.subplot(121)

    plt.plot(y1)


    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()

    y1 = hist['loss']


    min_y = min(y1)-ylim_pad[1]
    max_y = max(y1)+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()


# In[10]:


import tensorflow as tf
model.compile(loss ="BinaryCrossentropy",
              optimizer="adam",metrics=["accuracy"])
fit_model = model.fit(data_train , train_y,epochs=10,batch_size=32)


# 313/313 [==============================] - 22s 70ms/step - loss: 0.5515 - accuracy: 0.7616
# 313/313 [==============================] - 105s 277ms/step - loss: 0.5215 - accuracy: 0.7631

# In[ ]:





# In[ ]:


hist = pd.DataFrame(fit_model.history)
hist["epoch"] = fit_model.epoch
hist["epoch"] = hist["epoch"].apply(lambda x: x+1)

print(hist)
import matplotlib.pyplot as plt
plot_performance(history=hist)


# In[ ]:


test_y = model.predict(data_test)


# In[ ]:


print(len(test_y))
print(test_y)
test_y = pd.DataFrame(test_y)

sub = pd.DataFrame()
sub['ID'] = pd.Series(range(1,5001))
sub['Score'] = test_y
sub.to_csv('submission.csv',index=False)


# In[ ]:


count =0
print(len(test_y))
print(type(test_y))
for i in test_y.values:
    if i >0.6:
        count+=1
        
print(count)

