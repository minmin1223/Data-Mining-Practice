#!/usr/bin/env python
# coding: utf-8

# In[59]:


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


# In[60]:


from sklearn.feature_extraction.text import CountVectorizer
import gensim
df = pd.read_csv("/kaggle/input/amazon-fine-food-reviews/Reviews.csv")
data = df.loc[0:9999,["Score","Text"]]

def score_map(x):
    score = 1 if x >3 else 0
    return score

data["Score"] = data["Score"].map(score_map)
data["Score"][:20]


# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(data["Text"])
# array_X = X.toarray()
# print(X.shape)
# 
# #print(vectorizer.vocabulary_)
# unique, counts = np.unique(array_X, return_counts=True)
# the_dict = dict(zip(unique, counts))
# print(the_dict)
# print(type(X))

# In[61]:


import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')

 
stop_words = set(stopwords.words('english'))
print(stop_words)

data['Text'] = data['Text'].apply(lambda x: [word for word in x.lower().split(" ") if word not in stop_words and word.isalnum()])

print(data["Text"])
print(type(data["Text"]))


# from gensim.models import Word2Vec
# from sklearn.model_selection import GridSearchCV
# 
# parameters = {'size':[100,110,120],'window':[5,6,7],'iter':[5,10,15]}
# #csr_matrix
# #X = X.toarray()
# s_obj = gensim.models.Word2Vec(sentences = X, min_count=min_count)
# s_model = GridSearchCV(s_obj,parameters,cv=4)
# 
# 
# print(s_model.best_params_)
# 
# 
# 

# In[62]:


from gensim.models import Word2Vec

# 要分析的資料
sentences = data['Text'] 
# 詞向量維度，默認值為100，資料大可以調大一點
size = 100
# Maximum distance between the current and predicted word within a sentence.
window = 5
# 兩種模型選擇一種：0為CBOW（默認）, 1為Skip-Gram
sg = 0
# 解法選擇：0為Negative Sampling,>0為Hierarchical Softmax
hf = 0
# 用Negative Sampling時，負採樣的個數。默認值為5
negative = 5
# 用CBOW時，設定為0：xwxw為上下文的詞向量之和，設定1（默認值）：上下文的詞向量平均值
cbow_mean = 1
# 設定詞頻率小於某數，則刪除它（默認5）資料大則可以上調，10,20 accuracy為0.76
min_count = 10
#  Number of iterations (epochs) over the corpus（默認5）
epochs = 15

model = gensim.models.Word2Vec(sentences = sentences,epochs =epochs, min_count =min_count)

# min_count=5,window=5,epochs=10
# 0.7903 epoch=10,min_count=15
# 0.7932 epoch=10,min_count=10
# 0.7932 沒調
# 0.7935 epochs =15, min_count =5
# 0.7954 epochs =15, min_count =15
# 0.7889 epochs =20, min_count =15


# 0.7616
# 
# model_name = "genism"
# 要分析的資料
# sentences = X
# 詞向量維度，默認值為100，資料大可以調大一點
# size = 100
# Maximum distance between the current and predicted word within a sentence.
# window = 2
# 兩種模型選擇一種：0為CBOW（默認）, 1為Skip-Gram
# sg = 0
# 解法選擇：0為Negative Sampling,>0為Hierarchical Softmax
# hf = 0
# 用Negative Sampling時，負採樣的個數。默認值為5
# negative = 5
# 用CBOW時，設定為0：xwxw為上下文的詞向量之和，設定1（默認值）：上下文的詞向量平均值
# cbow_mean = 1
# 設定詞頻率小於某數，則刪除它（默認5）資料大則可以上調，10,20 accuracy為0.76
# min_count = 2
# Number of iterations (epochs) over the corpus（默認5）
# epochs = 2
# 

# 

# In[63]:


# it’s all of the words that appeared in the training data at least twice
print(len(model.wv.index_to_key))


# In[64]:


# 與family相似的詞，topn設定輸出的數量
model.wv.most_similar('family',topn =10)


# In[81]:


print(model.wv["family"])


# In[65]:


words = set(model.wv.index_to_key )
X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words])
                         for ls in data['Text'] ])


# In[92]:


X_train_vect_avg = []

#  we’ll end up with is now a single vector of length 100 that represents each text by averaging those word vectors for the words that were represented in that text message.
#print(X_train_vect) # 10000篇文章
print(X_train_vect.ndim) # 10000篇文章
print("X_train_vect[0]",X_train_vect[0].shape) # 18個詞 （每篇文章不一樣） 100個feature
#print(X_train_vect[0])
for v in X_train_vect:

    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))

    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))

#print(X_train_vect_avg[0]) # 10000篇文章
print("X_train_avg",len(X_train_vect_avg))
print("X_train_avg[0]",X_train_vect_avg[0].shape)
print(type(X_train_vect_avg[0]))


# In[67]:


count=0
for i, v in enumerate(X_train_vect_avg):
    if count <10:
        print(len(data["Text"].iloc[i]), len(v))
        count+=1
    
          
    # 第一個數字：一列Text裡有幾個字，第二個數字：100個vector去代表這一列文章中的字詞相近程度


# In[89]:


print(type(X_train_vect_avg))


# In[93]:


from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect_avg, data["Score"])



# In[94]:


from sklearn.model_selection import cross_val_score
t1 = datetime.now()
scores = cross_val_score(rf_model, X_train_vect_avg, data["Score"].values.ravel(), cv=4)
t2 = datetime.now()
time = t2-t1
print(time)


# In[95]:


print(t1,t2)


# In[70]:


print(scores)
print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))

