# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 19:21:23 2021

@author: janardhan
"""

import os

#set working directory
os.chdir('D:\Pandas')

#import necessary libraries
import numpy as np 
import pandas as pd
import seaborn as sns

#importing model ,perfomance metrics etc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix

#import the dataset 

data_stroke=pd.read_csv('strokedata.csv')

#copy of the dataset to work with 
data=data_stroke.copy()

#exploratory data analysis
"""
1.getting to know data
2.missing values
3.crosstab and visualisation
"""

#1.
print(data.info())

#missing values
print(data.isnull())

#sum of missing values in each column 
print(data.isnull().sum())

#summary of numerical 
sum_num=data.describe()
print(sum_num)

#summary of categorical 
sum_cat=data.describe(include="O")
print(sum_cat)


#unique value count in each column
data['gender'].value_counts()
data['smoking_status'].value_counts()

#list of all unique levels under a variable
np.unique(data['bmi'])
np.unique(data['gender'])
np.unique(data['id'])


#data_stroke=pd.read_csv('strokedata.csv',na_values=['?']) done to include ? as nan but for us all data perfect i.e no special chars
"""data_stroke=pd.read_csv('strokedata.csv',na_values=['?'])
data_stroke.isnull().sum()"""

data2=data.dropna(axis=0)

data2.isnull().sum()
#correlation between numerical vars
cor=data2.corr()

#gender proportion table

pd.crosstab(index=data2['gender'],columns='count',normalize=True)

#gender vs stroke 
pd.crosstab(index=data2['gender'],columns=data2['stroke'],margins=True,normalize='index')

#Frequency distribution of stroke status/bar plot
StrokeStat=sns.countplot(data2['stroke'])

#histogram of age .number of people in different age groups
sns.distplot(data2['age'],bins=10,kde=False)

#boxplot of age vs stroke 
sns.boxplot('stroke','age',data=data2)

#work_type vs stroke
pd.crosstab(index=data2['work_type'],columns=data2['stroke'],normalize=False,margins='index')

#LogisticRegression
new_data=pd.get_dummies(data=data2,drop_first=True)

#getting column names
columns_list=list(new_data.columns)
features=list(set(columns_list)-set(['stroke']))

y=new_data['stroke'].values
x=new_data[features].values

#data split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#instance of a model
logistic=LogisticRegression()
#model fit into training set
logistic.fit(train_x,train_y)

#prediction
pred=logistic.predict(test_x)
print(pred)

#perfomance of the model
cm=confusion_matrix(test_y,pred)
print(cm)
accuracy_score(test_y,pred)
print(accuracy_score)

#improvisation of model by removing insignificant variables
cols=['id','gender','work_type','Residence_type']

new_data=data2.drop(cols,axis=1)

#model repeated all over again
#LogisticRegression
new_data=pd.get_dummies(data=data2,drop_first=True)

#getting column names
columns_list=list(new_data.columns)
features=list(set(columns_list)-set(['stroke']))

y=new_data['stroke'].values
x=new_data[features].values

#data split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
#instance of a model
logistic=LogisticRegression()
#model fit into training set
logistic.fit(train_x,train_y)

#prediction
pred=logistic.predict(test_x)
print(pred)

#perfomance of the model
cm=confusion_matrix(test_y,pred)
print(cm)
accuracy_score(test_y,pred)
print(accuracy_score)

