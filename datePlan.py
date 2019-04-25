# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:34:48 2019

@author: surbhi
"""

import pandas as pd

data = pd.read_csv("zomato.csv")
#showing top 5 record
print data.head()


#Visualize the data in percentage from different countries
countries_code = data['Country Code'].unique()
tot_countries = []
for c in countries_code:
    x = data[data['Country Code']==c]
    n = x['Country Code'].count()
    tot_countries.append(n)

import matplotlib.pyplot as plt
plt.pie(tot_countries, labels = countries_code, autopct='%.0f%%', radius = 2)
plt.show()


#getting the data of india
dataIndia = data[data['Country Code']==1]
dataIndia = dataIndia.reset_index(drop=True)
#resetting the indices
print dataIndia.head()

#Visualize the data in percentage for different cities of India
tot_cities = dataIndia['City'].unique()
city_count = []
for city in tot_cities:
    xx = dataIndia[dataIndia['City']==city]
    m = xx['City'].count()
    city_count.append(m)
    
plt.pie(city_count, labels = tot_cities, autopct='%.0f%%', radius = 2)
plt.show()

#checking for null values if any
dataIndia.isnull().values.any()

#for the column Average Cost for two we are replacing with the average of the column
dataIndia['Average Cost for two']=dataIndia['Average Cost for two'].replace(0,dataIndia['Average Cost for two'].mean())

#we can see that the 'Cuisines' Column has multiple values so we will apply one hot encoding
df = pd.concat([dataIndia, dataIndia['Cuisines'].str.split(',',expand=True)], axis = 1)
df = df.rename(columns={0:'Cuisine'}) 
print df['Cuisine'].unique()
#keeping the main cusine of each restaurant
df = df.drop(1,axis =1) 
df = df.drop(2,axis =1)
df = df.drop(3,axis =1)
df = df.drop(4,axis =1)
df = df.drop(5,axis =1)
df = df.drop(6,axis =1)
df = df.drop(7,axis =1)


#removing unnecessary data
newIndia = df.drop(['Restaurant ID','Restaurant Name','Longitude', 'Latitude','Cuisines', 'Country Code', 'Rating color', 'Switch to order menu', 'Currency', 'Address', 'Locality', 'Locality Verbose'],  axis =1)
#two columns have been pop
cols = list(newIndia.columns.values)
cols.pop(cols.index('Rating text'))
cols.pop(cols.index('Aggregate rating'))
#Rearrangement of columns
newIndia = newIndia[cols+['Rating text']+['Aggregate rating']]
newIndia.head()


#Making features and labels out of the data
features = newIndia.iloc[:,1:9].values
labels = newIndia.iloc[:,9].values


#Label Encoding 
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
features[:,1]=lb.fit_transform(features[:,1])
features[:,2]=lb.fit_transform(features[:,2])
features[:,3]=lb.fit_transform(features[:,3])
features[:,7]=lb.fit_transform(features[:,7])
features[:,6]=lb.fit_transform(features[:,6])


#Applying one hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[6])
features = ohe.fit_transform(features).toarray()

#Spliting the data for training and testing
from sklearn.model_selection import train_test_split as tts
features_train,features_test,labels_train,labels_test= tts(features,labels,random_state=0,test_size=0.2)

#Fitting Decission Tree Regression to the model
from sklearn.tree import DecisionTreeRegressor
rs=DecisionTreeRegressor(random_state=0)
rs.fit(features_train,labels_train)
#predicted labels
pred_labels = rs.predict(features_test)
print pred_labels

#checking the accuracy of the model
score = rs.score(features_test,labels_test)
print "Accuracy "+ str(score*100)


import numpy as np
toPred = np.array([700,1,1,1,6,600,"Asian",4]).reshape(1,-1)
toPred[:,6] = lb.transform(toPred[:,6])
toPred = ohe.transform(toPred).toarray()
rate = rs.predict(toPred)

rate2 = int(rate - 1)
rate3 = int(rate + 1)

City = "Gurgaon"
Locality = "Sector 31"

res_df = dataIndia[(dataIndia['Aggregate rating'] >= rate2) & (dataIndia['Aggregate rating'] <=rate3)&(dataIndia['City'] ==City)&(dataIndia['Locality'] ==Locality)]
print res_df["Restaurant Name"]

