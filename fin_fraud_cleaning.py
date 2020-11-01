# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 23:49:54 2020

@author: micha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder,LabelEncoder


df = pd.read_csv("C:/Users/micha/Downloads/archive (2)/fraud_financial_project.csv",index_col=0)

print(df.describe()) # we have dataframe 1048575

print(df.isnull().sum()) # we have no null values
print(df.isnull().values.any())
print(df["type"].value_counts())

print(df["newbalanceDest"].describe()) # the numeric balances have no negative values which is good
# 1 - fraudulent transaction, 0 - non fradualent transaction
print(df["isFraud"].value_counts()/len(df["isFraud"])) # most of the values are 0


print(df["isFlaggedFraud"].value_counts()/len(df["isFlaggedFraud"])) # all of the values are 0 non fraudulent

# I want to binarize the categorical features. I will need to do this for my model
df_obj = df.dtypes == "object"
print(df_obj)

obj_columns = []
for i,j in enumerate(df_obj):
    if j:
        obj_columns.append(df_obj.index[i])
print(obj_columns)
print(df["nameDest"].value_counts())
dataframe_object = df.loc[:,obj_columns]
print(dataframe_object.head())

le = LabelEncoder()
enc = OneHotEncoder(categorical_features=df_obj,sparse=False)
dataframe_object = dataframe_object.apply(lambda col:le.fit_transform(col.astype(str)),axis=0,result_type="expand")

print(df.info())
print(dataframe_object["nameDest"].value_counts())
cols_type = df.dtypes != "object"
numerical_cols = []
for i,j in enumerate(cols_type):
    if j:
        numerical_cols.append(cols_type.index[i])
print(numerical_cols)
dataframe_numerical = df.loc[:,numerical_cols]
# I will now combine the binarize categorical features with my numerical features
clean_data = pd.concat([dataframe_object,dataframe_numerical],axis=1,sort=False)
print(clean_data.head())
clean_data.to_csv("finacial_fraud_clean.csv")    
# isflaggedfraud will be our target variables




