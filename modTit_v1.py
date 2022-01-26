# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 08:07:05 2022

@author: rsf_o
"""

import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error
#from sklearn.impute import SimpleImputer
from scipy.stats import mode
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:/Users/rsf_o/Documents/machineLearning/Kaggle/Titanic/train.csv')

y = df['Survived']
orig_X = df.drop(['Survived', 'Cabin', 'Name', 'Ticket'], axis = 1)


    # Categóricas
cat = [col for col in orig_X if orig_X[col].dtype == 'object']

categoricas = orig_X[cat]
numericas = orig_X.drop(cat, axis = 1)

categoricas['Embarked'].fillna(method = 'ffill', inplace = True)

    # Optei por remover a idade, pq preenchendo os NULOS sempre desequilibrava demais
numericas.drop(['Age'], axis = 1, inplace = True)


enc = OneHotEncoder(drop = 'first')

OH_X_treino = pd.DataFrame(enc.fit_transform(X_treino[cat]))
OH_X_treino.index = X_treino.index


# Juntando os dois dataFrames: categoricas + numericas
X = categoricas.join(numericas)


    #Split
X_treino, X_valid, y_treino, y_valid = train_test_split(X, y, train_size = 0.7, random_state = 0)

rf_model = RandomForestClassifier(max_leaf_nodes = 550,
                                 n_estimators = 1000,
                                 max_depth = 30)
rf_model.fit(X_treino, y_treino)

pred = rf_model.predict(X_teste)

