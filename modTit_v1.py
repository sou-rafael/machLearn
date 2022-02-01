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
from sklearn.impute import SimpleImputer
from scipy.stats import mode
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:/Users/rsf_o/Documents/machineLearning/Kaggle/Titanic/train.csv')

y = df['Survived']
orig_X = df.drop(['Survived', 'Cabin', 'Name', 'Ticket'], axis = 1)


    # Categóricas
cat = [col for col in orig_X if orig_X[col].dtype == 'object']

categoricas = orig_X[cat]
numericas = orig_X.drop(cat, axis = 1)




