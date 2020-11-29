# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 14:16:53 2020

@author: jbigg
"""

import numpy as np

import pickle

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE, RFECV

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, train_test_split

from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score, classification_report

df = pd.read_csv("breast_cancer_data.csv", header=0)

drop_features = ['id', 'Unnamed: 32']

data = df.drop(drop_features, axis=1)

feats = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']

X = data[feats]

y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

RFC = RandomForestClassifier()      

RFC1 = RFC.fit(X_train,y_train)

pickle.dump(RFC1, open('model1.pkl','wb'))

