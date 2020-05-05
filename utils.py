#!/usr/bin/python

import os,sys
import pandas
import numpy as np
from sklearn import model_selection
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
import xgboost as xgb
from sklearn.svm import SVC
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import set_random_seed
np.random.seed(12345)

'''get the data for cross valiation and prediction'''
def get_data(pos_file,neg_file,pred_file):
    pos_data = pandas.read_table(pos_file,sep=",",header=None).values
    neg_data = pandas.read_table(neg_file,sep=",",header=None).values
    pred_data = pandas.read_table(pred_file,sep=",",header=0).values
    train = np.vstack((pos_data,neg_data))
    np.random.shuffle(train)
    X = train[:,2:] 
    Y = train[:,0].astype('int')
    X_pred = pred_data[:,1:]
    Info_pred = pred_data[:,0]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minmax = min_max_scaler.fit_transform(X)
    X_pred_minmax = min_max_scaler.transform(X_pred)
    return X_minmax,Y,X_pred_minmax,Info_pred

'''get the data for deep learning model training'''
def get_DNN_data(pos_file,neg_file):
    pos_data = pandas.read_table(pos_file,sep=",",header=None).values
    neg_data = pandas.read_table(neg_file,sep=",",header=None).values
    train = np.vstack((pos_data,neg_data))
    np.random.shuffle(train)
    validation_size = 0.10
    train, test = train_test_split(train, test_size=validation_size)
    train, val = train_test_split(train, test_size=validation_size)
    train_pos = train[train[:,0]==1]
    train_neg = train[train[:,0]==0]
    train_pos_boot = resample(train_pos, replace=True, n_samples=train_neg.shape[0])
    train_com = np.vstack((train_pos_boot,train_neg))
    np.random.shuffle(train_com)
    X_train = np.array(train_com[:,2:])
    Y_train = np.array(train_com[:,0].astype('int'))
    X_val = np.array(val[:,2:])
    Y_val = np.array(val[:,0].astype('int'))
    X_test = np.array(test[:,2:])
    Y_test = np.array(test[:,0].astype('int'))
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_val_minmax = min_max_scaler.transform(X_val)
    X_test_minmax = min_max_scaler.transform(X_test)
    return X_train_minmax,Y_train,X_val_minmax,Y_val,X_test_minmax,Y_test

'''get the data for deep learning prediction'''
def get_DNN_pred_data(pos_file,neg_file,pred_file):
    pos_data = pandas.read_table(pos_file,sep=",",header=None).values
    neg_data = pandas.read_table(neg_file,sep=",",header=None).values
    pred_data = pandas.read_table(pred_file,sep=",",header=0).values
    train = np.vstack((pos_data,neg_data))
    np.random.shuffle(train)
    validation_size = 0.10
    train, val = train_test_split(train, test_size=validation_size)
    train_pos = train[train[:,0]==1]
    train_neg = train[train[:,0]==0]
    train_pos_boot = resample(train_pos, replace=True, n_samples=train_neg.shape[0])
    print(train_pos_boot.shape)
    train_com = np.vstack((train_pos_boot,train_neg))
    np.random.shuffle(train_com)
    X_train = np.array(train_com[:,2:])
    Y_train = np.array(train_com[:,0].astype('int'))
    X_val = np.array(val[:,2:])
    Y_val = np.array(val[:,0].astype('int'))
    X_pred = pred_data[:,1:]
    Info_pred = pred_data[:,0]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_val_minmax = min_max_scaler.transform(X_val)
    X_pred_minmax = min_max_scaler.transform(X_pred)
    return X_train_minmax,Y_train,X_val_minmax,Y_val,X_pred_minmax,Info_pred
 
'''Calculate ROC AUC during model training, obtained from <https://github.com/nathanshartmann/NILC-at-CWI-2018>'''
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

''' P_TA prob true alerts for binary classifier'''
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P

def roc_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

'''get the PRAUC values'''
def PRAUC(Yture,Ypred):
    precision, recall, _ = precision_recall_curve(Yture,Ypred)
    prauc = auc(recall,precision)
    return(prauc)

'''parameter tuning for XGB and SVM models'''
def para_tuning(model,para,X,Y):
    grid_obj = GridSearchCV(model, para, scoring = 'average_precision',cv=5)
    grid_obj = grid_obj.fit(X, Y)
    para_best = grid_obj.best_estimator_
    return(para_best)

'''calculate the metrics for model evaluation'''
def metrics(Y_test, rounded, pred_train_prob, fileout):
    confusion = confusion_matrix(Y_test, rounded)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    sepcificity = TN / float( TN + FP)
    sensitivity = TP / float(FN + TP)
    mcc = matthews_corrcoef(Y_test, rounded)
    f1 = f1_score(Y_test, rounded)
    fpr, tpr, thresholds = roc_curve(Y_test, pred_train_prob)
    aucvalue = auc(fpr, tpr)
    prauc = PRAUC(Y_test,pred_train_prob)
    return accuracy_score(Y_test, rounded),sepcificity,sensitivity,mcc,f1,aucvalue,prauc

'''get the DNN model'''
def get_DNN_model(params):
    model = Sequential()
    model.add(Dense(int(params['hdim']), input_dim=150,activation="relu",kernel_regularizer=regularizers.l2(params['l2_reg'])))
    model.add(Dropout(params['drop_out']))
    model.add(Dense(int(params['sdim']), activation="relu",kernel_regularizer=regularizers.l2(params['l2_reg'])))
    model.add(Dropout(params['drop_out']))
    model.add(Dense(int(params['tdim']), activation="relu",kernel_regularizer=regularizers.l2(params['l2_reg'])))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=params['learning_rate'],epsilon=10**-8)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[roc_auc])
    print(params)
    return model




