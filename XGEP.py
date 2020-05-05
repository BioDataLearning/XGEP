#!/usr/bin/python

from utils import *
import os,sys
import argparse
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

def getparas():
    parser = argparse.ArgumentParser(description="progrom usage")
    parser.add_argument("-e", "--posvec", type=str, help="Essential gene embedding vector")
    parser.add_argument("-n", "--negvec", type=str, help="Non-essential gene embedding vector")
    parser.add_argument("-u", "--unannovec", type=str, help="Embedding vector for genes not in the training dataset")
    parser.add_argument("-o", "--out", type=str, help="The output of evaluation metrics")
    parser.add_argument("-p", "--predp", type=str, help="The predicted probability for genes not in the training dataset")
    args = parser.parse_args()
    posfile = args.posvec
    negfile = args.negvec
    predfile = args.unannovec
    outfile = args.out
    outfile2 = args.predp
    return posfile,negfile,predfile,outfile,outfile2

'''perform cross validation'''
def cv(model, X_train, Y_train,fileout):
    cv_acc, cv_sep, cv_sen, cv_mcc, cv_f1, cv_auc, cv_prauc = [],[],[],[],[],[],[]
    kf = model_selection.KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(X_train):
        X_cv_train, X_cv_test = X_train[train_index,:], X_train[test_index,:]
        Y_cv_train, Y_cv_test = Y_train[train_index], Y_train[test_index]
        model.fit(X_cv_train,Y_cv_train)
        predictions = model.predict(X_cv_test)
        pred_train_prob = model.predict_proba(X_cv_test)[:, 1]
        accuracy, sepcificity, sensitivity, mcc, f1, aucvalue, prauc = metrics(Y_cv_test, predictions, pred_train_prob, fileout)
        cv_acc.append(accuracy)
        cv_sep.append(sepcificity)
        cv_sen.append(sensitivity)
        cv_mcc.append(mcc)
        cv_f1.append(f1)
        cv_auc.append(aucvalue)
        cv_prauc.append(prauc)
    fileout.write(str(model)+"\n"+"Accuracy_mean: "+str(np.mean(cv_acc))+"\n"+"Sepcificity_mean: "+str(np.mean(cv_sep))+"\n"+"Sensitivity_mean: "+str(np.mean(cv_sen))+"\n"+"MCC_mean: "+str(np.mean(cv_mcc))+"\n"+"Fscore_mean: "+str(np.mean(cv_f1))+"\n"+"AUC_mean: "+str(np.mean(cv_auc))+"\n"+"PRAUC_mean: "+str(np.mean(cv_prauc))+"\n")

def machine_learning(posfile,negfile,predfile,fileout):
    X_train, Y_train, X_pred, Info_pred = get_data(posfile,negfile,predfile)
    para_svm = {'C' : [0.25,1.0,2.0,4.0,8.0,16.0,32.0], 
            'kernel' : ['rbf'],
            'gamma' : [0.001,0.01,0.1,0.5,1.0,2.0,4.0,8.0],
            'class_weight' : [{1:6.2}],
            'probability' : [True]
           }
    para_xgb = {
           'n_estimators': [50,100,200,300,400,500,600], 
           'max_depth': [5,10,15,20,25,30],
           'learning_rate': [0.001,0.01,0.05,0.1],
           'scale_pos_weight':[6.2]
          }
    best_para_svm = para_tuning(SVC(), para_svm, X_train, Y_train)
    best_para_xgb = para_tuning(xgb.XGBClassifier(), para_xgb, X_train, Y_train)
    cv(best_para_svm,X_train,Y_train,fileout)
    cv(best_para_xgb,X_train,Y_train,fileout)
    best_para_svm.fit(X_train, Y_train)
    SVM_predictions = best_para_svm.predict(X_pred)
    SVM_pred_prob = best_para_svm.predict_proba(X_pred)[:, 1]
    best_para_xgb.fit(X_train, Y_train)
    XGB_predictions = best_para_xgb.predict(X_pred)
    XGB_pred_prob = best_para_xgb.predict_proba(X_pred)[:, 1]    
    return SVM_predictions,SVM_pred_prob,XGB_predictions,XGB_pred_prob,Info_pred

def deep_learning(posfile,negfile,predfile,fileout):
    acc, sep, sen, mcc, f1, auc, prauc = [],[],[],[],[],[],[]
    best = {'batch_size': 8.0, 'drop_out': 0.10680339747442835, 'hdim': 48.0, 'l2_reg': 0.00024102301670176588, 'learning_rate': 0.0012709235952012008, 'sdim': 32.0, 'tdim': 11.0}
    model = get_DNN_model(best)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    for i in range(10):
        X_train,Y_train,X_val,Y_val,X_test,Y_test = get_DNN_data(posfile,negfile)
        model.fit(X_train, Y_train, batch_size=2**int(best['batch_size']), epochs=100, shuffle=True, validation_data=(X_val,Y_val), callbacks=[earlystopper])
        predictions = model.predict(X_test)
        rounded = [round(x[0]) for x in predictions]
        pred_train_prob = [x[0] for x in model.predict_proba(X_test)]
        accuracy, sepcificity, sensitivity, mccvalue, f1value, aucvalue, praucvalue = metrics(Y_test, rounded, pred_train_prob, fileout)
        acc.append(accuracy)
        sep.append(sepcificity)
        sen.append(sensitivity)
        mcc.append(mccvalue)
        f1.append(f1value)
        auc.append(aucvalue)
        prauc.append(praucvalue)
    fileout.write("DNN\n"+"Accuracy_mean: "+str(np.mean(acc))+"\n"+"Sepcificity_mean: "+str(np.mean(sep))+"\n"+"Sensitivity_mean: "+str(np.mean(sen))+"\n"+"MCC_mean: "+str(np.mean(mcc))+"\n"+"Fscore_mean: "+str(np.mean(f1))+"\n"+"AUC_mean: "+str(np.mean(auc))+"\n"+"PRAUC_mean: "+str(np.mean(prauc))+"\n")
    X_train, Y_train, X_val, Y_val, X_pred, Info_pred = get_DNN_pred_data(posfile,negfile,predfile)
    model.fit(X_train, Y_train, batch_size=2**int(best['batch_size']), epochs=100, shuffle=True, validation_data=(X_val,Y_val), callbacks=[earlystopper])
    predictions = model.predict(X_pred)
    rounded = [round(x[0]) for x in predictions]
    pred_train_prob = [x[0] for x in model.predict_proba(X_pred)]
    return rounded, pred_train_prob    

def main():
    posfile,negfile,predfile,outfile,outfile2 = getparas()
    fileout = open(outfile, "w")
    SVM_predictions,SVM_pred_prob,XGB_predictions,XGB_pred_prob,Info_pred = machine_learning(posfile,negfile,predfile,fileout)
    DNN_predictions,DNN_pred_prob = deep_learning(posfile,negfile,predfile,fileout)
    predprob = {'Gene': Info_pred, 'SVM_prediction': SVM_predictions, 'SVM_pred_prob': SVM_pred_prob, 
                                   'XGB_prediction': XGB_predictions, 'XGB_pred_prob': XGB_pred_prob, 
                                   'DNN_prediction': DNN_predictions, 'DNN_pred_prob': DNN_pred_prob}
    df_pred = pandas.DataFrame(predprob,columns=predprob.keys())
    df_pred.to_csv(outfile2,index=False) 
    

if __name__ == '__main__':
        main()
