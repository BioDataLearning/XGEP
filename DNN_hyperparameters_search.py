#!/usr/bin/python

from utils import *
import os,sys
import argparse
import pandas
import csv
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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tabulate import tabulate
np.random.seed(12345)

def getparas():
    parser = argparse.ArgumentParser(description="progrom usage")
    parser.add_argument("-e", "--posvec", type=str, help="Essential gene embedding vector")
    parser.add_argument("-n", "--negvec", type=str, help="Non-essential gene embedding vector")
    parser.add_argument("-o", "--out", type=str, help="The optimized hyperparameters")
    args = parser.parse_args()
    posfile = args.posvec
    negfile = args.negvec
    outfile = args.out
    return posfile,negfile,outfile

''' objective function for Bayesian Optimization'''
def objective(params):
    model =  get_DNN_model(params)
    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    fits = model.fit(X_train_data, Y_train_data, batch_size=2**int(params['batch_size']), epochs=100, shuffle=True, validation_data=(X_val_data,Y_val_data), callbacks=[earlystopper])
    auc = np.mean(fits.history['val_roc_auc'])
    loss = 1-auc 
    out_file = 'DNN_performance_with_diff_hyperparameter.csv'
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params])
    of_connection.close()
    return {'loss': loss, 'params': params, 'status': STATUS_OK}
    
def main():
    posfile,negfile,outfile = getparas()
    fileout = open(outfile, "w")
    global X_train_data,X_val_data,Y_train_data,Y_val_data
    X_train_data,Y_train_data,X_val_data,Y_val_data,X_test,Y_test = get_DNN_data(posfile,negfile)
    space = {
        'hdim': hp.quniform('hdim',32, 64, 16),
        'sdim': hp.quniform('sdim',16, 32, 16),
        'l2_reg': hp.loguniform('l2_reg', np.log(0.00001), np.log(0.1)),
        'tdim': hp.quniform('tdim', 8,16,1),
        'drop_out': hp.uniform('drop_out', 0.0, 1.0),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.2)),
        'batch_size': hp.quniform('batch_size', 7, 8, 1)
    }
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
    fileout.write(str(best))

if __name__ == '__main__':
        main()
