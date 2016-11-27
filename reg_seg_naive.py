# -*- coding: utf-8 -*-
"""

This script predicts demand by assigning the same number to all the data.
It finds the best such number.

Created on Sat Aug 20 12:11:40 2016

@author: Alberto
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
  
def rms_error(y1,y2):
    """ Compute RMS Error."""
    return np.sqrt(np.mean((y1-y2)**2))

print 'Loading data...'
# Load training data (only the useful columns)
train=pd.read_csv('train.csv', header=0, usecols=[0,10])

# Validate by holding out data from given week
val_week=9

print 'Validating on data from week', val_week
print 'Creating training and validation sets...\n'
# Create log demand column
train['Demanda_log']=np.log(train['Demanda_uni_equil'].values+1)    

# Create training set
train_n=train[train['Semana']!=val_week]

# Create validation set
#np.random.seed(696)
val_n=train[train['Semana']==val_week]
val_size=len(val_n)
#indices=np.random.choice(n_size,n, replace=False)
#val_n=val_n.iloc[indices,:]
#val_n=train

log_mean=train_n['Demanda_log'].mean()
#mean=np.exp(log_mean)-1 #Converted to regular units
#mean_r=mean.round()
#means=np.array([mean_r-1, mean_r, mean_r+1])
#means_log= np.log(means+1) #Conver to log again
#
#y_val=val_n['Demanda_log'].values
## Find the integer that minimizes the error
#error_best=100 #Dummy value of error
#for value in means_log:
#    print 'Model that predicts the same value', np.exp(value)-1, 'for all records...'
#    # Validation
#    y_pred=np.ones(val_size)*value
#    print "RMS Log Error (without rounding):"
#    error=rms_error(y_pred, y_val)
#    print error
#    print "RMS Log Error (with rounding and no negative numbers):"
#    y_pred_r=y_pred
#    y_pred_r[y_pred_r<0]=0
#    y_pred_r=np.log((np.exp(y_pred_r)-1.).round()+1)    
#    error_r=rms_error(y_val, y_pred_r)
#    print error_r
#    print ""
#    if error_best>error:
#        error_best=error
#        error_best_r=error_r
#        best_value=value
#
#print "Best model. Predict this value for all records:"
#print best_value
#print "In real demand, this corresponds to", np.exp(best_value)-1
#print "RMS Log Error (without rounding):"
#print error_best
#print "RMS Log Error (with rounding and no negative numbers):"
#print error_best_r

joblib.dump(log_mean, 'fixed_value.pkl')
