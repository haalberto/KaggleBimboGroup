# -*- coding: utf-8 -*-
"""

This script computes different features for predicting the demand of products
at different locations. This particular model is meant for cases when the
product is new, so it does not use specific product information.

Created on Aug 28 2016

@author: Alberto
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
    
def rms_error(y1,y2):
    """ Compute RMS Error."""
    return np.sqrt(np.mean((y1-y2)**2))

print 'Loading data...'
# Load training data (only the useful columns)
train=pd.read_csv('train.csv', header=0, usecols=[0,3,4,5,10])

# Number of validation examples (for each week)
n=100000

# Validate by holding out data from given week
val_week=3

print 'Validating on data from week', val_week
print 'Creating training and validation sets...\n'
# Create log demand column
train['Demanda_log']=np.log(train['Demanda_uni_equil'].values+1)    

# Create training set
train_n=train[train['Semana']!=val_week]

# Create validation set
np.random.seed(696)
val_n=train[train['Semana']==val_week]
n_size=len(val_n)
indices=np.random.choice(n_size,n, replace=False)
val_n=val_n.iloc[indices,:]

# Forget unnecessary data.
train_n=train_n.drop('Semana', 1)
train_n=train_n.drop('Demanda_uni_equil', 1)

# Compute features
gr_clie=train_n.groupby('Cliente_ID')
clie_mean=gr_clie['Demanda_log'].agg({'Cliente_mean' : np.mean}) #Mean for each costumer
c_punique=train_n.groupby('Cliente_ID').Producto_ID.nunique() #Number of different products ordered by costumer
c_punique=np.log(c_punique+1).to_frame('CnumP')


# Join features to the data frame
train_n=train_n.join(clie_mean, on='Cliente_ID')
val_n=val_n.join(clie_mean, on='Cliente_ID')
train_n=train_n.join(c_punique, on='Cliente_ID')
val_n=val_n.join(c_punique, on='Cliente_ID')

# Imputation of missing values in validation
C_mean=train_n['Cliente_mean'].mean()
CnumP_mean=train_n['CnumP'].mean()
new_c=val_n['Cliente_mean'].isnull().sum()
print 'Number of new costumers:', new_c
if new_c>0:
    val_n.loc[val_n['Cliente_mean'].isnull(), 'Cliente_mean']=C_mean
    val_n.loc[val_n['CnumP'].isnull(), 'CnumP']=CnumP_mean

# Forget more data
del train
del clie_mean
del c_punique
train_n=train_n.drop('Cliente_ID', 1)
train_n=train_n.drop('Producto_ID', 1)
train_n=train_n.drop('Ruta_SAK', 1)

t_index=train_n.index.values.copy()
v_index=val_n.index.values.copy()
y_train=train_n.Demanda_log.values
y_val=val_n.Demanda_log.values
X_train=train_n.values[:,1:]
X_val=val_n.values[:,6:]
del train_n
del val_n


# Feature normalization
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_val=scaler.transform(X_val)

# Train model for a range of regularization values
# Regression parameters
alphas=np.array([0.01, 0.03, 0.1, 0.3, 1]) # Regularization parameter values
#alphas=np.array([0.00001, 0.0001, 0.001, 0.01])
#alphas=np.array([0.01, 0.1, 1, 10])
n_iter=5 # Number of passes over the training set
error_best=100 #Dummy value of error
for alpha in alphas:
    # Train model
    print 'Training model with alpha =', alpha, '...'
    model=SGDRegressor(alpha=alpha, penalty='l2', n_iter=n_iter, verbose=1)
    model.fit(X_train, y_train)
    
    print ''
    print 'Model parameters'
    print 'Intercept:', model.intercept_
    print 'Coefficients:', model.coef_    
    
    # Validation
    y_pred=model.predict(X_val)
    print "Total RMS Log Error (without rounding):"
    error=rms_error(y_pred, y_val)
    print error
    print "Total RMS Log Error (with rounding and no negative numbers):"
    y_pred_r=y_pred
    y_pred_r[y_pred_r<0]=0
    y_pred_r=np.log((np.exp(y_pred_r)-1.).round()+1)    
    error_r=rms_error(y_val, y_pred_r)
    print error_r  
    print ""
    if error_best>error:
        error_best=error
        error_best_r=error_r
        best_model=model

print "Best model"
print 'Model parameters'
print 'Intercept:', model.intercept_
print 'Coefficients:', model.coef_
print "RMS Log Error (without rounding):"
print error_best
print "RMS Log Error (with rounding and no negative numbers):"
print error_best_r


joblib.dump(best_model, 'no_product_deg1.pkl')

    

    
    