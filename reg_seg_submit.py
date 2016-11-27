# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 13:46:52 2016

Combines different models to predict demand of products.

The models were created by the following scripts:
reg_seg_naive.py, reg_seg_noprod.py, reg_seg.nocust.py, reg_seg._nocp.py

@author: Alberto
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

def rms_log_error(y1,y2):
    """ Compute RMS Logarithmic Error."""
    return np.sqrt(np.mean(np.log((1+y1)/(1+y2))**2))

#Load training data
train=pd.read_csv('train.csv', header=0, usecols=[0,3,4,5,10])

# Validate by holding out data from given week
val_week=3

print 'Validating on data from week', val_week
print 'Creating training and validation sets...\n'
# Create log demand column
train['Demanda_log']=np.log(train['Demanda_uni_equil'].values+1)    

# Create training set
train_n=train[train['Semana']!=val_week]

# Forget unnecessary data.
train_n=train_n.drop('Semana', 1)
train_n=train_n.drop('Demanda_uni_equil', 1)

# Compute features
gr_prod=train_n.groupby('Producto_ID')
prod_mean=gr_prod['Demanda_log'].agg({'Producto_mean' : np.mean}) #Mean for each product
gr_clie=train_n.groupby('Cliente_ID')
clie_mean=gr_clie['Demanda_log'].agg({'Cliente_mean' : np.mean}) #Mean for each costumer
gr_cp=train_n[['Cliente_ID', 'Producto_ID', 'Demanda_log']].groupby(['Cliente_ID', 'Producto_ID'])
cp_mean=gr_cp['Demanda_log'].agg({'CP_mean':np.mean}) #Mean for each costumer-product combination
gr_pr=train_n[['Ruta_SAK', 'Producto_ID', 'Demanda_log']].groupby(['Ruta_SAK', 'Producto_ID'])
pr_mean=gr_pr['Demanda_log'].agg({'PR_mean':np.mean}) #Mean for each route-product combination
c_punique=train_n.groupby('Cliente_ID').Producto_ID.nunique() #Number of different products ordered by costumer
c_punique=np.log(c_punique+1).to_frame('CnumP')
p_cunique=train_n.groupby('Producto_ID').Cliente_ID.nunique() #Number of costumers that order that product
p_cunique=np.log(p_cunique+1).to_frame('PnumC')

# Join features to the data frame
train_n=train_n.join(clie_mean, on='Cliente_ID')
train_n=train_n.join(prod_mean, on='Producto_ID')
train_n=train_n.join(cp_mean, on=['Cliente_ID', 'Producto_ID'])
train_n=train_n.join(pr_mean, on=['Ruta_SAK', 'Producto_ID'])
train_n=train_n.join(c_punique, on='Cliente_ID')
train_n=train_n.join(p_cunique, on='Producto_ID')

# Save some means for imputation of test data
C_mean=train_n['Cliente_mean'].mean()
P_mean=train_n['Producto_mean'].mean()
CP_mean=train_n['CP_mean'].mean()
CnumP_mean=train_n['CnumP'].mean()
PnumC_mean=train_n['PnumC'].mean()
PR_mean=train_n['PR_mean'].mean()

# Forget more data
del train
train_n=train_n.drop('Cliente_ID', 1)
train_n=train_n.drop('Producto_ID', 1)
train_n=train_n.drop('Ruta_SAK', 1)
# Create training matrix, delete dataframe
X_train=train_n.values[:,1:]
del train_n

# Feature normalization, delete training matrix
scaler=StandardScaler()
scaler.fit(X_train)
del X_train

# Load test data
test=pd.read_csv('test.csv', header=0, usecols=[0,4,5,6])

# Join means to the data frame
test=test.join(clie_mean, on='Cliente_ID')
test=test.join(prod_mean, on='Producto_ID')
test=test.join(cp_mean, on=['Cliente_ID', 'Producto_ID'])
test=test.join(pr_mean, on=['Ruta_SAK', 'Producto_ID'])
test=test.join(c_punique, on='Cliente_ID')
test=test.join(p_cunique, on='Producto_ID')



# Split test set based on missing values

# 1) Use full model if (costumer-product) pair information is available
fm_indices=~test['CP_mean'].isnull().values # Apply full model to these
print 'Full model will be applied to', fm_indices.sum(), 'records.'
print 'This is', fm_indices.sum()/(1.*len(test))*100, '%.'

# 2) CP-pair missing, but costumer and product not new
c_indices=~test['Cliente_mean'].isnull().values
p_indices=~test['Producto_mean'].isnull().values

sf_indices=(c_indices*p_indices)*~fm_indices
print 'Costumer and product known, but not combination:', sf_indices.sum() 
print 'This is', sf_indices.sum()/(1.*len(test))*100, '%.'

# 3) Model with only costumer information
onlyc_indices=c_indices*(~p_indices) #Apply costumer model to these
print 'Only costumer information in', onlyc_indices.sum(), 'records.'
print 'This is', onlyc_indices.sum()/(1.*len(test))*100, '%.'

# 4) Model with only product information
onlyp_indices=p_indices*(~c_indices) #Apply product model to these
print 'Only costumer information in', onlyp_indices.sum(), 'records.'
print 'This is', onlyp_indices.sum()/(1.*len(test))*100, '%.'

# List records not classified
simple_indices=(~c_indices*~p_indices) # Remaining records
print 'Remaining records:', simple_indices.sum()
print 'This is', simple_indices.sum()/(1.*len(test))*100, '%.' 

# Imputation of missing test data
new_c=test['Cliente_mean'].isnull().sum()
if new_c>0:
    test.loc[test['Cliente_mean'].isnull(), 'Cliente_mean']=C_mean
    test.loc[test['CnumP'].isnull(), 'CnumP']=CnumP_mean
new_p=test['Producto_mean'].isnull().sum()
if new_p>0:
    test.loc[test['Producto_mean'].isnull(), 'Producto_mean']=P_mean
new_cp=test['CP_mean'].isnull().sum()
if new_cp>0:
    test.loc[test['CP_mean'].isnull(), 'CP_mean']=CP_mean
    test.loc[test['PnumC'].isnull(), 'PnumC']=PnumC_mean
new_pr=test['PR_mean'].isnull().sum()
print 'Number of new (route,product) pairs:', new_pr
if new_pr>0:
    test.loc[test['PR_mean'].isnull(), 'PR_mean']=PR_mean

# Feature normalization
X_test=test.values[:,4:]
X_test=scaler.transform(X_test)

## Apply models

# 0) Predict same value for all records.
best_value=joblib.load('fixed_value.pkl')
y_pred=np.ones(len(test))*best_value

# 1) Full model

X_sub=X_test[fm_indices,:]
f_model=joblib.load('full_deg1.pkl')
y_vals=f_model.predict(X_sub)
y_pred[fm_indices]=y_vals

# 2) CP-pair missing, but costumer and product not new
X_sub=X_test[sf_indices,:]
X_sub=X_sub[:,[0,1,3,4,5]]
sf_model=joblib.load('semi_full.pkl')
y_vals=sf_model.predict(X_sub)
y_pred[sf_indices]=y_vals

# 3) Model with only costumer information
X_sub=X_test[onlyc_indices,:]
X_sub=X_sub[:,[0,4]]
c_model=joblib.load('no_product_deg1.pkl')
y_vals=c_model.predict(X_sub)
y_pred[onlyc_indices]=y_vals

# 4) Model with only product information
X_sub=X_test[onlyp_indices,:]
X_sub=X_sub[:,[1,3,5]]
p_model=joblib.load('no_costumer_deg1.pkl')
y_vals=p_model.predict(X_sub)
y_pred[onlyp_indices]=y_vals

# Final processing

# Eliminate negative numbers
y_pred[y_pred<0]=0
# Convert to actual demand (not log)
y_pred=(np.exp(y_pred)-1.)
# Create data frame with predictions
n_tests=len(y_pred)
predictions=pd.DataFrame({'id' : range(n_tests), 'Demanda_uni_equil' : y_pred}, columns=['id', 'Demanda_uni_equil'])
# Save predictions
predictions.to_csv('predictions_mixed.csv', index=False) 