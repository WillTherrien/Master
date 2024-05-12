#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:15:47 2020

@author: ubuntu
"""
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import numpy as np



def MAPE(y_true, y_pred): 
  y_true, y_pred = np.array(y_true), np.array(y_pred)
  return np.mean(np.abs((y_true - y_pred) / np.maximum(np.ones(len(y_true)), np.abs(y_true))))*100


size = 0.1
l_rate = 1.2
n_est = 1000
max_d = 3

#################################################################################################################
#                                     TRADITIONAL                                                               #
#################################################################################################################
inputs = ["pi1","pi2","pi3","pi4","v_f"]
outputs = "v_f"
###############################################################
#                         1 pendule                           #
###############################################################                                       
#Load data set
data = pd.read_excel('1pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_1 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_1.fit(X_train_1,y_train_1)

###############################################################
#                         2 pendules                          #
###############################################################                                       
#Load data set
data = pd.read_excel('2pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_2 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_2.fit(X_train_2,y_train_2)

###############################################################
#                        4 pendules                           #
###############################################################                                       
#Load data set
data = pd.read_excel('4pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_4 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_4.fit(X_train_4,y_train_4)


###############################################################
#                      16 pendules                            #
###############################################################                                       
#Load data set
data = pd.read_excel('16pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_16 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_16, X_test_16, y_train_16, y_test_16 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_16.fit(X_train_16,y_train_16)



###############################################################
#                     256 pendules                            #
###############################################################                                       
#Load data set
data = pd.read_excel('256pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_256 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_256, X_test_256, y_train_256, y_test_256 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_256.fit(X_train_256,y_train_256)

###############################################################
#                   1000 pendules                             #
###############################################################                                       
#Load data set
data = pd.read_excel('1000pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_1000 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_1000, X_test_1000, y_train_1000, y_test_1000 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_1000.fit(X_train_1000,y_train_1000)

###############################################################
#                   10000 pendules                            #
###############################################################                                       
#Load data set
data = pd.read_excel('100kpendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_10000 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_10000, X_test_10000, y_train_10000, y_test_10000 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_10000.fit(X_train_10000,y_train_10000)

###############################################################
#                   unknown pendule                           #
###############################################################                                       
#Load data set
data = pd.read_excel('unknown_pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_U = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_U, X_test_U, y_train_U, y_test_U = train_test_split(X, y, test_size=0.9, random_state=42)
 
#Fit models
model_U.fit(X_train_U,y_train_U)

####################################################################################################################
#                                                  TEST                                                            #
####################################################################################################################

# Data U --> Model 1
predU_1 = model_1.predict(X_test_U)
MAPE_U_1 = MAPE(y_test_U, predU_1)

# Data U --> Model 2
predU_2 = model_2.predict(X_test_U)
MAPE_U_2 = MAPE(y_test_U, predU_2)

# Data U --> Model 4
predU_4 = model_4.predict(X_test_U)
MAPE_U_4 = MAPE(y_test_U, predU_4)

# Data U --> Model 16
predU_16 = model_16.predict(X_test_U)
MAPE_U_16 = MAPE(y_test_U, predU_16)

# Data U --> Model 256
predU_256 = model_256.predict(X_test_U)
MAPE_U_256 = MAPE(y_test_U, predU_256)

# Data U --> Model 1000
predU_1000 = model_1000.predict(X_test_U)
MAPE_U_1000 = MAPE(y_test_U, predU_1000)

# Data U --> Model 10000
predU_10000 = model_10000.predict(X_test_U)
MAPE_U_10000 = MAPE(y_test_U, predU_10000)

print(" ")
print("#################################################################################### ")
print("                                    TABLE PHYSICAL                             ")
print("#################################################################################### ")
print(" ")      
data = [[MAPE_U_1, MAPE_U_2, MAPE_U_4,MAPE_U_16,MAPE_U_256,MAPE_U_1000,MAPE_U_10000]]
col=["Modèle 1 pendule", "Modèle 2 pendules", "Modèle 4 pendules","Modèle 16 pendules","Modèle 256 pendules","Modèle 1000 pendules","Modèle 10000 pendules"]
row=["Data pendule inconnu"]
print(pd.DataFrame(data, row, col))
print(" ")





#################################################################################################################
#                                                BUCKINGHAM BASED                                               #
#################################################################################################################
inputs = ["m","l","g","v_i","h_i","v_f","pi1","pi4"]
outputs = "pi1"
###############################################################
#                         1 pendule                           #
###############################################################                                       
#Load data set
data = pd.read_excel('1pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_1 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_1.fit(X_train_1,y_train_1)

###############################################################
#                         2 pendules                          #
###############################################################                                       
#Load data set
data = pd.read_excel('2pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_2 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_2.fit(X_train_2,y_train_2)

###############################################################
#                        4 pendules                           #
###############################################################                                       
#Load data set
data = pd.read_excel('4pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_4 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_4.fit(X_train_4,y_train_4)


###############################################################
#                      16 pendules                            #
###############################################################                                       
#Load data set
data = pd.read_excel('16pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_16 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_16, X_test_16, y_train_16, y_test_16 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_16.fit(X_train_16,y_train_16)



###############################################################
#                     256 pendules                            #
###############################################################                                       
#Load data set
data = pd.read_excel('256pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_256 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_256, X_test_256, y_train_256, y_test_256 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_256.fit(X_train_256,y_train_256)

###############################################################
#                   1000 pendules                             #
###############################################################                                       
#Load data set
data = pd.read_excel('1000pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_1000 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_1000, X_test_1000, y_train_1000, y_test_1000 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_1000.fit(X_train_1000,y_train_1000)

###############################################################
#                   10000 pendules                            #
###############################################################                                       
#Load data set
data = pd.read_excel('100kpendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_10000 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_10000, X_test_10000, y_train_10000, y_test_10000 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_10000.fit(X_train_10000,y_train_10000)

###############################################################
#                   unknown pendule                           #
###############################################################                                       
#Load data set
data = pd.read_excel('unknown_pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_U = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_U, X_test_U, y_train_U, y_test_U = train_test_split(X, y, test_size=0.9, random_state=42)
 
#Fit models
model_U.fit(X_train_U,y_train_U)

####################################################################################################################
#                                                  TEST                                                            #
####################################################################################################################

# Data U --> Model 1
predU_1 = model_1.predict(X_test_U)
MAPE_U_1 = MAPE(y_test_U, predU_1)

# Data U --> Model 2
predU_2 = model_2.predict(X_test_U)
MAPE_U_2 = MAPE(y_test_U, predU_2)

# Data U --> Model 4
predU_4 = model_4.predict(X_test_U)
MAPE_U_4 = MAPE(y_test_U, predU_4)

# Data U --> Model 16
predU_16 = model_16.predict(X_test_U)
MAPE_U_16 = MAPE(y_test_U, predU_16)

# Data U --> Model 256
predU_256 = model_256.predict(X_test_U)
MAPE_U_256 = MAPE(y_test_U, predU_256)

# Data U --> Model 1000
predU_1000 = model_1000.predict(X_test_U)
MAPE_U_1000 = MAPE(y_test_U, predU_1000)

# Data U --> Model 10000
predU_10000 = model_10000.predict(X_test_U)
MAPE_U_10000 = MAPE(y_test_U, predU_10000)

print(" ")
print("#################################################################################### ")
print("                                    BUCKINGHAM                                       ")
print("#################################################################################### ")
print(" ")      
data = [[MAPE_U_1, MAPE_U_2, MAPE_U_4,MAPE_U_16,MAPE_U_256,MAPE_U_1000,MAPE_U_10000]]
col=["Modèle 1 pendule", "Modèle 2 pendules", "Modèle 4 pendules","Modèle 16 pendules","Modèle 256 pendules","Modèle 1000 pendules","Modèle 10000 pendules"]
row=["Data pendule inconnu"]
print(pd.DataFrame(data, row, col))
print(" ")


#################################################################################################################
#                                                AUGMENTED BUCKINGHAM BASED                                     #
#################################################################################################################
inputs = ["m","l","g","v_i","h_i","v_f","pi1"]
outputs = "pi1"
###############################################################
#                         1 pendule                           #
###############################################################                                       
#Load data set
data = pd.read_excel('1pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_1 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_1.fit(X_train_1,y_train_1)

###############################################################
#                         2 pendules                          #
###############################################################                                       
#Load data set
data = pd.read_excel('2pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_2 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_2.fit(X_train_2,y_train_2)

###############################################################
#                        4 pendules                           #
###############################################################                                       
#Load data set
data = pd.read_excel('4pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_4 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_4.fit(X_train_4,y_train_4)


###############################################################
#                      16 pendules                            #
###############################################################                                       
#Load data set
data = pd.read_excel('16pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_16 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_16, X_test_16, y_train_16, y_test_16 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_16.fit(X_train_16,y_train_16)



###############################################################
#                     256 pendules                            #
###############################################################                                       
#Load data set
data = pd.read_excel('256pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_256 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_256, X_test_256, y_train_256, y_test_256 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_256.fit(X_train_256,y_train_256)

###############################################################
#                   1000 pendules                             #
###############################################################                                       
#Load data set
data = pd.read_excel('1000pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_1000 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_1000, X_test_1000, y_train_1000, y_test_1000 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_1000.fit(X_train_1000,y_train_1000)

###############################################################
#                   10000 pendules                            #
###############################################################                                       
#Load data set
data = pd.read_excel('100kpendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_10000 = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_10000, X_test_10000, y_train_10000, y_test_10000 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_10000.fit(X_train_10000,y_train_10000)

###############################################################
#                   unknown pendule                           #
###############################################################                                       
#Load data set
data = pd.read_excel('unknown_pendule.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(inputs,axis=1)
X = X.values

#Create outputs
y = data[outputs].values

#Create models
model_U = XGBRegressor(learning_rate=l_rate,n_estimators=n_est,max_depth=max_d)

#Split data
X_train_U, X_test_U, y_train_U, y_test_U = train_test_split(X, y, test_size=0.9, random_state=42)
 
#Fit models
model_U.fit(X_train_U,y_train_U)

####################################################################################################################
#                                                  TEST                                                            #
####################################################################################################################

# Data U --> Model 1
predU_1 = model_1.predict(X_test_U)
MAPE_U_1 = MAPE(y_test_U, predU_1)

# Data U --> Model 2
predU_2 = model_2.predict(X_test_U)
MAPE_U_2 = MAPE(y_test_U, predU_2)

# Data U --> Model 4
predU_4 = model_4.predict(X_test_U)
MAPE_U_4 = MAPE(y_test_U, predU_4)

# Data U --> Model 16
predU_16 = model_16.predict(X_test_U)
MAPE_U_16 = MAPE(y_test_U, predU_16)

# Data U --> Model 256
predU_256 = model_256.predict(X_test_U)
MAPE_U_256 = MAPE(y_test_U, predU_256)

# Data U --> Model 1000
predU_1000 = model_1000.predict(X_test_U)
MAPE_U_1000 = MAPE(y_test_U, predU_1000)

# Data U --> Model 10000
predU_10000 = model_10000.predict(X_test_U)
MAPE_U_10000 = MAPE(y_test_U, predU_10000)


print(" ")
print("#################################################################################### ")
print("                                AUGMENTED BUCKINGHAM                                  ")
print("#################################################################################### ")
print(" ")      
data = [[MAPE_U_1, MAPE_U_2, MAPE_U_4,MAPE_U_16,MAPE_U_256,MAPE_U_1000,MAPE_U_10000]]
col=["Modèle 1 pendule", "Modèle 2 pendules", "Modèle 4 pendules","Modèle 16 pendules","Modèle 256 pendules","Modèle 1000 pendules","Modèle 10000 pendules"]
row=["Data pendule inconnu"]
print(pd.DataFrame(data, row, col))
print(" ")












