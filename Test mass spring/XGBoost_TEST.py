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

MAPE_X = []
MAPE_Y = []
MAPE_T = []
MAPE_P1 = []
MAPE_P2 = []
MAPE_P3 = []
MAPE_P1p = []
MAPE_P2p = []
MAPE_P3p = []
T_SIZE  = []
size = 0.2

#################################################################################################################
#                                     TRADITIONAL                                                               #
#################################################################################################################

###############################################################
#                           BLOC 1                            #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block1.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["x_f"].values

#Create models
model_1 = XGBRegressor()

#Split data
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_1.fit(X_train_1,y_train_1)

###############################################################
#                           BLOC 2                            #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block2.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["x_f"].values

#Create models
model_2 = XGBRegressor()

#Split data
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_2.fit(X_train_2,y_train_2)

###############################################################
#                           BLOC 3                            #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block3.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["x_f"].values

#Create models
model_3 = XGBRegressor()

#Split data
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_3.fit(X_train_3,y_train_3)


###############################################################
#                          TOUTES                             #
###############################################################                                       
#Load data set
data = pd.read_excel('full_blocks.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["x_f"].values

#Create models
model_F = XGBRegressor()

#Split data
X_train_F, X_test_F, y_train_F, y_test_F = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_F.fit(X_train_F,y_train_F)



###############################################################
#                     HUGE BLOCK                              #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block_big.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["x_f"].values

#Create models
model_B = XGBRegressor()

#Split data
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_B.fit(X_train_B,y_train_B)

###############################################################
#                     SMALL BLOCK                              #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block_mini.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["x_f"].values

#Create models
model_M = XGBRegressor()

#Split data
X_train_M, X_test_M, y_train_M, y_test_M = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_M.fit(X_train_M,y_train_M)

####################################################################################################################
#                                                  TEST                                                            #
####################################################################################################################

# Data 1 --> Model 1
pred1_1 = model_1.predict(X_test_1)
MAPE_1_1 = MAPE(y_test_1, pred1_1)

# Data 1 --> Model 2
pred1_2 = model_2.predict(X_test_1)
MAPE_1_2 = MAPE(y_test_1, pred1_2)

# Data 1 --> Model 3
pred1_3 = model_3.predict(X_test_1)
MAPE_1_3 = MAPE(y_test_1, pred1_3)

# Data 1 --> Model F
pred1_F = model_F.predict(X_test_1)
MAPE_1_F = MAPE(y_test_1, pred1_F)

# Data 2 --> Model 1
pred2_1 = model_1.predict(X_test_2)
MAPE_2_1 = MAPE(y_test_2, pred2_1)

# Data 2 --> Model 2
pred2_2 = model_2.predict(X_test_2)
MAPE_2_2 = MAPE(y_test_2, pred2_2)

# Data 2 --> Model 3
pred2_3 = model_3.predict(X_test_2)
MAPE_2_3 = MAPE(y_test_2, pred2_3)

# Data 2 --> Model F
pred2_F = model_F.predict(X_test_2)
MAPE_2_F = MAPE(y_test_2, pred2_F)

# Data 3 --> Model 1
pred3_1 = model_1.predict(X_test_3)
MAPE_3_1 = MAPE(y_test_3, pred3_1)

# Data 3 --> Model 2
pred3_2 = model_2.predict(X_test_3)
MAPE_3_2 = MAPE(y_test_3, pred3_2)

# Data 3 --> Model 3
pred3_3 = model_3.predict(X_test_3)
MAPE_3_3 = MAPE(y_test_3, pred3_3)

# Data 3 --> Model F
pred3_F = model_F.predict(X_test_3)
MAPE_3_F = MAPE(y_test_3, pred3_F)

# Data big--> Model F
predB_F = model_F.predict(X_test_B)
MAPE_B_F = MAPE(y_test_B, predB_F)

# Data mini--> Model F
predM_F = model_F.predict(X_test_M)
MAPE_M_F = MAPE(y_test_M, predM_F)


print(" ")
print("#################################################################################### ")
print("                                    TABLE PHYSICAL                             ")
print("#################################################################################### ")
print(" ")      
data = [[MAPE_1_1, MAPE_2_1, MAPE_3_1,0,0],
[MAPE_1_2, MAPE_2_2, MAPE_3_2,0,0],
[MAPE_1_3, MAPE_2_3 , MAPE_3_3,0,0],
[MAPE_1_F, MAPE_2_F , MAPE_3_F,MAPE_B_F,MAPE_M_F]]
col=["Data block 1", "Data block 2", "Data block 3","Data huge block","Data mini block"]
row=["Model block 1", "Model block 2", "Model block 3","Model FULL"]
print(pd.DataFrame(data, row, col))
print(" ")
print("Moyenne self-prediction (MAPE): "+str((MAPE_1_1+MAPE_2_2+MAPE_3_3)/3.0))
print("Moyenne cross-prediction (MAPE): "+str((MAPE_1_2+MAPE_1_3+MAPE_2_1+MAPE_2_3+MAPE_3_1+MAPE_3_2)/6.0))
print("Moyenne prediction known system in full model (MAPE): "+str((MAPE_1_F+MAPE_2_F+MAPE_3_F)/3.0))
print("Moyenne prediction unknown system in full model (MAPE): "+str((MAPE_B_F+MAPE_M_F)/2.0))
print(" ")




#################################################################################################################
#                                                BUCKINGHAM BASED                                               #
#################################################################################################################

###############################################################
#                           BLOC 1                            #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block1.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_1 = XGBRegressor()

#Split data
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_1.fit(X_train_1,y_train_1)

###############################################################
#                           BLOC 2                            #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block2.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_2 = XGBRegressor()

#Split data
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_2.fit(X_train_2,y_train_2)

###############################################################
#                           BLOC 3                            #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block3.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_3 = XGBRegressor()

#Split data
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_3.fit(X_train_3,y_train_3)


###############################################################
#                          TOUTES                             #
###############################################################                                       
#Load data set
data = pd.read_excel('full_blocks.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_F = XGBRegressor()

#Split data
X_train_F, X_test_F, y_train_F, y_test_F = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_F.fit(X_train_F,y_train_F)



###############################################################
#                     HUGE BLOCK                              #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block_big.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_B = XGBRegressor()

#Split data
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_B.fit(X_train_B,y_train_B)

###############################################################
#                     SMALL BLOCK                              #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block_mini.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","pi6","pi7","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_M = XGBRegressor()

#Split data
X_train_M, X_test_M, y_train_M, y_test_M = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_M.fit(X_train_M,y_train_M)

####################################################################################################################
#                                                  TEST                                                            #
####################################################################################################################

# Data 1 --> Model 1
pred1_1 = model_1.predict(X_test_1)
MAPE_1_1 = MAPE(y_test_1, pred1_1)

# Data 1 --> Model 2
pred1_2 = model_2.predict(X_test_1)
MAPE_1_2 = MAPE(y_test_1, pred1_2)

# Data 1 --> Model 3
pred1_3 = model_3.predict(X_test_1)
MAPE_1_3 = MAPE(y_test_1, pred1_3)

# Data 1 --> Model F
pred1_F = model_F.predict(X_test_1)
MAPE_1_F = MAPE(y_test_1, pred1_F)

# Data 2 --> Model 1
pred2_1 = model_1.predict(X_test_2)
MAPE_2_1 = MAPE(y_test_2, pred2_1)

# Data 2 --> Model 2
pred2_2 = model_2.predict(X_test_2)
MAPE_2_2 = MAPE(y_test_2, pred2_2)

# Data 2 --> Model 3
pred2_3 = model_3.predict(X_test_2)
MAPE_2_3 = MAPE(y_test_2, pred2_3)

# Data 2 --> Model F
pred2_F = model_F.predict(X_test_2)
MAPE_2_F = MAPE(y_test_2, pred2_F)

# Data 3 --> Model 1
pred3_1 = model_1.predict(X_test_3)
MAPE_3_1 = MAPE(y_test_3, pred3_1)

# Data 3 --> Model 2
pred3_2 = model_2.predict(X_test_3)
MAPE_3_2 = MAPE(y_test_3, pred3_2)

# Data 3 --> Model 3
pred3_3 = model_3.predict(X_test_3)
MAPE_3_3 = MAPE(y_test_3, pred3_3)

# Data 3 --> Model F
pred3_F = model_F.predict(X_test_3)
MAPE_3_F = MAPE(y_test_3, pred3_F)

# Data big--> Model F
predB_F = model_F.predict(X_test_B)
MAPE_B_F = MAPE(y_test_B, predB_F)

# Data mini--> Model F
predM_F = model_F.predict(X_test_M)
MAPE_M_F = MAPE(y_test_M, predM_F)


print(" ")
print("#################################################################################### ")
print("                                    BUCKINGHAM                                       ")
print("#################################################################################### ")
print(" ")      
data = [[MAPE_1_1, MAPE_2_1, MAPE_3_1,0,0],
[MAPE_1_2, MAPE_2_2, MAPE_3_2,0,0],
[MAPE_1_3, MAPE_2_3 , MAPE_3_3,0,0],
[MAPE_1_F, MAPE_2_F , MAPE_3_F,MAPE_B_F,MAPE_M_F]]
col=["Data block 1", "Data block 2", "Data block 3","Data huge block","Data mini block"]
row=["Model block 1", "Model block 2", "Model block 3","Model FULL"]
print(pd.DataFrame(data, row, col))
print(" ")
print("Moyenne self-prediction (MAPE): "+str((MAPE_1_1+MAPE_2_2+MAPE_3_3)/3.0))
print("Moyenne cross-prediction (MAPE): "+str((MAPE_1_2+MAPE_1_3+MAPE_2_1+MAPE_2_3+MAPE_3_1+MAPE_3_2)/6.0))
print("Moyenne prediction known system in full model (MAPE): "+str((MAPE_1_F+MAPE_2_F+MAPE_3_F)/3.0))
print("Moyenne prediction unknown system in full model (MAPE): "+str((MAPE_B_F+MAPE_M_F)/2.0))
print(" ")


#################################################################################################################
#                                                AUGMENTED BUCKINGHAM BASED                                     #
#################################################################################################################

###############################################################
#                           BLOC 1                            #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block1.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_1 = XGBRegressor()

#Split data
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_1.fit(X_train_1,y_train_1)

###############################################################
#                           BLOC 2                            #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block2.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_2 = XGBRegressor()

#Split data
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_2.fit(X_train_2,y_train_2)

###############################################################
#                           BLOC 3                            #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block3.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_3 = XGBRegressor()

#Split data
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_3.fit(X_train_3,y_train_3)


###############################################################
#                          TOUTES                             #
###############################################################                                       
#Load data set
data = pd.read_excel('full_blocks.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_F = XGBRegressor()

#Split data
X_train_F, X_test_F, y_train_F, y_test_F = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_F.fit(X_train_F,y_train_F)



###############################################################
#                     HUGE BLOCK                              #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block_big.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_B = XGBRegressor()

#Split data
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_B.fit(X_train_B,y_train_B)


###############################################################
#                     SMALL BLOCK                              #
###############################################################                                       
#Load data set
data = pd.read_excel('sim_block_mini.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Create inputs
X = data.drop(["m","k","theta","mu","g","v_i","x_i","pi5","x_f"],axis=1)
X = X.values

#Create outputs
y = data["pi5"].values

#Create models
model_M = XGBRegressor()

#Split data
X_train_M, X_test_M, y_train_M, y_test_M = train_test_split(X, y, test_size=size, random_state=42)
 
#Fit models
model_M.fit(X_train_M,y_train_M)

####################################################################################################################
#                                                  TEST                                                            #
####################################################################################################################

# Data 1 --> Model 1
pred1_1 = model_1.predict(X_test_1)
MAPE_1_1 = MAPE(y_test_1, pred1_1)

# Data 1 --> Model 2
pred1_2 = model_2.predict(X_test_1)
MAPE_1_2 = MAPE(y_test_1, pred1_2)

# Data 1 --> Model 3
pred1_3 = model_3.predict(X_test_1)
MAPE_1_3 = MAPE(y_test_1, pred1_3)

# Data 1 --> Model F
pred1_F = model_F.predict(X_test_1)
MAPE_1_F = MAPE(y_test_1, pred1_F)

# Data 2 --> Model 1
pred2_1 = model_1.predict(X_test_2)
MAPE_2_1 = MAPE(y_test_2, pred2_1)

# Data 2 --> Model 2
pred2_2 = model_2.predict(X_test_2)
MAPE_2_2 = MAPE(y_test_2, pred2_2)

# Data 2 --> Model 3
pred2_3 = model_3.predict(X_test_2)
MAPE_2_3 = MAPE(y_test_2, pred2_3)

# Data 2 --> Model F
pred2_F = model_F.predict(X_test_2)
MAPE_2_F = MAPE(y_test_2, pred2_F)

# Data 3 --> Model 1
pred3_1 = model_1.predict(X_test_3)
MAPE_3_1 = MAPE(y_test_3, pred3_1)

# Data 3 --> Model 2
pred3_2 = model_2.predict(X_test_3)
MAPE_3_2 = MAPE(y_test_3, pred3_2)

# Data 3 --> Model 3
pred3_3 = model_3.predict(X_test_3)
MAPE_3_3 = MAPE(y_test_3, pred3_3)

# Data 3 --> Model F
pred3_F = model_F.predict(X_test_3)
MAPE_3_F = MAPE(y_test_3, pred3_F)

# Data big--> Model F
predB_F = model_F.predict(X_test_B)
MAPE_B_F = MAPE(y_test_B, predB_F)

# Data mini--> Model F
predM_F = model_F.predict(X_test_M)
MAPE_M_F = MAPE(y_test_M, predM_F)


print(" ")
print("#################################################################################### ")
print("                                AUGMENTED BUCKINGHAM                                  ")
print("#################################################################################### ")
print(" ")      
data = [[MAPE_1_1, MAPE_2_1, MAPE_3_1,0,0],
[MAPE_1_2, MAPE_2_2, MAPE_3_2,0,0],
[MAPE_1_3, MAPE_2_3 , MAPE_3_3,0,0],
[MAPE_1_F, MAPE_2_F , MAPE_3_F,MAPE_B_F,MAPE_M_F]]
col=["Data block 1", "Data block 2", "Data block 3","Data huge block","Data mini block"]
row=["Model block 1", "Model block 2", "Model block 3","Model FULL"]
print(pd.DataFrame(data, row, col))
print(" ")
print("Moyenne self-prediction (MAPE): "+str((MAPE_1_1+MAPE_2_2+MAPE_3_3)/3.0))
print("Moyenne cross-prediction (MAPE): "+str((MAPE_1_2+MAPE_1_3+MAPE_2_1+MAPE_2_3+MAPE_3_1+MAPE_3_2)/6.0))
print("Moyenne prediction known system in full model (MAPE): "+str((MAPE_1_F+MAPE_2_F+MAPE_3_F)/3.0))
print("Moyenne prediction unknown system in full model (MAPE): "+str((MAPE_B_F+MAPE_M_F)/2.0))
print(" ")












