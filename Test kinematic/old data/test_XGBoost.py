#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:15:47 2020

@author: ubuntu
"""
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
#import numpy as np
from sklearn.metrics import mean_absolute_error as MAPE


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
print("")
print("-------------------------------------------------------------------------------")
print("")
###############################################################
#                     RACECAR  (Vehicle 1)                    #
###############################################################                                       
#Load data set
data = pd.read_excel('racecar.xlsx',index_col=False,engine='openpyxl')

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Only use physical input (v_i, length, a and delta)
X_p1 = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","X","Y","theta"],axis=1)
X_p1 = X_p1.values

#Only use y as output
y_X_p1 = data["X"].values
y_Y_p1 = data["Y"].values
y_T_p1 = data["theta"].values

#Create models
model_X_p1 = XGBRegressor()
model_Y_p1 = XGBRegressor()
model_T_p1 = XGBRegressor()

#Split data
X_X_train1, X_X_test1, y_X_train1, y_X_test1 = train_test_split(X_p1, y_X_p1, test_size=size, random_state=42)
X_Y_train1, X_Y_test1, y_Y_train1, y_Y_test1 = train_test_split(X_p1, y_Y_p1, test_size=size, random_state=42)    
X_T_train1, X_T_test1, y_T_train1, y_T_test1 = train_test_split(X_p1, y_T_p1, test_size=size, random_state=42) 

#Fit models
model_X_p1.fit(X_X_train1,y_X_train1)
model_Y_p1.fit(X_Y_train1,y_Y_train1)
model_T_p1.fit(X_T_train1,y_T_train1)

print(model_X_p1.predict(X_X_test1))
print(model_Y_p1.predict(X_Y_test1))
print(model_T_p1.predict(X_T_test1))

print(y_X_test1)
print(y_Y_test1)
print(y_T_test1)



###############################################################
#                      LIMO  (Vehicle 2)                      #
###############################################################                                       
#Load data set
data = pd.read_excel('limo.xlsx',index_col=False,engine='openpyxl')

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Only use physical input (v_i, length, a and delta)
X_p2 = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","X","Y","theta"],axis=1)
X_p2 = X_p2.values

#Only use y as output
y_X_p2 = data["X"].values
y_Y_p2 = data["Y"].values
y_T_p2 = data["theta"].values

#Create models
model_X_p2 = XGBRegressor()
model_Y_p2 = XGBRegressor()
model_T_p2 = XGBRegressor()

#Split data
X_X_train2, X_X_test2, y_X_train2, y_X_test2 = train_test_split(X_p2, y_X_p2, test_size=size, random_state=42)
X_Y_train2, X_Y_test2, y_Y_train2, y_Y_test2 = train_test_split(X_p2, y_Y_p2, test_size=size, random_state=42)    
X_T_train2, X_T_test2, y_T_train2, y_T_test2 = train_test_split(X_p2, y_T_p2, test_size=size, random_state=42)        

#Fit models
model_X_p2.fit(X_X_train2,y_X_train2)
model_Y_p2.fit(X_Y_train2,y_Y_train2)
model_T_p2.fit(X_T_train2,y_T_train2)

###############################################################
#                      X-MAXX  (Vehicle 3)                    #
###############################################################                                       
#Load data set
data = pd.read_excel('Xmaxx.xlsx',index_col=False,engine='openpyxl')

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Only use physical input (v_i, length, a and delta)
X_p3 = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","X","Y","theta"],axis=1)
X_p3 = X_p3.values

#Only use y as output
y_X_p3 = data["X"].values
y_Y_p3 = data["Y"].values
y_T_p3 = data["theta"].values

#Create models
model_X_p3 = XGBRegressor()
model_Y_p3 = XGBRegressor()
model_T_p3 = XGBRegressor()

#Split data
X_X_train3, X_X_test3, y_X_train3, y_X_test3 = train_test_split(X_p3, y_X_p3, test_size=size, random_state=42)
X_Y_train3, X_Y_test3, y_Y_train3, y_Y_test3 = train_test_split(X_p3, y_Y_p3, test_size=size, random_state=42)    
X_T_train3, X_T_test3, y_T_train3, y_T_test3 = train_test_split(X_p3, y_T_p3, test_size=size, random_state=42)    

    
    

#Fit models
model_X_p3.fit(X_X_train3,y_X_train3)
model_Y_p3.fit(X_Y_train3,y_Y_train3)
model_T_p3.fit(X_T_train3,y_T_train3)

###############################################################
#                          TOUTES                             #
###############################################################                                       
#Load data set
data = pd.read_excel('merged.xlsx',index_col=False,engine='openpyxl')

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Only use physical input (v_i, length, a and delta)
X_pF = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","X","Y","theta"],axis=1)
X_pF = X_pF.values

#Only use y as output
y_X_pF = data["X"].values
y_Y_pF = data["Y"].values
y_T_pF = data["theta"].values

#Create models
model_X_pF = XGBRegressor()
model_Y_pF = XGBRegressor()
model_T_pF = XGBRegressor()

#Split data
X_X_trainF, X_X_testF, y_X_trainF, y_X_testF = train_test_split(X_pF, y_X_pF, test_size=size, random_state=42)
X_Y_trainF, X_Y_testF, y_Y_trainF, y_Y_testF = train_test_split(X_pF, y_Y_pF, test_size=size, random_state=42)    
X_T_trainF, X_T_testF, y_T_trainF, y_T_testF = train_test_split(X_pF, y_T_pF, test_size=size, random_state=42)        

#Fit models
model_X_pF.fit(X_X_trainF,y_X_trainF)
model_Y_pF.fit(X_Y_trainF,y_Y_trainF)
model_T_pF.fit(X_T_trainF,y_T_trainF)    

###############################################################
#                     CIVIC FULL SIZE                         #
###############################################################                                       
#Load data set
data = pd.read_excel('civic.xlsx',index_col=False,engine='openpyxl')

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Only use physical input (v_i, length, a and delta)
X_pC = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","X","Y","theta"],axis=1)
X_pC = X_pC.values

#Only use y as output
y_X_pC = data["X"].values
y_Y_pC = data["Y"].values
y_T_pC = data["theta"].values

#Create models
model_X_pC = XGBRegressor()
model_Y_pC = XGBRegressor()
model_T_pC = XGBRegressor()

#Split data
X_X_trainC, X_X_testC, y_X_trainC, y_X_testC = train_test_split(X_pC, y_X_pC, test_size=size, random_state=42)
X_Y_trainC, X_Y_testC, y_Y_trainC, y_Y_testC = train_test_split(X_pC, y_Y_pC, test_size=size, random_state=42)    
X_T_trainC, X_T_testC, y_T_trainC, y_T_testC = train_test_split(X_pC, y_T_pC, test_size=size, random_state=42)  



####################################################################################################################
#                                                TRADITIONAL   TEST                                                #
####################################################################################################################

print("# Data 1 --> Model 1")
y_X_pred1_1 = model_X_p1.predict(X_X_test1)
y_Y_pred1_1 = model_Y_p1.predict(X_Y_test1)
y_T_pred1_1 = model_T_p1.predict(X_T_test1)
print(MAPE(y_X_test1, y_X_pred1_1))
print(MAPE(y_Y_test1, y_Y_pred1_1))
print(MAPE(y_T_test1, y_T_pred1_1))


print("# Data 1 --> Model 2")
y_X_pred1_2 = model_X_p2.predict(X_X_test1)
y_Y_pred1_2 = model_Y_p2.predict(X_Y_test1)
y_T_pred1_2 = model_T_p2.predict(X_T_test1)
print(MAPE(y_X_test1, y_X_pred1_2))
print(MAPE(y_Y_test1, y_Y_pred1_2))
print(MAPE(y_T_test1, y_T_pred1_2))


print("# Data 1 --> Model 3")
y_X_pred1_3 = model_X_p3.predict(X_X_test1)
y_Y_pred1_3 = model_Y_p3.predict(X_Y_test1)
y_T_pred1_3 = model_T_p3.predict(X_T_test1)
print(MAPE(y_X_test1, y_X_pred1_3))
print(MAPE(y_Y_test1, y_Y_pred1_3))
print(MAPE(y_T_test1, y_T_pred1_3))

print("# Data 1 --> Model F")
y_X_pred1_F = model_X_pF.predict(X_X_test1)
y_Y_pred1_F = model_Y_pF.predict(X_Y_test1)
y_T_pred1_F = model_T_pF.predict(X_T_test1)
print(MAPE(y_X_test1, y_X_pred1_F))
print(MAPE(y_Y_test1, y_Y_pred1_F))
print(MAPE(y_T_test1, y_T_pred1_F))
print("# Data 2 --> Model 1")
y_X_pred2_1 = model_X_p1.predict(X_X_test2)
y_Y_pred2_1 = model_Y_p1.predict(X_Y_test2)
y_T_pred2_1 = model_T_p1.predict(X_T_test2)
print(MAPE(y_X_test2, y_X_pred2_1))
print(MAPE(y_Y_test2, y_Y_pred2_1))
print(MAPE(y_T_test2, y_T_pred2_1))

print("# Data 2 --> Model 2")
y_X_pred2_2 = model_X_p2.predict(X_X_test2)
y_Y_pred2_2 = model_Y_p2.predict(X_Y_test2)
y_T_pred2_2 = model_T_p2.predict(X_T_test2)
print(MAPE(y_X_test2, y_X_pred2_2))
print(MAPE(y_Y_test2, y_Y_pred2_2))
print(MAPE(y_T_test2, y_T_pred2_2))

print("# Data 2 --> Model 3")
y_X_pred2_3 = model_X_p3.predict(X_X_test2)
y_Y_pred2_3 = model_Y_p3.predict(X_Y_test2)
y_T_pred2_3 = model_T_p3.predict(X_T_test2)
print(MAPE(y_X_test2, y_X_pred2_3))
print(MAPE(y_Y_test2, y_Y_pred2_3))
print(MAPE(y_T_test2, y_T_pred2_3))

print("# Data 2 --> Model F")
y_X_pred2_F = model_X_pF.predict(X_X_test2)
y_Y_pred2_F = model_Y_pF.predict(X_Y_test2)
y_T_pred2_F = model_T_pF.predict(X_T_test2)
print(MAPE(y_X_test2, y_X_pred2_F))
print(MAPE(y_Y_test2, y_Y_pred2_F))
print(MAPE(y_T_test2, y_T_pred2_F))



print("# Data 3 --> Model 1")
y_X_pred3_1 = model_X_p1.predict(X_X_test3)
y_Y_pred3_1 = model_Y_p1.predict(X_Y_test3)
y_T_pred3_1 = model_T_p1.predict(X_T_test3)
print(MAPE(y_X_test3, y_X_pred3_1))
print(MAPE(y_Y_test3, y_Y_pred3_1))
print(MAPE(y_T_test3, y_T_pred3_1))

print("# Data 3 --> Model 2")
y_X_pred3_2 = model_X_p2.predict(X_X_test3)
y_Y_pred3_2 = model_Y_p2.predict(X_Y_test3)
y_T_pred3_2 = model_T_p2.predict(X_T_test3)
print(MAPE(y_X_test3, y_X_pred3_2))
print(MAPE(y_Y_test3, y_Y_pred3_2))
print(MAPE(y_T_test3, y_T_pred3_2))

print("# Data 3 --> Model 3")
y_X_pred3_3 = model_X_p3.predict(X_X_test3)
y_Y_pred3_3 = model_Y_p3.predict(X_Y_test3)
y_T_pred3_3 = model_T_p3.predict(X_T_test3)
print(MAPE(y_X_test3, y_X_pred3_3))
print(MAPE(y_Y_test3, y_Y_pred3_3))
print(MAPE(y_T_test3, y_T_pred3_3))

print("# Data 3 --> Model F")
y_X_pred3_F = model_X_pF.predict(X_X_test3)
y_Y_pred3_F = model_Y_pF.predict(X_Y_test3)
y_T_pred3_F = model_T_pF.predict(X_T_test3)
print(MAPE(y_X_test3, y_X_pred3_F))
print(MAPE(y_Y_test3, y_Y_pred3_F))
print(MAPE(y_T_test3, y_T_pred3_F))

print("# Data civic --> Model F")
y_X_predC_F = model_X_pF.predict(X_X_testC)
y_Y_predC_F = model_Y_pF.predict(X_Y_testC)
y_T_predC_F = model_T_pF.predict(X_T_testC)
print(MAPE(y_X_testC, y_X_predC_F))
print(MAPE(y_Y_testC, y_Y_predC_F))
print(MAPE(y_T_testC, y_T_predC_F))









