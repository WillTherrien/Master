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
#                     RACECAR  (Vehicle 1)                    #
###############################################################                                       
#Load data set
data = pd.read_excel('racecar_data.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Only use physical input (v_i, length, a and delta)
X_p1 = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","pi8","pi9","pi10","pi11","X","Y","theta"],axis=1)
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



###############################################################
#                      LIMO  (Vehicle 2)                      #
###############################################################                                       
#Load data set
data = pd.read_excel('limo_data.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Only use physical input (v_i, length, a and delta)
X_p2 = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","pi8","pi9","pi10","pi11","X","Y","theta"],axis=1)
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
data = pd.read_excel('Xmaxx_data.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

#Only use physical input (v_i, length, a and delta)
X_p3 = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","pi8","pi9","pi10","pi11","X","Y","theta"],axis=1)
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

####################################################################################################################
#                                                TRADITIONAL   TEST                                                #
####################################################################################################################

# Data 1 --> Model 1
y_X_pred1_1 = model_X_p1.predict(X_X_test1)
y_Y_pred1_1 = model_Y_p1.predict(X_Y_test1)
y_T_pred1_1 = model_T_p1.predict(X_T_test1)
MAPE_X1_1 = MAPE(y_X_test1, y_X_pred1_1)
MAPE_Y1_1 = MAPE(y_Y_test1, y_Y_pred1_1)
MAPE_T1_1 = MAPE(y_T_test1, y_T_pred1_1)
MAPE_1_1 = (MAPE_X1_1+MAPE_Y1_1+MAPE_T1_1)/3

# Data 1 --> Model 2
y_X_pred1_2 = model_X_p2.predict(X_X_test1)
y_Y_pred1_2 = model_Y_p2.predict(X_Y_test1)
y_T_pred1_2 = model_T_p2.predict(X_T_test1)
MAPE_X1_2 = MAPE(y_X_test1, y_X_pred1_2)
MAPE_Y1_2 = MAPE(y_Y_test1, y_Y_pred1_2)
MAPE_T1_2 = MAPE(y_T_test1, y_T_pred1_2)
MAPE_1_2 = (MAPE_X1_2+MAPE_Y1_2+MAPE_T1_2)/3

# Data 1 --> Model 3
y_X_pred1_3 = model_X_p3.predict(X_X_test1)
y_Y_pred1_3 = model_Y_p3.predict(X_Y_test1)
y_T_pred1_3 = model_T_p3.predict(X_T_test1)
MAPE_X1_3 = MAPE(y_X_test1, y_X_pred1_3)
MAPE_Y1_3 = MAPE(y_Y_test1, y_Y_pred1_3)
MAPE_T1_3 = MAPE(y_T_test1, y_T_pred1_3)
MAPE_1_3 = (MAPE_X1_3+MAPE_Y1_3+MAPE_T1_3)/3

# Data 2 --> Model 1
y_X_pred2_1 = model_X_p1.predict(X_X_test2)
y_Y_pred2_1 = model_Y_p1.predict(X_Y_test2)
y_T_pred2_1 = model_T_p1.predict(X_T_test2)
MAPE_X2_1 = MAPE(y_X_test2, y_X_pred2_1)
MAPE_Y2_1 = MAPE(y_Y_test2, y_Y_pred2_1)
MAPE_T2_1 = MAPE(y_T_test2, y_T_pred2_1)
MAPE_2_1 = (MAPE_X2_1+MAPE_Y2_1+MAPE_T2_1)/3

# Data 2 --> Model 2
y_X_pred2_2 = model_X_p2.predict(X_X_test2)
y_Y_pred2_2 = model_Y_p2.predict(X_Y_test2)
y_T_pred2_2 = model_T_p2.predict(X_T_test2)
MAPE_X2_2 = MAPE(y_X_test2, y_X_pred2_2)
MAPE_Y2_2 = MAPE(y_Y_test2, y_Y_pred2_2)
MAPE_T2_2 = MAPE(y_T_test2, y_T_pred2_2)
MAPE_2_2 = (MAPE_X2_2+MAPE_Y2_2+MAPE_T2_2)/3

# Data 2 --> Model 3
y_X_pred2_3 = model_X_p3.predict(X_X_test2)
y_Y_pred2_3 = model_Y_p3.predict(X_Y_test2)
y_T_pred2_3 = model_T_p3.predict(X_T_test2)
MAPE_X2_3 = MAPE(y_X_test2, y_X_pred2_3)
MAPE_Y2_3 = MAPE(y_Y_test2, y_Y_pred2_3)
MAPE_T2_3 = MAPE(y_T_test2, y_T_pred2_3)
MAPE_2_3 = (MAPE_X2_3+MAPE_Y2_3+MAPE_T2_3)/3

# Data 3 --> Model 1
y_X_pred3_1 = model_X_p1.predict(X_X_test3)
y_Y_pred3_1 = model_Y_p1.predict(X_Y_test3)
y_T_pred3_1 = model_T_p1.predict(X_T_test3)
MAPE_X3_1 = MAPE(y_X_test3, y_X_pred3_1)
MAPE_Y3_1 = MAPE(y_Y_test3, y_Y_pred3_1)
MAPE_T3_1 = MAPE(y_T_test3, y_T_pred3_1)
MAPE_3_1 = (MAPE_X3_1+MAPE_Y3_1+MAPE_T3_1)/3

# Data 3 --> Model 2
y_X_pred3_2 = model_X_p2.predict(X_X_test3)
y_Y_pred3_2 = model_Y_p2.predict(X_Y_test3)
y_T_pred3_2 = model_T_p2.predict(X_T_test3)
MAPE_X3_2 = MAPE(y_X_test3, y_X_pred3_2)
MAPE_Y3_2 = MAPE(y_Y_test3, y_Y_pred3_2)
MAPE_T3_2 = MAPE(y_T_test3, y_T_pred3_2)
MAPE_3_2 = (MAPE_X3_2+MAPE_Y3_2+MAPE_T3_2)/3

# Data 3 --> Model 3
y_X_pred3_3 = model_X_p3.predict(X_X_test3)
y_Y_pred3_3 = model_Y_p3.predict(X_Y_test3)
y_T_pred3_3 = model_T_p3.predict(X_T_test3)
MAPE_X3_3 = MAPE(y_X_test3, y_X_pred3_3)
MAPE_Y3_3 = MAPE(y_Y_test3, y_Y_pred3_3)
MAPE_T3_3 = MAPE(y_T_test3, y_T_pred3_3)
MAPE_3_3 = (MAPE_X3_3+MAPE_Y3_3+MAPE_T3_3)/3


print("TABLE PHYSICAL")
data = [[MAPE_1_1, MAPE_2_1, MAPE_3_1],
[MAPE_1_2, MAPE_2_2, MAPE_3_2],
[MAPE_1_3, MAPE_2_3 , MAPE_3_3]]
col=["Data vehicle 1", "Data vehicle 2", "Data vehicle 3"]
row=["Model vehicle 1", "Model vehicle 2", "Model vehicle 3"]
print(pd.DataFrame(data, row, col))




#################################################################################################################
#                                                BUCKINGHAM BASED                                               #
#################################################################################################################

###############################################################
#                     RACECAR  (Vehicle 1)                    #
###############################################################                                       
#Load data set
data = pd.read_excel('racecar_data.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################
X_a1 = data.drop(["v_i","l","mu","a","delta","pi1","pi2","pi3","X","Y","theta","m","N_f","N_r","pi9","pi10","pi11"],axis=1)
X_a1 = X_a1.values



#Only use y as output
y_X_a1 = data["pi1"].values
y_Y_a1 = data["pi2"].values
y_T_a1 = data["pi3"].values


#Create models
model_X_a1 = XGBRegressor()
model_Y_a1 = XGBRegressor()
model_T_a1 = XGBRegressor()  

#Split data
X_X_train1, X_X_test1, y_X_train1, y_X_test1 = train_test_split(X_a1, y_X_a1, test_size=size, random_state=42)
X_Y_train1, X_Y_test1, y_Y_train1, y_Y_test1 = train_test_split(X_a1, y_Y_a1, test_size=size, random_state=42)    
X_T_train1, X_T_test1, y_T_train1, y_T_test1 = train_test_split(X_a1, y_T_a1, test_size=size, random_state=42)        

#Fit models
model_X_a1.fit(X_X_train1,y_X_train1)
model_Y_a1.fit(X_Y_train1,y_Y_train1)
model_T_a1.fit(X_T_train1,y_T_train1)



###############################################################
#                      LIMO  (Vehicle 2)                      #
###############################################################                                       
#Load data set
data = pd.read_excel('limo_data.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

X_a2 = data.drop(["v_i","l","mu","a","delta","pi1","pi2","pi3","X","Y","theta","m","N_f","N_r","pi9","pi10","pi11"],axis=1)
X_a2 = X_a2.values



#Only use y as output
y_X_a2 = data["pi1"].values
y_Y_a2 = data["pi2"].values
y_T_a2 = data["pi3"].values


#Create models
model_X_a2 = XGBRegressor()
model_Y_a2 = XGBRegressor()
model_T_a2 = XGBRegressor()  

#Split data
X_X_train2, X_X_test2, y_X_train2, y_X_test2 = train_test_split(X_a2, y_X_a2, test_size=size, random_state=42)
X_Y_train2, X_Y_test2, y_Y_train2, y_Y_test2 = train_test_split(X_a2, y_Y_a2, test_size=size, random_state=42)    
X_T_train2, X_T_test2, y_T_train2, y_T_test2 = train_test_split(X_a2, y_T_a2, test_size=size, random_state=42)        

#Fit models
model_X_a2.fit(X_X_train2,y_X_train2)
model_Y_a2.fit(X_Y_train2,y_Y_train2)
model_T_a2.fit(X_T_train2,y_T_train2)

###############################################################
#                      X-MAXX  (Vehicle 3)                    #
###############################################################                                       
#Load data set
data = pd.read_excel('Xmaxx_data.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

X_a3 = data.drop(["v_i","l","mu","a","delta","pi1","pi2","pi3","X","Y","theta","m","N_f","N_r","pi9","pi10","pi11"],axis=1)
X_a3 = X_a3.values



#Only use y as output
y_X_a3 = data["pi1"].values
y_Y_a3 = data["pi2"].values
y_T_a3 = data["pi3"].values


#Create models
model_X_a3 = XGBRegressor()
model_Y_a3 = XGBRegressor()
model_T_a3 = XGBRegressor()  

#Split data
X_X_train3, X_X_test3, y_X_train3, y_X_test3 = train_test_split(X_a3, y_X_a3, test_size=size, random_state=42)
X_Y_train3, X_Y_test3, y_Y_train3, y_Y_test3 = train_test_split(X_a3, y_Y_a3, test_size=size, random_state=42)    
X_T_train3, X_T_test3, y_T_train3, y_T_test3 = train_test_split(X_a3, y_T_a3, test_size=size, random_state=42)        

#Fit models
model_X_a3.fit(X_X_train3,y_X_train3)
model_Y_a3.fit(X_Y_train3,y_Y_train3)
model_T_a3.fit(X_T_train3,y_T_train3)

####################################################################################################################
#                                               BUCKINGHAM  TEST                                                   #
####################################################################################################################

# Data 1 --> Model 1
y_X_pred1_1 = model_X_a1.predict(X_X_test1)
y_Y_pred1_1 = model_Y_a1.predict(X_Y_test1)
y_T_pred1_1 = model_T_a1.predict(X_T_test1)
MAPE_X1_1 = MAPE(y_X_test1, y_X_pred1_1)
MAPE_Y1_1 = MAPE(y_Y_test1, y_Y_pred1_1)
MAPE_T1_1 = MAPE(y_T_test1, y_T_pred1_1)
MAPE_1_1 = (MAPE_X1_1+MAPE_Y1_1+MAPE_T1_1)/3

# Data 1 --> Model 2
y_X_pred1_2 = model_X_a2.predict(X_X_test1)
y_Y_pred1_2 = model_Y_a2.predict(X_Y_test1)
y_T_pred1_2 = model_T_a2.predict(X_T_test1)
MAPE_X1_2 = MAPE(y_X_test1, y_X_pred1_2)
MAPE_Y1_2 = MAPE(y_Y_test1, y_Y_pred1_2)
MAPE_T1_2 = MAPE(y_T_test1, y_T_pred1_2)
MAPE_1_2 = (MAPE_X1_2+MAPE_Y1_2+MAPE_T1_2)/3

# Data 1 --> Model 3
y_X_pred1_3 = model_X_a3.predict(X_X_test1)
y_Y_pred1_3 = model_Y_a3.predict(X_Y_test1)
y_T_pred1_3 = model_T_a3.predict(X_T_test1)
MAPE_X1_3 = MAPE(y_X_test1, y_X_pred1_3)
MAPE_Y1_3 = MAPE(y_Y_test1, y_Y_pred1_3)
MAPE_T1_3 = MAPE(y_T_test1, y_T_pred1_3)
MAPE_1_3 = (MAPE_X1_3+MAPE_Y1_3+MAPE_T1_3)/3

# Data 2 --> Model 1
y_X_pred2_1 = model_X_a1.predict(X_X_test2)
y_Y_pred2_1 = model_Y_a1.predict(X_Y_test2)
y_T_pred2_1 = model_T_a1.predict(X_T_test2)
MAPE_X2_1 = MAPE(y_X_test2, y_X_pred2_1)
MAPE_Y2_1 = MAPE(y_Y_test2, y_Y_pred2_1)
MAPE_T2_1 = MAPE(y_T_test2, y_T_pred2_1)
MAPE_2_1 = (MAPE_X2_1+MAPE_Y2_1+MAPE_T2_1)/3

# Data 2 --> Model 2
y_X_pred2_2 = model_X_a2.predict(X_X_test2)
y_Y_pred2_2 = model_Y_a2.predict(X_Y_test2)
y_T_pred2_2 = model_T_a2.predict(X_T_test2)
MAPE_X2_2 = MAPE(y_X_test2, y_X_pred2_2)
MAPE_Y2_2 = MAPE(y_Y_test2, y_Y_pred2_2)
MAPE_T2_2 = MAPE(y_T_test2, y_T_pred2_2)
MAPE_2_2 = (MAPE_X2_2+MAPE_Y2_2+MAPE_T2_2)/3

# Data 2 --> Model 3
y_X_pred2_3 = model_X_a3.predict(X_X_test2)
y_Y_pred2_3 = model_Y_a3.predict(X_Y_test2)
y_T_pred2_3 = model_T_a3.predict(X_T_test2)
MAPE_X2_3 = MAPE(y_X_test2, y_X_pred2_3)
MAPE_Y2_3 = MAPE(y_Y_test2, y_Y_pred2_3)
MAPE_T2_3 = MAPE(y_T_test2, y_T_pred2_3)
MAPE_2_3 = (MAPE_X2_3+MAPE_Y2_3+MAPE_T2_3)/3

# Data 3 --> Model 1
y_X_pred3_1 = model_X_a1.predict(X_X_test3)
y_Y_pred3_1 = model_Y_a1.predict(X_Y_test3)
y_T_pred3_1 = model_T_a1.predict(X_T_test3)
MAPE_X3_1 = MAPE(y_X_test3, y_X_pred3_1)
MAPE_Y3_1 = MAPE(y_Y_test3, y_Y_pred3_1)
MAPE_T3_1 = MAPE(y_T_test3, y_T_pred3_1)
MAPE_3_1 = (MAPE_X3_1+MAPE_Y3_1+MAPE_T3_1)/3

# Data 3 --> Model 2
y_X_pred3_2 = model_X_a2.predict(X_X_test3)
y_Y_pred3_2 = model_Y_a2.predict(X_Y_test3)
y_T_pred3_2 = model_T_a2.predict(X_T_test3)
MAPE_X3_2 = MAPE(y_X_test3, y_X_pred3_2)
MAPE_Y3_2 = MAPE(y_Y_test3, y_Y_pred3_2)
MAPE_T3_2 = MAPE(y_T_test3, y_T_pred3_2)
MAPE_3_2 = (MAPE_X3_2+MAPE_Y3_2+MAPE_T3_2)/3

# Data 3 --> Model 3
y_X_pred3_3 = model_X_a3.predict(X_X_test3)
y_Y_pred3_3 = model_Y_a3.predict(X_Y_test3)
y_T_pred3_3 = model_T_a3.predict(X_T_test3)
MAPE_X3_3 = MAPE(y_X_test3, y_X_pred3_3)
MAPE_Y3_3 = MAPE(y_Y_test3, y_Y_pred3_3)
MAPE_T3_3 = MAPE(y_T_test3, y_T_pred3_3)
MAPE_3_3 = (MAPE_X3_3+MAPE_Y3_3+MAPE_T3_3)/3

print("TABLE BUCKINGHAM")
data = [[MAPE_1_1, MAPE_2_1, MAPE_3_1],
[MAPE_1_2, MAPE_2_2, MAPE_3_2],
[MAPE_1_3, MAPE_2_3 , MAPE_3_3]]
col=["Data vehicle 1", "Data vehicle 2", "Data vehicle 3"]
row=["Model vehicle 1", "Model vehicle 2", "Model vehicle 3"]
print(pd.DataFrame(data, row, col))


#################################################################################################################
#                                                AUGMENTED BUCKINGHAM BASED                                     #
#################################################################################################################

###############################################################
#                     RACECAR  (Vehicle 1)                    #
###############################################################                                       
#Load data set
data = pd.read_excel('racecar_data.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################
X_a1 = data.drop(["v_i","l","mu","a","delta","pi1","pi2","pi3","X","Y","theta","m","N_f","N_r","pi11"],axis=1)
X_a1 = X_a1.values



#Only use y as output
y_X_a1 = data["pi1"].values
y_Y_a1 = data["pi2"].values
y_T_a1 = data["pi3"].values


#Create models
model_X_a1 = XGBRegressor()
model_Y_a1 = XGBRegressor()
model_T_a1 = XGBRegressor()  

#Split data
X_X_train1, X_X_test1, y_X_train1, y_X_test1 = train_test_split(X_a1, y_X_a1, test_size=size, random_state=42)
X_Y_train1, X_Y_test1, y_Y_train1, y_Y_test1 = train_test_split(X_a1, y_Y_a1, test_size=size, random_state=42)    
X_T_train1, X_T_test1, y_T_train1, y_T_test1 = train_test_split(X_a1, y_T_a1, test_size=size, random_state=42)        

#Fit models
model_X_a1.fit(X_X_train1,y_X_train1)
model_Y_a1.fit(X_Y_train1,y_Y_train1)
model_T_a1.fit(X_T_train1,y_T_train1)



###############################################################
#                      LIMO  (Vehicle 2)                      #
###############################################################                                       
#Load data set
data = pd.read_excel('limo_data.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

X_a2 = data.drop(["v_i","l","mu","a","delta","pi1","pi2","pi3","X","Y","theta","m","N_f","N_r","pi11"],axis=1)
X_a2 = X_a2.values



#Only use y as output
y_X_a2 = data["pi1"].values
y_Y_a2 = data["pi2"].values
y_T_a2 = data["pi3"].values


#Create models
model_X_a2 = XGBRegressor()
model_Y_a2 = XGBRegressor()
model_T_a2 = XGBRegressor()  

#Split data
X_X_train2, X_X_test2, y_X_train2, y_X_test2 = train_test_split(X_a2, y_X_a2, test_size=size, random_state=42)
X_Y_train2, X_Y_test2, y_Y_train2, y_Y_test2 = train_test_split(X_a2, y_Y_a2, test_size=size, random_state=42)    
X_T_train2, X_T_test2, y_T_train2, y_T_test2 = train_test_split(X_a2, y_T_a2, test_size=size, random_state=42)        

#Fit models
model_X_a2.fit(X_X_train2,y_X_train2)
model_Y_a2.fit(X_Y_train2,y_Y_train2)
model_T_a2.fit(X_T_train2,y_T_train2)

###############################################################
#                      X-MAXX  (Vehicle 3)                    #
###############################################################                                       
#Load data set
data = pd.read_excel('Xmaxx_data.xlsx',index_col=False)

###############################################################
#             CREATE AND TRAIN PHYSICAL MODELS                #
###############################################################

X_a3 = data.drop(["v_i","l","mu","a","delta","pi1","pi2","pi3","X","Y","theta","m","N_f","N_r","pi11"],axis=1)
X_a3 = X_a3.values



#Only use y as output
y_X_a3 = data["pi1"].values
y_Y_a3 = data["pi2"].values
y_T_a3 = data["pi3"].values


#Create models
model_X_a3 = XGBRegressor()
model_Y_a3 = XGBRegressor()
model_T_a3 = XGBRegressor()  

#Split data
X_X_train3, X_X_test3, y_X_train3, y_X_test3 = train_test_split(X_a3, y_X_a3, test_size=size, random_state=42)
X_Y_train3, X_Y_test3, y_Y_train3, y_Y_test3 = train_test_split(X_a3, y_Y_a3, test_size=size, random_state=42)    
X_T_train3, X_T_test3, y_T_train3, y_T_test3 = train_test_split(X_a3, y_T_a3, test_size=size, random_state=42)        

#Fit models
model_X_a3.fit(X_X_train3,y_X_train3)
model_Y_a3.fit(X_Y_train3,y_Y_train3)
model_T_a3.fit(X_T_train3,y_T_train3)

####################################################################################################################
#                                               AUGMENTED BUCKINGHAM  TEST                                         #
####################################################################################################################

# Data 1 --> Model 1
y_X_pred1_1 = model_X_a1.predict(X_X_test1)
y_Y_pred1_1 = model_Y_a1.predict(X_Y_test1)
y_T_pred1_1 = model_T_a1.predict(X_T_test1)
MAPE_X1_1 = MAPE(y_X_test1, y_X_pred1_1)
MAPE_Y1_1 = MAPE(y_Y_test1, y_Y_pred1_1)
MAPE_T1_1 = MAPE(y_T_test1, y_T_pred1_1)
MAPE_1_1 = (MAPE_X1_1+MAPE_Y1_1+MAPE_T1_1)/3

# Data 1 --> Model 2
y_X_pred1_2 = model_X_a2.predict(X_X_test1)
y_Y_pred1_2 = model_Y_a2.predict(X_Y_test1)
y_T_pred1_2 = model_T_a2.predict(X_T_test1)
MAPE_X1_2 = MAPE(y_X_test1, y_X_pred1_2)
MAPE_Y1_2 = MAPE(y_Y_test1, y_Y_pred1_2)
MAPE_T1_2 = MAPE(y_T_test1, y_T_pred1_2)
MAPE_1_2 = (MAPE_X1_2+MAPE_Y1_2+MAPE_T1_2)/3

# Data 1 --> Model 3
y_X_pred1_3 = model_X_a3.predict(X_X_test1)
y_Y_pred1_3 = model_Y_a3.predict(X_Y_test1)
y_T_pred1_3 = model_T_a3.predict(X_T_test1)
MAPE_X1_3 = MAPE(y_X_test1, y_X_pred1_3)
MAPE_Y1_3 = MAPE(y_Y_test1, y_Y_pred1_3)
MAPE_T1_3 = MAPE(y_T_test1, y_T_pred1_3)
MAPE_1_3 = (MAPE_X1_3+MAPE_Y1_3+MAPE_T1_3)/3

# Data 2 --> Model 1
y_X_pred2_1 = model_X_a1.predict(X_X_test2)
y_Y_pred2_1 = model_Y_a1.predict(X_Y_test2)
y_T_pred2_1 = model_T_a1.predict(X_T_test2)
MAPE_X2_1 = MAPE(y_X_test2, y_X_pred2_1)
MAPE_Y2_1 = MAPE(y_Y_test2, y_Y_pred2_1)
MAPE_T2_1 = MAPE(y_T_test2, y_T_pred2_1)
MAPE_2_1 = (MAPE_X2_1+MAPE_Y2_1+MAPE_T2_1)/3

# Data 2 --> Model 2
y_X_pred2_2 = model_X_a2.predict(X_X_test2)
y_Y_pred2_2 = model_Y_a2.predict(X_Y_test2)
y_T_pred2_2 = model_T_a2.predict(X_T_test2)
MAPE_X2_2 = MAPE(y_X_test2, y_X_pred2_2)
MAPE_Y2_2 = MAPE(y_Y_test2, y_Y_pred2_2)
MAPE_T2_2 = MAPE(y_T_test2, y_T_pred2_2)
MAPE_2_2 = (MAPE_X2_2+MAPE_Y2_2+MAPE_T2_2)/3

# Data 2 --> Model 3
y_X_pred2_3 = model_X_a3.predict(X_X_test2)
y_Y_pred2_3 = model_Y_a3.predict(X_Y_test2)
y_T_pred2_3 = model_T_a3.predict(X_T_test2)
MAPE_X2_3 = MAPE(y_X_test2, y_X_pred2_3)
MAPE_Y2_3 = MAPE(y_Y_test2, y_Y_pred2_3)
MAPE_T2_3 = MAPE(y_T_test2, y_T_pred2_3)
MAPE_2_3 = (MAPE_X2_3+MAPE_Y2_3+MAPE_T2_3)/3

# Data 3 --> Model 1
y_X_pred3_1 = model_X_a1.predict(X_X_test3)
y_Y_pred3_1 = model_Y_a1.predict(X_Y_test3)
y_T_pred3_1 = model_T_a1.predict(X_T_test3)
MAPE_X3_1 = MAPE(y_X_test3, y_X_pred3_1)
MAPE_Y3_1 = MAPE(y_Y_test3, y_Y_pred3_1)
MAPE_T3_1 = MAPE(y_T_test3, y_T_pred3_1)
MAPE_3_1 = (MAPE_X3_1+MAPE_Y3_1+MAPE_T3_1)/3

# Data 3 --> Model 2
y_X_pred3_2 = model_X_a2.predict(X_X_test3)
y_Y_pred3_2 = model_Y_a2.predict(X_Y_test3)
y_T_pred3_2 = model_T_a2.predict(X_T_test3)
MAPE_X3_2 = MAPE(y_X_test3, y_X_pred3_2)
MAPE_Y3_2 = MAPE(y_Y_test3, y_Y_pred3_2)
MAPE_T3_2 = MAPE(y_T_test3, y_T_pred3_2)
MAPE_3_2 = (MAPE_X3_2+MAPE_Y3_2+MAPE_T3_2)/3

# Data 3 --> Model 3
y_X_pred3_3 = model_X_a3.predict(X_X_test3)
y_Y_pred3_3 = model_Y_a3.predict(X_Y_test3)
y_T_pred3_3 = model_T_a3.predict(X_T_test3)
MAPE_X3_3 = MAPE(y_X_test3, y_X_pred3_3)
MAPE_Y3_3 = MAPE(y_Y_test3, y_Y_pred3_3)
MAPE_T3_3 = MAPE(y_T_test3, y_T_pred3_3)
MAPE_3_3 = (MAPE_X3_3+MAPE_Y3_3+MAPE_T3_3)/3

print("TABLE AUGMENTED BUCKINGHAM")
data = [[MAPE_1_1, MAPE_2_1, MAPE_3_1],
[MAPE_1_2, MAPE_2_2, MAPE_3_2],
[MAPE_1_3, MAPE_2_3 , MAPE_3_3]]
col=["Data vehicle 1", "Data vehicle 2", "Data vehicle 3"]
row=["Model vehicle 1", "Model vehicle 2", "Model vehicle 3"]
print(pd.DataFrame(data, row, col))











