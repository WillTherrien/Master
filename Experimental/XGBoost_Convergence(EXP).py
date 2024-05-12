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
import matplotlib.pyplot as plt
from textwrap import wrap
from sklearn.metrics import mean_absolute_error as MAE
from rich.console import Console
from rich.table import Table

MAE_X = []
MAE_Y = []
MAE_T = []
MAE_P1 = []
MAE_P2 = []
MAE_P3 = []
MAE_P1a = []
MAE_P2a = []
MAE_P3a = []
T_SIZE  = []
S = []
nbr_test = 1.0
for i in range(0,10):
    
    ###############################################################
    #               READ DATA FROM EXCEL FILE                     #
    ###############################################################                                       
    #Load data set
    data = pd.read_excel('Xmaxx_data.xlsx',index_col=False, engine='openpyxl')

    ###############################################################
    #             CREATE AND TRAIN PHYSICAL MODELS                #
    ###############################################################

    #Only use physical input (v_i, length, a and delta)
    X_p = data.drop(["pi1","pi2","pi3","pi4","pi5","pi6","pi7","pi8","pi9","pi10","X","Y","theta"],axis=1)
    X_p = X_p.values
    
    


    #Only use y as output
    y_X_p = data["X"].values
    y_Y_p = data["Y"].values
    y_T_p = data["theta"].values
    
    #Size
    nbr_test = nbr_test*2.0
    if nbr_test > len(X_p)*0.8:
        nbr_test = len(X_p)*0.8
    size = (len(X_p)-nbr_test)/len(X_p)
    S.append(str(nbr_test)+" ("+str(round((1-size)*100,2))+"% du data)")
    
    #Create models
    model_X_p = XGBRegressor()
    model_Y_p = XGBRegressor()
    model_T_p = XGBRegressor()

    #Split data
    X_X_train, X_X_test, y_X_train, y_X_test = train_test_split(X_p, y_X_p, test_size=size, random_state=42)
    X_Y_train, X_Y_test, y_Y_train, y_Y_test = train_test_split(X_p, y_Y_p, test_size=size, random_state=42)    
    X_T_train, X_T_test, y_T_train, y_T_test = train_test_split(X_p, y_T_p, test_size=size, random_state=42)   

    #Fit models
    model_X_p.fit(X_X_train,y_X_train)
    model_Y_p.fit(X_Y_train,y_Y_train)
    model_T_p.fit(X_T_train,y_T_train)

    #Predict
    y_X_pred = model_X_p.predict(X_X_test)
    y_Y_pred = model_Y_p.predict(X_Y_test)
    y_T_pred = model_T_p.predict(X_T_test)

    MAE_X.append(MAE(y_X_test, y_X_pred))
    MAE_Y.append(MAE(y_Y_test, y_Y_pred))
    MAE_T.append(MAE(y_T_test, y_T_pred))


    ###############################################################
    #               BUCKINGHAM DIMENSIONLESS MODELS               #
    ###############################################################

    #Only use physical input (v_i, length, a and delta)
    X_b = data.drop(["v_i","l","a","delta","mu","Nf","Nr","g","pi1","pi2","pi3","pi9","pi10","X","Y","theta"],axis=1)
    X_b = X_b.values

    list_L = data["l"].values

    #Only use y as output
    y_X_b = data["pi1"].values
    y_Y_b = data["pi2"].values
    y_T_b = data["pi3"].values


    #Create models
    model_X_b = XGBRegressor()
    model_Y_b = XGBRegressor()
    model_T_b = XGBRegressor()  

    #Split data
    X_X_train, X_X_test, y_X_train, y_X_test = train_test_split(X_b, y_X_b, test_size=size, random_state=42)
    X_Y_train, X_Y_test, y_Y_train, y_Y_test = train_test_split(X_b, y_Y_b, test_size=size, random_state=42)    
    X_T_train, X_T_test, y_T_train, y_T_test = train_test_split(X_b, y_T_b, test_size=size, random_state=42)

    fake, l , fake ,fake = train_test_split(list_L, y_X_p, test_size=size, random_state=42)             
    
    #Fit models
    model_X_b.fit(X_X_train,y_X_train)
    model_Y_b.fit(X_Y_train,y_Y_train)
    model_T_b.fit(X_T_train,y_T_train)

    #Predict
    y_X_pred = model_X_b.predict(X_X_test)
    y_Y_pred = model_Y_b.predict(X_Y_test)
    y_T_pred = model_T_b.predict(X_T_test)

    MAE_P1.append(MAE(y_X_test*l, y_X_pred*l))
    MAE_P2.append(MAE(y_Y_test*l, y_Y_pred*l))
    MAE_P3.append(MAE(y_T_test, y_T_pred)) 
    
    ###############################################################
    #                 AUGMENTED BUCKINGHAM MODEL                  #
    ###############################################################

    #Only use physical input (v_i, length, a and delta)
    X_a = data.drop(["v_i","l","a","delta","mu","Nf","Nr","g","pi1","pi2","pi3","X","Y","theta"],axis=1)
    X_a = X_a.values



    #Only use y as output
    y_X_a = data["pi1"].values
    y_Y_a = data["pi2"].values
    y_T_a = data["pi3"].values


    #Create models
    model_X_a = XGBRegressor()
    model_Y_a = XGBRegressor()
    model_T_a = XGBRegressor()  

    #Split data
    X_X_train, X_X_test, y_X_train, y_X_test = train_test_split(X_a, y_X_a, test_size=size, random_state=42)
    X_Y_train, X_Y_test, y_Y_train, y_Y_test = train_test_split(X_a, y_Y_a, test_size=size, random_state=42)    
    X_T_train, X_T_test, y_T_train, y_T_test = train_test_split(X_a, y_T_a, test_size=size, random_state=42)        

    #Fit models
    model_X_a.fit(X_X_train,y_X_train)
    model_Y_a.fit(X_Y_train,y_Y_train)
    model_T_a.fit(X_T_train,y_T_train)

    #Predict
    y_X_pred = model_X_a.predict(X_X_test)
    y_Y_pred = model_Y_a.predict(X_Y_test)
    y_T_pred = model_T_a.predict(X_T_test)

    MAE_P1a.append(MAE(y_X_test*l, y_X_pred*l))
    MAE_P2a.append(MAE(y_Y_test*l, y_Y_pred*l))
    MAE_P3a.append(MAE(y_T_test, y_T_pred)) 
    
    T_SIZE.append(round(nbr_test))

pM = []
bM = []
aM = []
for i in range (0,len(MAE_X)):
    pM.append(" X: "+str(round(MAE_X[i],4))+"\n Y: "+str(round(MAE_Y[i],4))+"\n θ: "+str(round(MAE_T[i],4)))
    bM.append(" X: "+str(round(MAE_P1[i],4))+"\n Y: "+str(round(MAE_P2[i],4))+"\n θ: "+str(round(MAE_P3[i],4)))
    aM.append(" X: "+str(round(MAE_P1a[i],4))+"\n Y: "+str(round(MAE_P2a[i],4))+"\n θ: "+str(round(MAE_P3a[i],4)))

    
table = Table(show_lines=True)
columns = ["TRAINING DATA SIZE", "MODELS BASED ON DIMENSIONNALIZED PARAMS", "BUCKINGHAM BASED MODELS", "AUGMENTED BUCKINGHAM BASED MODELS"]
rows = [[S[0], pM[0], bM[0], aM[0]],
        [S[1], pM[1], bM[1], aM[1]],
        [S[2], pM[2], bM[2], aM[2]],
        [S[3], pM[3], bM[3], aM[3]],
        [S[4], pM[4], bM[4], aM[4]],
        [S[5], pM[5], bM[5], aM[5]],
        [S[6], pM[6], bM[6], aM[6]],
        [S[7], pM[7], bM[7], aM[7]],
        [S[8], pM[8], bM[8], aM[8]],
        [S[9], pM[9], bM[9], aM[9]],]
    

for column in columns:
    table.add_column(column)

for row in rows:
    table.add_row(*row, style='bright_green')

console = Console()
console.print(table)    
    

fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True)


ax1.plot(T_SIZE, MAE_X,  '--b', marker="x",label="Traditional")
ax1.plot(T_SIZE, MAE_P1,  '--g',label="Buckingham")
ax1.plot(T_SIZE, MAE_P1a,  '--r',marker="^",label="Augmented Buckingham")
ax1.set_ylabel("\n".join(wrap('MAE (m) according to the position in X-axis',20)))

ax2.plot(T_SIZE, MAE_Y,  '--b',marker="x",label="Traditional")
ax2.plot(T_SIZE, MAE_P2,  '--g',label="Buckingham")
ax2.plot(T_SIZE, MAE_P2a,  '--r',marker="^",label="Augmented Buckingham")
ax2.set_ylabel("\n".join(wrap('MAE (m) according to the position in Y-axis',20)))

ax3.plot(T_SIZE, MAE_T,  '--b',marker="x",label="Traditional")
ax3.plot(T_SIZE, MAE_P3,  '--g',label="Buckingham")
ax3.plot(T_SIZE, MAE_P3a,  '--r',marker="^",label="Augmented Buckingham")
ax3.set_ylabel("\n".join(wrap('MAE (rad) according to the yaw of the vehicle',20)))
ax3.set_xlabel('Training dataset size')


ax1.legend(loc=1)
ax2.legend(loc=1)
ax3.legend(loc=1)




plt.show()
















