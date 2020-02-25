#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:22:53 2019

@author: aadityagurav
"""

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates=[]
prices=[]
#print("Debug")
print(plt.get_backend())
def get_data(filename):
    with open (filename,'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        i=0
        for row in csvFileReader:
            dates.append(i)
            i=i+1
            prices.append(float(row[1]))
        #print(dates,prices)
    return

def predict_prices(dates,prices,x):
    dates=np.reshape(dates,(len(dates),1))
    #print(dates)
    svr_lin=SVR(kernel= 'linear',C=1e3)
    svr_poly=SVR(kernel= 'poly', C=1e3, degree=2)
    svr_rbf=SVR(kernel= 'rbf',C=1e3,gamma=0.1)
    svr_lin.fit(dates,prices)
    svr_poly.fit(dates,prices)
    svr_rbf.fit(dates,prices)
    
    plt.subplot(2,1,1)
    plt.scatter(dates,prices,color='black',label='Data')
    plt.subplot(2,1,2)
    plt.plot(dates,svr_rbf.predict(dates),color='red',label='RBF Model')
    plt.subplot(2,2,1)
    plt.plot(dates,svr_lin.predict(dates),color='green',label='Linear Model')
    plt.subplot(2,2,2)
    plt.plot(dates,svr_poly.predict(dates),color='blue',label='Polynomial Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0],svr_lin.predict(x)[0],svr_poly.predict(x)[0]

get_data('GOOG-2.csv')
#print("debug1")
predicted_prices=predict_prices(dates,prices,-1)
#print("debug2")
#print(type(predicted_prices))
print("RBF KERNEL:%.2f"%predicted_prices[0])
print("LINEAR KERNEL:%.2f"%predicted_prices[1])
print("POLYNOMIAL KERNEL:%.2f"%predicted_prices[2])
print(prices[-1])
plt.show()
#print("debug2")
