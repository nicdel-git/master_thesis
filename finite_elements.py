#Here is defined a Finite Element Method class
#This class can be used to produce estimated prices of any European payoff function

import matplotlib.pyplot as plt
import numpy as np
from BS_data import black_scholes
import pandas as pd
from sklearn.metrics import mean_squared_error

SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

#Call function
def C(K,S):
    if(S<K):
        return 0
    else:
        return S-K
    
#Butterfly function contructed from 3 evenly spaced call functions
def B(S, K, h):
    return (C(K-h,S)-2*C(K,S)+C(K+h,S))/h

def B_uneven(S, K1, K2, K3):
    h1 = K2 - K1
    h2 = K3 - K2
    a = 1/h1
    b = (h1 + h2)/(h1*h2)
    c = 1/h2
    return a*C(K1,S)-b*C(K2,S)+c*C(K3,S)

def B_coefs(K1, K2, K3):
    h1 = K2 - K1
    h2 = K3 - K2
    a = 1/h1
    b = (h1 + h2)/(h1*h2)
    c = 1/h2
    return a, b, c

#finite elements class
class fem:
    def __init__(self, K_min, K_max, K_vec, price_vec):
        self.K_min = K_min
        self.K_max = K_max
        self.K_vec = K_vec
        self.price_vec = price_vec
        
    def plot_hats(self, N_points=100):
        x = np.linspace(self.K_min,self.K_max,N_points)
        
        f, ax = plt.subplots(figsize=(12, 4))
        for i in range(len(self.K_vec)-2):
            b = []
            for j in x:
                b.append(B_uneven(j,self.K_vec[i],self.K_vec[i+1],self.K_vec[i+2]))
            ax.plot(x, b, linestyle = '--')
            
        return ax
    
    def gamma(self, f):
        gamma = []
        for i in range(len(self.K_vec)-2):
            gamma.append(f(self.K_vec[i+1]))
        return gamma
    
    def price(self, f):
        price = 0
        gamma = self.gamma(f)
        for j in range(len(self.K_vec)-2):
            coefs = B_coefs(self.K_vec[j],self.K_vec[j+1],self.K_vec[j+2])
            temp = gamma[j]*(coefs[0]*self.price_vec[j]-coefs[1]*self.price_vec[j+1]+coefs[2]*self.price_vec[j+2])
            price += temp
        return price
    
    def payoff_approx(self, f, N_points=100):
        x = np.linspace(self.K_min,self.K_max,N_points)
        gamma = self.gamma(f)
        y_pred = np.zeros(len(x))
        y_th = []
        for i in range(len(x)):
            y_th.append(f(x[i]))
            for j in range(len(self.K_vec)-2):
                y_pred[i] += gamma[j]*B_uneven(x[i],self.K_vec[j],self.K_vec[j+1],self.K_vec[j+2])
        error = mean_squared_error(y_th, y_pred)
        return y_th, y_pred, error
    
    def plot_prediction(self, f, N_points=100, show_hats=False):
        x = np.linspace(self.K_min,self.K_max,N_points)
        
        gamma = self.gamma(f)
        price_pred = self.price(f)
        y_th, y_pred, error = self.payoff_approx(f, N_points)
        
        f2, ax = plt.subplots(figsize=(12, 4))
        if(show_hats):
            for i in range(0,len(self.K_vec)-2):
                b = []
                for j in x:
                    b.append(gamma[i]*B_uneven(j,self.K_vec[i],self.K_vec[i+1],self.K_vec[i+2]))
                ax.plot(x, b, linestyle = '--')
        ax.plot(x, y_pred, alpha=1, color = 'r', linestyle = '--', label = 'FEM '+str(len(self.K_vec))+' points, MSE ='+str(round(error,4)) + ', price ='+str(round(price_pred,4)))
        ax.plot(x, y_th, alpha=0.5, color='b', linestyle='-', linewidth = 3, label = 'f(S)')
        ax.set_xlabel("Stock price")
        ax.set_ylabel("Payoff")
        ax.legend()
        return ax

def f(x, K_prime = 51):
    return C(K_prime,x)

def main_fem(fct = f):
    K_min = 30
    K_max = 70
    N_hat = 16
    T = 0.2
    S = 50
    sigma = 0.22
    r = 0.04
    N_points = 100

    K_vec = np.linspace(K_min,K_max,N_hat)
    price = []
    for i in K_vec:
        price.append(black_scholes(S, i, T, r, sigma))
    
    hat_class= fem(K_min, K_max, K_vec, price)
    ax = hat_class.plot_prediction(fct, N_points)
    plt.show()