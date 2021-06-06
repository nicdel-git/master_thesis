### Black Scholes data generation

import numpy as np
import pandas as pd
import scipy.stats as si
import tensorflow as tf
import tensorflow_probability as tfp
import math

#call option pricing formula
def black_scholes(S, K, T, r, sigma):
    
    if(T == 0):
        return max(0,S-K)
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
        call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    
    return call

#put option pricing formula
def black_scholes_put(S, K, T, r, sigma):
    
    if(T == 0):
        return max(0,K-S)
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        
        put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    
    return put

#call option price computed by Monte Carlo simulation
def mc_BS(S, K, T, r, sigma, nSim):
    WT = np.random.normal(0,np.sqrt(T),nSim)
    ST = S*np.exp((r - 0.5*sigma**2)*T + sigma*WT)
    payoff = np.exp(-r*T)*np.maximum(ST-K,0)
    return np.mean(payoff)

#delta for a call option
def delta_call(S, K, T, r, sigma):
    
    if(T == 0):
        if((S-K)>0):
            return 1
        else:
            return 0
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
        delta_call = si.norm.cdf(d1, 0.0, 1.0)
    
    return delta_call

#theta for a call option
def theta_call(S, K, T, r, sigma):
    
    if(T == 0):
        return 0
    else:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        prob_density = 1 / np.sqrt(2 * np.pi) * np.exp(-d1 ** 2 * 0.5)

        theta = (-sigma * S * prob_density) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    
    return theta

#generate training data using linspace sequences for the input grid
def training_data(t0, t1, T, S_min, S_max, K, r, sigma, N, virtual_points = True):
    t_space = np.linspace(t0, t1, N)
    S_space = np.linspace(S_min, S_max, N)

    space = []
    for t in t_space:
        for S in S_space:
            space.append([t, S])

    #add 50 virtual data points on edge of grid
    if(virtual_points):
        for S in [S_max + 1, S_max + 2]:
            for t in np.linspace(0,T,10):
                space.append([t,S])
        for S in [S_min - 2, S_min - 1]:
            for t in np.linspace(0,T,10):
                space.append([t,S])
        for S in np.linspace(S_min,S_max,10):
            for t in [T]:
                space.append([t,S])

    space_frame = pd.DataFrame(space,columns=['t', 'S'])

    price = []
    price_mc = []
    delta = []
    theta = []
    for index, row in space_frame.iterrows():
        price.append(black_scholes(row['S'], K, T - row['t'], r, sigma))
        price_mc.append(mc_BS(row['S'], K, T - row['t'], r, sigma, 2500))
        delta.append(delta_call(row['S'], K, T - row['t'], r, sigma))
        theta.append(theta_call(row['S'], K, T - row['t'], r, sigma))

    space_frame['price'] = price
    space_frame['price_mc'] = price_mc
    space_frame['delta'] = delta
    space_frame['theta'] = theta

    return space_frame

#egenerate training data using halton sequences for the input grid
def training_data_halton(t0, t1, T, S_min, S_max, K, r, sigma, N, virtual_points = True):
    x = tfp.mcmc.sample_halton_sequence(2, num_results=N*N, dtype=tf.float64)
    t_space = t0 + t1*x[:,0]
    t_space = np.array(t_space)
    S_space = S_min + (S_max-S_min)*x[:,1]
    S_space = np.array(S_space)

    space = list(zip(t_space, S_space))

    #50 virtual data points
    if(virtual_points):
        for S in [S_max + 1, S_max + 2]:
            for t in np.linspace(0,T,10):
                space.append([t,S])
        for S in [S_min - 2, S_min - 1]:
            for t in np.linspace(0,T,10):
                space.append([t,S])
        for S in np.linspace(S_min,S_max,10):
            for t in [T]:
                space.append([t,S])

    space_frame = pd.DataFrame(space,columns=['t', 'S'])

    price = []
    price_mc = []
    delta = []
    theta = []
    for index, row in space_frame.iterrows():
        price.append(black_scholes(row['S'], K, T - row['t'], r, sigma))
        price_mc.append(mc_BS(row['S'], K, T - row['t'], r, sigma, 2500))
        delta.append(delta_call(row['S'], K, T - row['t'], r, sigma))
        theta.append(theta_call(row['S'], K, T - row['t'], r, sigma))

    space_frame['price'] = price
    space_frame['price_mc'] = price_mc
    space_frame['delta'] = delta
    space_frame['theta'] = theta

    return space_frame

#generate test data: here a curve of call prices of varying strikes
def test_data(T, t, S_min, S_max, K, r, sigma, N):
    t_test = t*np.ones(N)
    S_test = np.linspace(S_min,S_max,N)
    test_frame = pd.DataFrame(zip(t_test,S_test),columns=['t', 'S'])

    price = []
    price_mc = []
    delta = []
    theta = []
    for index, row in test_frame.iterrows():
        price.append(black_scholes(row['S'], K, T - row['t'], r, sigma))
        price_mc.append(mc_BS(row['S'], K, T - row['t'], r, sigma, 2500))
        delta.append(delta_call(row['S'], K, T - row['t'], r, sigma))
        theta.append(theta_call(row['S'], K, T - row['t'], r, sigma))

    test_frame['price'] = price
    test_frame['price_mc'] = price_mc
    test_frame['delta'] = delta
    test_frame['theta'] = theta

    return test_frame

#generate the training and test data in the linspace grid case
def call_data(virtual_points = True):
    #here we define the 'standard' variable values
    K = 50
    t0 = -0.01
    t1 = 0.38
    T = 0.4
    S_min = 30
    S_max = 70
    r = 0.04
    sigma = 0.22
    N = 20
    N_test = 51
    t_test = 0.2
    
    t1 = training_data(t0, t1, T, S_min, S_max, K, r, sigma, N, virtual_points)
    t2 = test_data(T, t_test, S_min, S_max, K, r, sigma, N_test)
    
    return t1, t2

#generate the training and test data in the halton grid case
def call_data_halton(virtual_points = True):
    #here we define the 'standard' variable values
    K = 50
    t0 = -0.01
    t1 = 0.38
    T = 0.4
    S_min = 30
    S_max = 70
    r = 0.04
    sigma = 0.22
    N = 20
    N_test = 30
    t_test = 0.2
    
    t1 = training_data_halton(t0, t1, T, S_min, S_max, K, r, sigma, N, virtual_points)
    t2 = test_data(T, t_test, S_min, S_max, K, r, sigma, N_test)
    
    return t1, t2


#Generate data with binomial tree model
#Code taken and modified from https://en.wikipedia.org/wiki/Binomial_options_pricing_model
def BinomialPricing(S, K, T, r, sigma, q, n, exercise_type, option_type):
    if(T == 0):
        if(option_type == 'Call'):
            return (S-K+np.abs(S-K))/2.0
        else:
            return (K-S+np.abs(K-S))/2.0
    deltaT = T / n
    up = np.exp(sigma * np.sqrt(deltaT))
    p0 = (up*np.exp(-q * deltaT) - np.exp(-r * deltaT)) / (up**2 - 1)
    p1 = np.exp(-r * deltaT) - p0
    #Set final values of tree
    p = np.zeros(n+1)
    for i in range(n+1):
        if(option_type == 'Call'):
            p[i] = S * up**(2*i - n) - K
        else:
            p[i] = K - S * up**(2*i - n)
        if(p[i] < 0):
            p[i] = 0
    #Backpropagate final payoffs
    for j in range(n-1, -1, -1):
        for i in range(j+1):
            p[i] = p0 * p[i+1] + p1 * p[i]
            if(exercise_type == 'American'):
                if(option_type == 'Call'):
                    exercise = S * up**(2*i - j) - K
                else:
                    exercise = K - S * up**(2*i - j)
                if(p[i] < exercise):
                    p[i] = exercise
            if(exercise_type == 'Bermuda' and (j == math.floor(n/3) or j == math.floor(2*n/3))):
                if(option_type == 'Call'):
                    exercise = S * up**(2*i - j) - K
                else:
                    exercise = K - S * up**(2*i - j)
                if(p[i] < exercise):
                    p[i] = exercise
    return p[0]


def call_binomial_data():
    T_max = 0.4
    S = 50
    sigma = 0.22
    r = 0.04
    K_max = 70
    K_min = 30
    n = 100
    N = 11
    K = np.linspace(K_min,K_max,N)
    T = np.linspace(0, T_max, N)

    #First training data set
    space = []
    for t in T:
        for k in K:
            space.append([t, k])
    data_frame = pd.DataFrame(space,columns=['T', 'K'])

    price_call_Am = []
    price_call_Eur = []
    price_put_Am = []
    price_put_Eur = []
    for index, row in data_frame.iterrows():
        price_call_Am.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'American', 'Call'))
        price_call_Eur.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'European', 'Call'))
        price_put_Am.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'American', 'Put'))
        price_put_Eur.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'European', 'Put'))
    data_frame['price_call_Am'] = price_call_Am
    data_frame['price_call_Eur'] = price_call_Eur
    data_frame['price_put_Am'] = price_put_Am
    data_frame['price_put_Eur'] = price_put_Eur


    #Second, shifted, training data set
    delta_K = (K_max - K_min)/(2*(N-1))
    delta_T = T_max/(2*(N-1))
    K2 = np.linspace(K_min + delta_K, K_max + delta_K , N)
    T2 = np.linspace(0 + delta_T, T_max + delta_T, N)

    space2 = []
    for t in T2:
        for k in K2:
            space2.append([t, k])
    data_frame2 = pd.DataFrame(space2,columns=['T', 'K'])

    price_call_Am2 = []
    price_call_Eur2 = []
    price_put_Am2 = []
    price_put_Eur2 = []
    for index, row in data_frame2.iterrows():
        price_call_Am2.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'American', 'Call'))
        price_call_Eur2.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'European', 'Call'))
        price_put_Am2.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'American', 'Put'))
        price_put_Eur2.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'European', 'Put'))
    data_frame2['price_call_Am'] = price_call_Am2
    data_frame2['price_call_Eur'] = price_call_Eur2
    data_frame2['price_put_Am'] = price_put_Am2
    data_frame2['price_put_Eur'] = price_put_Eur2


    #Test data set
    N_prime = 81
    K_prime = np.linspace(30,70,N_prime)
    T_prime = [T_max] #np.linspace(0, T_max, N_prime)

    space_prime = []
    for t in T_prime:
        for k in K_prime:
            space_prime.append([t, k])
    data_frame_prime = pd.DataFrame(space_prime,columns=['T', 'K'])

    price_call_Am_prime = []
    price_call_Eur_prime = []
    price_put_Am_prime = []
    price_put_Eur_prime = []
    for index, row in data_frame_prime.iterrows():
        price_call_Am_prime.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'American', 'Call'))
        price_call_Eur_prime.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'European', 'Call'))
        price_put_Am_prime.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'American', 'Put'))
        price_put_Eur_prime.append(BinomialPricing(S, row['K'], row['T'], r, sigma, 0, n, 'European', 'Put'))
    data_frame_prime['price_call_Am'] = price_call_Am_prime
    data_frame_prime['price_call_Eur'] = price_call_Eur_prime
    data_frame_prime['price_put_Am'] = price_put_Am_prime
    data_frame_prime['price_put_Eur'] = price_put_Eur_prime

    return data_frame, data_frame2, data_frame_prime