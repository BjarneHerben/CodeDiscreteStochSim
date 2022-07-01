from locale import currency
import numpy as np
import sys
import numba
import copy
from time import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming

#Function that calculates the log energy of the target density
@numba.njit()
def energy_function(state, weights, dim):
    ans = 0
    for i in np.arange(dim):
        ans += np.log(weights[i, state[i]])
    return ans

#Function that calculates the probability ratio
def prob_ratio(gen, current_state, weights):
    ans = 0
    ans += (np.log(weights[gen[0], current_state[gen[1]]]) + np.log(weights[gen[1], current_state[gen[0]]]))
    ans -= (np.log(weights[gen[0], current_state[gen[0]]]) + np.log(weights[gen[1], current_state[gen[1]]]))
    return np.exp(ans)

#Function that performs the move/applies the generator
def apply_gen(current_state, gen):
    i, j = gen
    current_state[j], current_state[i] = current_state[i], current_state[j]
    return current_state

#Function that calculates the rates
def calculate_rates(current_state, weights, dim, g):
    rates = np.zeros((dim,dim))
    for i in np.arange(dim):
        for j in np.arange(dim):
            rates[i,j] = g(prob_ratio((i,j), current_state, weights))
            rates[j,i] = g(prob_ratio((i,j), current_state, weights))
    return rates.reshape(dim**2,)

#Function that updates the rates
def update_rates(current_state, rates, rates_sum, gen, weights, g):
    dim = int(np.sqrt(len(rates)))
    rates = rates.reshape((dim,dim))
    i, j = gen[0], gen[1]
    
    #Updating the ith row and column
    for k in np.arange(dim):
        rates[i,k] = g(prob_ratio((i,k), current_state, weights))
        rates[k,i] = g(prob_ratio((i,k), current_state, weights))
        
    #Updating the jth row and column
    for k in np.arange(dim):
        rates[j,k] = g(prob_ratio((j,k), current_state, weights))
        rates[k,j] = g(prob_ratio((j,k), current_state, weights))
    
    return rates.reshape(dim**2,)