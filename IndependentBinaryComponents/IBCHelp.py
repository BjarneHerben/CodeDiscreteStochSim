import numpy as np
import sys
import numba
from time import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming

#Function that calculates the log energy of the target density
@numba.njit()
def energy_function(state, pvals, dim):
    ans = 0
    for i in np.arange(dim):
        ans += (1-state[i])*np.log(pvals[i]) + state[i]*np.log(1-pvals[i])
    return ans

#Function that calculates the probability ratio
@numba.njit()
def prob_ratio(component, current_state, pvals):
    if current_state[component] == 1:
        ans = np.log(pvals[component]) - np.log(1-pvals[component])
    else:
        ans = np.log(1-pvals[component]) - np.log(pvals[component])
    return np.exp(ans)

#Function that performs the move/applied the generator
@numba.njit()
def apply_gen(current_state, component):
    #Flip the single component
    current_state[component] = 0**(current_state[component])
    return current_state

#Function that calculates the ratios
def calculate_rates(current_state, pvals, dim, g):
    rates = np.zeros(dim)
    for i in np.arange(dim):
        if current_state[i] == 1:
            rates[i] = g(np.exp(np.log(pvals[i]) - np.log(1-pvals[i])))
        else:
            rates[i] = g(np.exp(np.log(1-pvals[i]) - np.log(pvals[i])))
    return rates

#Function that updates the rates
def update_rates(current_state, rates, rates_sum, component, pvals, g):
    if current_state[component] == 1:
        rates[component] = g(np.exp(np.log(pvals[component]) - np.log(1-pvals[component])))
    else:
        rates[component] = g(np.exp(np.log(1-pvals[component]) - np.log(pvals[component])))
    return rates
        
    