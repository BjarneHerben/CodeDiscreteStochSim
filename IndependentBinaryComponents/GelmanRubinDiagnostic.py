import numpy as np
import sys
import numba
from time import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from IBCHelp import *

## The functions to calculate the required quantities

#Calculate the empirical mean of a 1D array
def mean(x):
    x = np.array(x)
    return np.sum(x)/len(x)

#Calculate the empirical variance of a 1D array
def var(x):
    n = len(x)
    x = np.array(x)
    mu = mean(x)
    res_array = (1/(n-1))*(x - np.ones(n)*mu)**2
    return np.sum(res_array)

#Calculate the covariance between two 1D arrays
def cov(x, y):
    n = len(x)
    x = np.array(x)
    y = np.array(y)
    mu_x = mean(x)
    mu_y = mean(y)
    res_array = (1/n)*(x - mu_x*np.ones(n))*(y-mu_y*np.ones(n))
    return np.sum(res_array)

#Creating the overdispersed distribution
#and sampling the start for each sequence
def start_sequences(dim, #The dimension of the state space
                    pvals, #The p value for each binary component
                    N, #The number of draws from the overdispersed distribution
                    num_rw, #The number of random walk steps to be part of the overdispersed
                    m #The number of sequences that need to be started
                    ):
    #Creating the (empirical) overdispersed distribution
    #The placeholders
    N_samples = np.zeros((N,dim))
    starting_point = np.zeros((m,dim))
    energy_rate = np.zeros(N)
    
    #Sample the N realizations from the overdispersed distribution
    for i in np.arange(0,N):
        for step in np.arange(0, num_rw):
            #A random starting state
            current_state = np.random.randint(0, 2, dim)
            
            #Sampling a random generator
            component = np.random.randint(0, dim)
            
            #The probability of accepting the proposed state
            target_ratio = prob_ratio(component, current_state, pvals)
            prob_acc = np.minimum(1, target_ratio)
            
            #The acceptation-rejection step
            if np.random.rand(1) <= prob_acc:
                #Apply the generator to obtain the new state
                current_state = apply_gen(current_state, component)
            
        #Insert the results
        N_samples[i, :] = current_state
        energy_rate[i] = np.abs(1/energy_function(current_state, pvals, dim))*(1e6)
    
    #Importance resampling without replacement from the overdispered distribution
    for i in np.arange(0,m):
        #Sampling the row index of the state
        row = np.searchsorted(energy_rate.cumsum(), np.random.rand(1)*(energy_rate.sum()))[0]

        #The without replacement part
        energy_rate[row] = 0
        
        #Adding to the starting states
        starting_point[i,:] = N_samples[row,:]
    return N_samples, energy_rate, starting_point


#The function that returns the R ratio
def gelman_rubin(traces, #A list with all the Gelman-Rubin traces for each starting sequence
                    n, #The number of iterations for each simulated sequence (2m are inserted) 
                    m #The number of different starting sequences
                    ):
    #Placeholder arrays
    averages = np.zeros(m)
    sq_mean = np.zeros(m)
    variances = np.zeros(m)
    
    #Performing some basic operations
    for i in np.arange(m):
        trace = traces[i]
        
        #Removing the first n iterations
        traces[i] = trace[n+1:]
        
        #All the averages etc
        averages[i] = mean(trace[n+1:])
        variances[i] = var(trace[n+1:])
        sq_mean[i] = mean(trace[n+1:]**2)
        
    #Calculating the (sqrt) R ratio
    
    #Variance between the m sequence means
    B_n = var(averages)
    
    #Average of the in-sequence variances
    W = mean(variances)
    
    #Target mean estimated by the average of all simulations
    target = mean(averages)
    
    #Weighted estimate of the target variance
    sigma = ((n-1)/(n))*W + B_n
    
    #Large V for the t-distribution
    V = sigma + (1/m)*B_n
    
    #The estimated variance for V
    varV = (((n-1)/n)**2)*(1/m)*var(variances) + (((m+1)/(m*n))**2)*(2/(m-1))*(B_n*n)**2
    varV += 2*(((m+1)*(n-1))/(m*(n**2)))*(n/m)*(cov(variances, averages**2) - 2*target*cov(variances, averages))
        
    #The degrees of freedom for the t-distribution
    df = (2*V**2)/varV
    
    #The square root of the R-estimate (potential scale reduction)
    R = np.sqrt((V/W)*(df/(df-2)))
    
    return R