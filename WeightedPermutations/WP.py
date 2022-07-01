import trace
import numpy as np
import copy as c
import math as m
import matplotlib.pyplot as plt
import time
import random as rand
from scipy.spatial.distance import hamming

#Some basic parameters
lmbd = 5
n = 500
N = 10**4
initial = np.arange(0, n, 1)

#Sampling the log normal weights as in figure 1 of the paper
weights = np.exp(np.random.normal(loc=0, scale=lmbd**2, size=(n,n)))

#sampling the weights from the Chi-square distributio
# weights = np.array((n,n))
# print(weights)

#The list that contains what is being swapped
gens = []
for i in range(n-1):
    for j in range(i+1, n):
        gens.append((i,j))

#Some balancing functions
g1 = lambda t: np.sqrt(t)
g2 = lambda t: t/(1+t)
id = lambda t: t

#The random walk sampler (compare acceptance ratio)
def rw_sampler(N=N, n=n, initial=initial, weights=weights, swap=swap):
    trace, net_trace = [initial], [initial]
    state = initial
    for l in range(N):
        k = rand.randrange(0,(1/2)*n*(n-1))
        i, j = swap[k]
        rho_i, rho_j = state[i], state[j]
        ratio = (weights[i, rho_j]*weights[j, rho_i])/(weights[i, rho_i]*weights[j, rho_j])
        alpha = min(1, ratio)
        u = np.random.uniform(size=1)
        if u <= alpha:
            state[i], state[j] = state[j], state[i]
            trace.append(c.copy(state))
            net_trace.append(c.copy(state))  
        else:
            trace.append(c.copy(state))
    acceptance_ratio = len(net_trace)/len(trace)
    print("Acceptance ratio: ", acceptance_ratio)
    return trace, net_trace, acceptance_ratio    
        

#Creating the proposal kernel
def Q_g(state, weights, g, n):
    distr, ratios = [], []
    for i in range(n-1):
        for j in range(i+1, n):
            rho_i, rho_j = state[i], state[j]
            ratio = (weights[i, rho_j]*weights[j, rho_i])/(weights[i, rho_i]*weights[j, rho_j])
            ratios.append(ratio)
            distr.append(g(ratio))
    Z_g = sum(distr)
    return distr/Z_g, Z_g, ratios

#Sample from the proposal kernel
def permutation_sample(N=N, n=n, initial=initial, g=g2, weights=weights, swap=swap):
    trace, net_trace = [initial], [initial]
    state = initial
    for i in range(N):
        Q, Z_gx, ratios = Q_g(state, weights, g, n) #The proposal kernel
        i = np.argmax(np.random.multinomial(1, Q)) #Proposed flipped components
        ratio = ratios[i]
        flip = swap[i]
        state[flip[0]], state[flip[1]] = state[flip[1]], state[flip[0]]
        Z_gy = Q_g(state, weights, g, n)[1]
        alpha = min(1, ratio*((Z_gx*g(1/ratio))/(Z_gy*g(ratio))))
        u = np.random.uniform(size=1)
        if u <= alpha:
            trace.append(c.copy(state))
            net_trace.append(c.copy(state))
        else:
            state[flip[0]], state[flip[1]] = state[flip[1]], state[flip[0]]
            trace.append(c.copy(state))
    acceptance_ratio = len(net_trace)/len(trace)
    print("Acceptance ratio: ", acceptance_ratio)
    return trace, net_trace, acceptance_ratio

#Making a traceplot
def trace_plot(traces, names, starting_state=initial):
    N, n = len(traces[0]), len(traces)
    mcmc_time = np.array(range(N))
    #As a summary statistic use the Hamming distance from the starting state
    for i in range(n):
        trace = traces[i]
        res= []
        for el in trace:
            dist = hamming(el, starting_state)
            res.append(dist)
        plt.plot(mcmc_time, res, label=names[i])
    plt.legend()
    plt.show()
    
#Plotting the ACF
def plot_ACF(trace, starting_state=initial):
    N, n = len(traces[0]), len(traces)
    #As a summary statistic use the Hamming distance from the starting state
    for i in range(n):
        trace = traces[i]
        res= []
        for el in trace:
            dist = hamming(el, starting_state)
            res.append(dist)
        plt.acorr(res, label=names[i])
    plt.legend()
    plt.show()

#Testing the sampling function
trace_g2, net_trace_g2, ar_g2 = permutation_sample(g=g2) #0.99 acceptance ratio
trace_rw, net_trace_rw, ar_rw = rw_sampler()

#Plotting the results
traces, names = [trace_g2, trace_rw], ["g2", "rw"]
trace_plot(traces, names)

#Summary statistic --> The Hamming distance from the starting state
def summary_statistic(trace, starting_state=initial):
    N = len(trace)
    res = np.ones(N)
    for i in range(N):
        dist = hamming(trace[i], starting_state)
        res[i] = dist
    return res

#Calculating the autocorrelation for each for a list of lags
def autocorr(x, lags):
    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]
    return np.array(corr)

#The autocorrelation plot
lags = np.arange(0, 500, 1)
hamm_rw = summary_statistic(trace_rw, initial)
hamm_g2 = summary_statistic(trace_g2, initial)
acf_rw = autocorr(hamm_rw, lags)
acf_g2 = autocorr(hamm_g2, lags)

plt.plot(lags, acf_rw, label="rw")
plt.plot(lags, acf_g2, label="g2")
plt.legend()
plt.show()

     