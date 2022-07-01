import numpy as np
import sys
import numba
from time import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from DatabaseClass import create_databases
from BasicFunctions import *

#The random walk sampler
def rw_sampler(N, #The total number of accepted samples
               N_1, #Length of the x database
               N_2, #Length of the y database
               lmbd, #The lambda value
               p_match, #The matching probability 
               M_truth, #The actual matching
               M_initial, #The initial M vector
               M_reverse_initial, #The initial M_reverse vector
               p_cat, #The probability for each category
               l, #The number of fields for each record
               beta, #The distortion probability
               x, #The x database
               y, #The y database
               print_rate=100 #The rate at which to print the progressbar
               ):
    #Initialisation of the parameters
    num_accepted = 0
    sample_index = 1
    current_M = M_initial
    current_M_reverse = M_reverse_initial
    
    #The placeholders for the results
    trace = np.ones((N+1, N_1))
    energy = np.zeros(N+1)
    hamming_distances = np.zeros(N+1)
    
    #Insert the initial values
    trace[0,:] = M_initial
    energy[0] = energy_function(M_initial, N_1, lmbd, p_match, beta, p_cat, l, x, y)
    hamming_distances[0] = N_1*hamming(M_initial, M_truth)
    
    start_time = time()
    while sample_index <= N:
        #Sampling a random generator
        i = np.random.randint(1, N_1+1)
        j = np.random.randint(1, N_2+1)
        
        #The probability of accepting the proposed state
        move, i_accent, j_accent = move_type(current_M, current_M_reverse, i, j)
        i_accent, j_accent = int(i_accent), int(j_accent)
        target_ratio = prob_ratio(i, j, move, lmbd, p_match, l, beta, p_cat, x, y, i_accent, j_accent)
        prob_acc = np.minimum(1, target_ratio)
        
        #The acceptation-rejection step
        if np.random.rand(1) <= prob_acc:
            #Apply the generator to obtain the new state
            current_M, current_M_reverse = apply_gen(i, j, current_M, current_M_reverse, move, i_accent, j_accent)
            
            #Update the number of accepted proposals
            num_accepted += 1
            
        
        #Insert the M configuration into the trace
        trace[sample_index,:] = current_M
        #Insert the energy of the target density
        energy[sample_index] = energy_function(current_M, N_1, lmbd, p_match, beta, p_cat, l, x, y)
        #Insert the Hamming distance to the ground-truth M
        hamming_distances[sample_index] = N_1*hamming(current_M, M_truth)
        #Update the sample index
        sample_index += 1
        
        #Print the progressbar
        if sample_index % print_rate == 0:
            progressBar(sample_index-1, N)
    
    runtime = time() - start_time
    print("")
    print("Acceptance ratio: ", np.round(num_accepted/N, 4))
    print("Runtime: ", np.round(runtime, 2))
    
    return trace, energy, hamming_distances, num_accepted, runtime

#Code for testing the random walk sampler
# if __name__ == "__main__":
#     #The global parameters
#     lmbd = 1e2
#     l = 15 #The number of categories for each record
#     p_cat = [0.05, 0.15, 0.4, 0.2, 0.2] #The pval vector for each category
#     beta = 0.01
#     N = int(1e3) #The number of samples

#     #Creating the x and y databases for sampling
#     p_match, x, y, M_truth, M_reverse_truth = create_databases(lmbd, l, p_cat, beta)

#     #Initial state of the matching
#     N_1, N_2 = len(x), len(y)
    
#     #Generate an initial state
#     M_initial, M_reverse_initial = random_state(N_1, N_2)
    
# #Testing the rw_sampler
# trace, energy, hamming_distances, num_iterations, runtime = rw_sampler(N, N_1, N_2, lmbd, p_match, M_truth,
#                                                                        M_initial, M_reverse_initial, p_cat, l, beta, x,
#                                                                        y, print_rate=1)

# print("The number of iterations needed: ",num_iterations)




