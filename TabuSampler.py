import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from BasicFunctions import *
from DatabaseClass import create_databases
from time import time

#The global parameters needed to define the functions
# if __name__ == "__main__":
#     lmbd = 10
#     l = 15 #The number of categories for each record
#     p_cat = np.array([0.05, 0.15, 0.4, 0.2, 0.2]) #The pval vector for each category
#     beta = 0.01
#     p_match, x, y, M_gen, M_reverse_gen = create_databases(lmbd, l, p_cat, beta)

#     #Other parameters that are needed
#     N_1 = len(x)
#     N_2 = len(y)

#     print("The golden truth p_match: ", p_match)
#     print("The first record of the x database:")
#     print(x[0, :])
#     print("The first record of the y database:")
#     print(y[0,:])
#     print("The golden truth M vector: ")
#     print(M)

#     #The balancing functions - used to calculate the jumping rates
#     g1 = lambda t: np.sqrt(t)
#     g2 = lambda t: t/(1+t)
#     id = lambda t: t

#     #All the considered generators
#     gens = np.zeros((N_1,N_2))
#     print("The total number of generators: ",N_1*N_2)

#The function that calculates the initial rates for the Tabu sampler
def rates_tabu(M, M_reverse, g, lmbd, p_cat, l,
               beta, p_match, x, y, N_1, N_2):
    ratios = np.zeros((N_1,N_2))
    for i in np.arange(0, N_1, 1):
       for j in np.arange(0, N_2, 1):
           move, i_accent, j_accent = move_type(M, M_reverse, i+1, j+1)
           ratios[i,j] = prob_ratio(i+1, j+1, move, lmbd, p_match, l, beta, p_cat, x, y, i_accent, j_accent)
    rates = g(ratios)

    #Flattening the rates before returning
    rates_return = rates.reshape(N_1*N_2,)

    return rates_return

#The function that updates the rates
def update_tabu(i, j, current_rates, M, M_reverse, g, N_1, N_2,
                lmbd, p_match, beta, p_cat, l, x, y,
                i_accent=0, j_accent=0):
    #Put the rates back into a matrix shape
    rates_matrix = current_rates.reshape(N_1, N_2)

    #Changing the rates in the ith row
    for k in np.arange(1, N_2+1):
        move, i_star, j_star = move_type(M, M_reverse, i, k)
        ratio = prob_ratio(i, k, move, lmbd, p_match, l, beta, p_cat, x, y, i_star, j_star)
        rates_matrix[i-1, k-1] = g(ratio)

    #changing the rates in the jth column
    for h in np.arange(1, N_1+1):
        move, i_star, j_star = move_type(M, M_reverse, h, j)
        ratio = prob_ratio(h, j, move, lmbd, p_match, l, beta, p_cat, x, y, i_star, j_star)
        rates_matrix[h-1, j-1] = g(ratio)

    #Changing the rates in the column of i_accent (if involved in the move)
    if i_accent != 0:
        for k in np.arange(1, N_2+1):
            move, i_star, j_star = move_type(M, M_reverse, i_accent, k)
            ratio = prob_ratio(i_accent, k, move, lmbd, p_match, l, beta, p_cat, x, y, i_star, j_star)
            rates_matrix[i_accent-1, k-1] = g(ratio)

    #Changing the rates in the row of j_accent (if involved in the move)
    if j_accent != 0:
        for h in np.arange(1, N_1+1):
            move, i_star, j_star = move_type(M, M_reverse, h, j_accent)
            ratio = prob_ratio(h, j_accent, move, lmbd, p_match, l, beta, p_cat, x, y, i_star, j_star)
            rates_matrix[h-1, j_accent-1] = g(ratio)

    #Reshaping the rates again before returning
    rates_return = rates_matrix.reshape(N_1*N_2,)
    return rates_return

# #Testing the two functions
# rates = rates_tabu(M, M_reverse, g2)
# print(rates)
# rates_updated = update_tabu(2, 3, rates, M, M_reverse, g2, N_1, N_2)
# print(rates_updated)

#Constructing the actual Tabu sampler itself
def tabu_sampler(N_1, N_2, num_gens, M_initial, M_reverse_initial, g,
                 target_time, M_truth, thin_rate, print_rate,
                 lmbd, p_match, l, p_cat,
                 beta, x, y):
    #Initialisation of the parameters
    process_time = 0
    num_iterations = 0
    num_flips = 0
    sample_index = 1
    N = int(target_time/thin_rate)
    current_M = M_initial
    current_M_reverse = M_reverse_initial

    #Placeholders for the results
    trace = np.ones((N, N_1))
    hamming_distances = np.zeros(N)
    energy = np.zeros(N)
    alpha = np.ones(num_gens)

    #Insert the initial values
    trace[0, :] = M_initial
    energy[0] = energy_function(M_initial, N_1, lmbd, p_match, beta, p_cat, l, x, y)
    hamming_distances[0] = N_1*hamming(M_initial, M_truth)

    #Calculate all the initial rates and the normalising constant
    rates = rates_tabu(M_initial, M_reverse_initial, g, lmbd, p_cat, l, beta, p_match, x, y, N_1, N_2)
    gamma_jump = (rates*alpha).sum()
    gamma_flip = (rates*(np.ones(num_gens) -alpha)).sum()

    start_time = time()
    while process_time < target_time or sample_index < N:
        
        #Adding the results into the placeholder arrays
        while process_time > sample_index*thin_rate:
            #Break if all the samples have been collected
            if sample_index >= int(target_time/thin_rate):
                break
            #Insert current M into the trace
            trace[sample_index, :] = current_M
            #Insert the target density energy
            energy[sample_index] = energy_function(current_M, N_1, lmbd, p_match, beta, p_cat, l, x, y)
            #Insert the Hamming distance
            hamming_distances[sample_index] = N_1*hamming(current_M, M_truth)
            
            sample_index += 1

        #Sample the time advance
        gamma = np.maximum(gamma_jump, gamma_flip)
        time_advance = np.random.exponential(size=1, scale=float(1/gamma))
        process_time += time_advance

        #The probability of taking a step
        move_probability = gamma_jump/gamma

        if np.random.rand(1) <= move_probability:
            #Pick the index of the generator
            generator_index = np.searchsorted((rates*alpha).cumsum(), np.random.rand(1)*gamma_jump)[0]

            #Taking the i,j pair that belongs to this index
            i = (generator_index // N_2) + 1 
            j = (generator_index % N_2) + 1

            #Removing this generator from the set of forward ones
            alpha[generator_index] = 0

            #Determining the type of move that this is
            move, i_accent, j_accent = move_type(current_M, current_M_reverse, i, j)

            i_accent = int(i_accent)
            j_accent = int(j_accent)
            
            #Perform the actual move
            current_M, current_M_reverse = apply_gen(i, j, current_M, current_M_reverse, move)
            
            #Updating the rates and the gammas
            rates = update_tabu(i, j, rates, current_M, current_M_reverse, g, N_1, N_2, lmbd, p_match, beta, p_cat, l, x, y, i_accent, j_accent)
            gamma_jump = (rates*alpha).sum()
            gamma_flip = (rates*(np.ones(num_gens)-alpha)).sum()
            
        else:
            #Change the possible generators that can be used
            num_flips += 1
            gamma_jump, gamma_flip = gamma_flip, gamma_jump
            alpha = np.ones(num_gens) - alpha
        num_iterations += 1

        if num_iterations % print_rate == 0:
            progressBar(process_time, target_time)
            
    runtime = time() - start_time
    print("Average excursion length: ", np.floor(num_iterations/(num_flips+1)))
    print("Runtime: ", np.round(runtime, 2))


    return trace, energy, hamming_distances, alpha, num_iterations, runtime

# #Testing the Tabu sampler here as well
# import numpy as np
# import numba
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import hamming
# from time import time

# #Import the helper functions and the Tabu sampling implementation
# from BasicFunctions import *

# #Autocorrelation function as implemented by Power and Goldman
# @numba.njit()
# def autocorr(x, lags):
#     mean=np.mean(x)
#     var=np.var(x)
#     xp=x-mean
#     corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]
#     return np.array(corr)

# #Generating the database
# from DatabaseClass import create_databases

# #Global parameters used
# lmbd = 10
# l = 15 #The number of categories for each record
# p_cat = np.array([0.05, 0.15, 0.4, 0.2, 0.2]) #The pval vector for each category
# beta = 0.01

# p_match, x, y, M_truth, M_reverse_truth = create_databases(lmbd, l, p_cat, beta)

# #Other parameters that are needed
# N_1 = len(x)
# N_2 = len(y)
# num_gens = N_1*N_2


# print("The golden truth p_match: ", p_match)
# print("The number of generators: ", num_gens)
# print("The first record of the x database:")
# print(x[0, :])
# print("The first record of the y database:")
# print(y[0,:])
# print("The golden truth M vector: ")
# print(M_truth)

# #Generate a random M and M_initial to start with
# M_initial, M_reverse_initial = starting_state(lmbd, p_match, N_1, N_2)

# print("The starting state for the simulations:")
# print(M_initial)

# #Defining the Barker balancing function
# g = lambda t: t/(1+t)

# # #Testing the rates_tabu function
# # test_rates = rates_tabu(M_truth, M_reverse_truth, g, lmbd, p_cat, l, beta, p_match, x, y, N_1, N_2)
# # print(test_rates)

# #Running the Tabu sampler
# target_time = 1000
# thin_rate = 0.05
# print_rate = 100
# N = int(target_time/thin_rate)

# trace, energy, hamming_distances, alpha, num_iterations, runtime = tabu_sampler(N_1, N_2,
#                                                                                 num_gens,
#                                                                                 M_initial, M_reverse_initial,
#                                                                                 g, target_time, M_truth,
#                                                                                 thin_rate, print_rate, lmbd, p_match,
#                                                                                 l, p_cat, beta, x, y)

