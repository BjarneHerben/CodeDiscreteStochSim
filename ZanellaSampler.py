import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from BasicFunctions import *
from DatabaseClass import create_databases
from time import time

#Constructing the actual Tabu sampler itself
def zanella_sampler(N_1, N_2, num_gens, M_initial, M_reverse_initial, g,
                 target_time, M_truth, thin_rate, print_rate,
                 lmbd, p_match, l, p_cat,
                 beta, x, y):
    #Initialisation of the parameters
    process_time = 0
    num_iterations = 0
    sample_index = 1
    N = int(target_time/thin_rate)
    current_M = M_initial
    current_M_reverse = M_reverse_initial

    #Placeholders for the results
    trace = np.ones((N, N_1))
    hamming_distances = np.zeros(N)
    energy = np.zeros(N)

    #Insert the initial values
    trace[0, :] = M_initial
    energy[0] = energy_function(M_initial, N_1, lmbd, p_match, beta, p_cat, l, x, y)
    hamming_distances[0] = N_1*hamming(M_initial, M_truth)

    #Calculate all the initial rates and the normalising constant
    jump_rates = rates(M_initial, M_reverse_initial, g, lmbd, p_cat, l, beta, p_match, x, y, N_1, N_2)
    gamma = (jump_rates).sum()

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
        time_advance = np.random.exponential(size=1, scale=float(1/gamma))
        process_time += time_advance


        #Pick the index of the generator
        generator_index = np.searchsorted((jump_rates).cumsum(), np.random.rand(1)*gamma)[0]

        #Taking the i,j pair that belongs to this index
        i = (generator_index // N_2) + 1 
        j = (generator_index % N_2) + 1

        #Determining the type of move that this is
        move, i_accent, j_accent = move_type(current_M, current_M_reverse, i, j)

        i_accent = int(i_accent)
        j_accent = int(j_accent)
        
        #Perform the actual move
        current_M, current_M_reverse = apply_gen(i, j, current_M, current_M_reverse, move)
        
        #Updating the rates and the gammas
        jump_rates = update_rates(i, j, jump_rates, current_M, current_M_reverse, g, N_1, N_2, lmbd, p_match, beta, p_cat, l, x, y, i_accent, j_accent)
        gamma = (jump_rates).sum()
        
        num_iterations += 1

        if num_iterations % print_rate == 0:
            progressBar(process_time, target_time)
            
    runtime = time() - start_time
    print("Runtime: ", np.round(runtime, 2))

    return trace, energy, hamming_distances, num_iterations, runtime

# #Testing the Zanella sampler here as well
# import numpy as np
# import numba
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import hamming
# from time import time

# #Import the helper functions and the Tabu sampling implementation
# from BasicFunctions import *

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

# #Running the Zanellla sampler
# target_time = 10000
# thin_rate = 0.05
# print_rate = 1
# N = int(target_time/thin_rate)

# trace, energy, hamming_distances, num_iterations, runtime = zanella_sampler(N_1, N_2,
#                                                                                 num_gens,
#                                                                                 M_initial, M_reverse_initial,
#                                                                                 g, target_time, M_truth,
#                                                                                 thin_rate, print_rate, lmbd, p_match,
#                                                                                 l, p_cat, beta, x, y)