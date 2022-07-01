import numpy as np
import sys
import copy as c
from math import isnan
from scipy.spatial.distance import hamming
from time import time
import random as rand
from DatabaseClass import create_databases
from BasicFunctions import *

if __name__ == "__main__":
    #The global parameters
    lmbd = 1E2
    l = 15 #The number of categories for each record
    p_cat = [0.05, 0.15, 0.4, 0.2, 0.2] #The pval vector for each category
    beta = 0.01
    N =1E2 #The number of samples

    #Creating the x and y databases for sampling
    p_match, x, y, M = create_databases(lmbd, l, p_cat, beta)

    #Actual values of the hyperparameters
    print("The ground-truth lambda value: ",lmbd)
    print("The ground-truth p_match: ",p_match)


    #Initial state of the matching (randomly generated)
    N_1, N_2 = len(x), len(y)
    M = np.zeros(N_1)
    M_reverse = np.zeros(N_2) #added for extra efficiency
    M, M_reverse = starting_state_fixed(lmbd, p_match, N_1, N_2)

    #The balancing functions - used to calculate the jumping rates
    g1 = lambda t: np.sqrt(t)
    g2 = lambda t: t/(1+t)
    id = lambda t: t

    #All the considered generators
    gens = []
    for i in range(1, N_1+1):
        for j in range(1, N_2+1):
            gens.append((i,j))
    gens = np.array(gens) #Convert to an array
    num_gens = len(gens)

if __name__ == "__main__":      
    print("The total number of generators: ", num_gens)

#The function that creates and samples a Zanella process
def Zanella_sampler_fixed(T, #The process time until sampling is terminated
                    x, #The first database
                    y, #The second database
                    N_1, #The number of entries in the x database
                    N_2, #The number of entries in the y database
                    M_init, #The initial matching position
                    M_reverse_init, #Keeping track for the update moves
                    l, #The number of fields of the record
                    beta, #The distortion probability
                    p_cat, #The probability of being in a certain category
                    g, #The balancing function
                    gens, #The considered generators
                    num_gens, #The number of considered generators
                    lmbd, #The true lambda
                    p_match, #The true p_match
                    N=1E12 #optional number of moves maximum
                    ):
    process_time, num_moves = 0, 0
    trace, time_trace = [M_init], [0]
    M, M_reverse = M_init, M_reverse_init
    
    #Start the time
    start_time = time()
    #Create the initial rates
    rates, gamma, ratios = initial_jumping(M, M_reverse, gens, num_gens, p_cat, g, x, y, l, beta, lmbd, p_match) #The initial rates etc.
    rates = rates.flatten()
    while process_time < T and num_moves < N:
        #The MH-part of the within Gibbs-sampler
        i_accent, j_accent = None, None
        gen_sampled = np.argmax(np.random.multinomial(1, rates)) #sample a generator
        time_advance = np.random.exponential(scale=1/gamma) #sample a time
        process_time += time_advance
        i, j = gens[gen_sampled]
        #Updating M and M_reverse
        if M[i-1] == 0 and M_reverse[j-1] == 0:
            M[i-1], M_reverse[j-1] = j, i #The update to the matching
        elif M[i-1] == j:
            M[i-1], M_reverse[j-1] = 0, 0
        elif M[i-1] == 0 and M_reverse[j-1] != 0:
            i_accent = int(M_reverse[j-1])
            M[i-1], M_reverse[j-1] = j, i #Add the move
            M[i_accent-1] = 0 #Delete the move - other one already altered
        elif M[i-1] != 0 and M_reverse[j-1] == 0:
            j_accent = int(M[i-1])
            M[i-1], M_reverse[j-1] = j, i
            M_reverse[j_accent-1] = 0
        else:
            i_accent, j_accent = int(M_reverse[j-1]), int(M[i-1])
            M[i-1], M[i_accent-1] = j, j_accent
            M_reverse[j-1], M_reverse[j_accent-1] = i, i_accent
        #Updating the rates for sampling
        rates, gamma, ratios = update_rates(rates, ratios, M, M_reverse, gamma, i, j, g, x, y, l, beta, N_1, N_2, p_cat, lmbd, p_match)
        #Also updating for the i_accent if needed
        if i_accent is not None:
            rates, gamma, ratios = update_rates(rates, ratios, M, M_reverse, gamma, i_accent, j, g, x, y, l, beta, N_1, N_2, p_cat, lmbd, p_match)
        #Same for the j_accent
        if j_accent is not None:
            rates, gamma, ratios = update_rates(rates, ratios, M, M_reverse, gamma, i, j_accent, g, x, y, l, beta, N_1, N_2, p_cat, lmbd, p_match)
        #Save the updates
        trace.append(c.copy(M))
        time_trace.append(c.copy(process_time))
        num_moves += 1
        progressBar(num_moves, N, process_time, T, bar_length=40) #progress for the sampler
    
    #The total runtime
    runtime = time() - start_time
    return trace, time_trace, num_moves, runtime
    


def Tabu_sampler_fixed(T, #The process time until sampling is terminated
                    x, #The first database
                    y, #The second database
                    N_1, #The number of entries in the x database
                    N_2, #The number of entries in the y database
                    M_init, #The initial matching position
                    M_reverse_init, #Keeping track for the update moves
                    l, #The number of fields of the record
                    beta, #The distortion probability
                    p_cat, #The probability of being in a certain category
                    g, #The balancing function
                    num_gens, #The number of generators
                    gens, #The considered generators
                    lmbd, #The true lambda
                    p_match, #The true p_match
                    N=1E12 #optional number of moves maximum
                    ):
    process_time, num_moves = 0, 0
    trace, time_trace = [M_init], []
    M, M_reverse = M_init, M_reverse_init
    reversion_trace, reversion = [], 0
    
    #Start the time
    start_time = time()
    
    #Sample the directions and the global direction
    directions, global_direction = initial_directions(num_gens)
    
    #Create the initial rates
    rates_forward, rates_back, gamma_forward, gamma_back, ratios_forward, ratios_back = initial_rates_Tabu(M, M_reverse, gens, p_cat, num_gens, g, x, y, l, beta, directions, lmbd, p_match)
    rates_forward = rates_forward.flatten()
    rates_back = rates_back.flatten()
    
    while num_moves < N:
        #The MH-part of the within Gibbs-sampler
        i_accent, j_accent = None, None
        gamma = max(gamma_forward, gamma_back)
        time_advance = np.random.exponential(scale=1/gamma)
        process_time += time_advance
        if global_direction == 1:
            prob_move = gamma_forward/gamma
        else:
            prob_move = gamma_back/gamma
        u = np.random.uniform()
        if u <= prob_move:
            if global_direction == 1:
                rates_forward = rates_forward/sum(rates_forward)
                gen_sampled = np.argmax(np.random.multinomial(1, rates_forward))
                directions[gen_sampled] = -1
                rates_forward, rates_back = gamma_forward*rates_forward, gamma_back*rates_back
                gamma_forward -= rates_forward[gen_sampled]
                rates_back[gen_sampled], rates_forward[gen_sampled] = rates_forward[gen_sampled], 0
                gamma_back += rates_back[gen_sampled]
                rates_forward, rates_back = rates_forward/gamma_forward, rates_back/gamma_back
            else:
                rates_back = rates_back/sum(rates_back)
                gen_sampled = np.argmax(np.random.multinomial(1, rates_back))
                directions[gen_sampled] = 1
                gamma_back -= (rates_back[gen_sampled]*gamma_back)
                rates_forward[gen_sampled], rates_back[gen_sampled] = rates_back[gen_sampled], 0
                gamma_forward += (rates_forward[gen_sampled]*gamma_forward)
            #Perform the move
            i, j = gens[gen_sampled]
            #Updating M and M_reverse
            if M[i-1] == 0 and M_reverse[j-1] == 0:
                M[i-1], M_reverse[j-1] = j, i #The update to the matching
            elif M[i-1] == j:
                M[i-1], M_reverse[j-1] = 0, 0
            elif M[i-1] == 0 and M_reverse[j-1] != 0:
                i_accent = int(M_reverse[j-1])
                M[i-1], M_reverse[j-1] = j, i #Add the move
                M[i_accent-1] = 0 #Delete the move - other one already altered
            elif M[i-1] != 0 and M_reverse[j-1] == 0:
                j_accent = int(M[i-1])
                M[i-1], M_reverse[j-1] = j, i
                M_reverse[j_accent-1] = 0
            else:
                i_accent, j_accent = int(M_reverse[j-1]), int(M[i-1])
                M[i-1], M[i_accent-1] = j, j_accent
                M_reverse[j-1], M_reverse[j_accent-1] = i, i_accent
            #Updating the rates for sampling
            rates_forward, rates_back, gamma_forward, gamma_back, ratios_forward, ratios_back = update_Tabu(rates_forward, rates_back, ratios_forward, ratios_back, M, M_reverse, gamma_forward, gamma_back, i, j, g, x, y, l, beta, N_1, N_2, p_cat, directions, lmbd, p_match)
            #Also updating for the i_accent if needed
            if i_accent is not None:
                rates_forward, rates_back, gamma_forward, gamma_back, ratios_forward, ratios_back = update_Tabu(rates_forward, rates_back, ratios_forward, ratios_back, M, M_reverse, gamma_forward, gamma_back, i_accent, j, g, x, y, l, beta, N_1, N_2, p_cat, directions, lmbd, p_match)
            #Same for the j_accent
            if j_accent is not None:
                rates_forward, rates_back, gamma_forward, gamma_back, ratios_forward, ratios_back = update_Tabu(rates_forward, rates_back, ratios_forward, ratios_back, M, M_reverse, gamma_forward, gamma_back, i, j_accent, g, x, y, l, beta, N_1, N_2, p_cat, directions, lmbd, p_match)
            #Save the updates
            rates_forward = rates_forward.flatten()
            rates_back = rates_back.flatten()
            trace.append(c.copy(M))
            time_trace.append(c.copy(process_time))
            num_moves += 1
            reversion += 1
            progressBar(num_moves, N, process_time, T, bar_length=40) #progress for the sampler
        else:
            global_direction = global_direction*(-1)
            reversion_trace.append(c.copy(reversion))
            reversion = 0
        print("The forward gamma = ",gamma_forward)
        print("The backwards ganma = ", gamma_back)
        if isnan(gamma_forward) or isnan(gamma_back):
            print("Code will be terminated")
            print(len(rates_forward[rates_forward != 0]))
            print(len(rates_back[rates_back != 0]))
            return trace, time_trace, num_moves, runtime, reversion_trace
    #The total runtime
    runtime = time() - start_time
    return trace, time_trace, num_moves, runtime, reversion_trace

#Testing the fixed Zanella sampler
# if __name__ == "__main__":
#     trace, time_trace, num_moves = Zanella_sampler_fixed(T, x, y, N_1, N_2, M, M_reverse, l, beta, p_cat, g2, gens, num_gens, lmbd, p_match, 1E3)
#     print("The total number of moves made: ",num_moves)
#     print("The trace of the process:")
#     print(trace)
#     print("The time trace of the process: ")
#     print(time_trace)

#Testing the fixed p_match and lambda Tabu sampler
# if __name__ == "__main__":
#     trace, time_trace, num_moves = Tabu_sampler_fixed(T, x, y, N_1, N_2, M, M_reverse, l, beta, p_cat, id, gens, lmbd, p_match, 1E3)
#     print("The total number of moves made: ",num_moves)
#     print("The trace of the process:")
#     print(trace)
#     print("The time trace of the process: ")
#     print(time_trace)

#Testing the random starting state
# M, M_reverse = starting_state_fixed(lmbd, p_match, N_1, N_2)
# print(len(M[M!=0]))
# print(lmbd*p_match)
        

