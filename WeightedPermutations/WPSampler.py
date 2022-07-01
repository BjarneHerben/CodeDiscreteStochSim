import numpy as np
import sys
import numba
from time import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from WPHelp import *

#Progressbar function copied from the implementation by Power and Goldman
def progressBar(num_moves, N, bar_length=40):
        percent = float(num_moves) / float(N)
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

def rw_sampler(dim, #Dimension of the state space
               N, #The number of sampled steps
               pvals, #The p values for each binary component
               starting_state, #The starting state
               baseline_state, #The baseline state from which the Hamming distance is taken
               print_rate=100
                ):
    #Initialisation of the parameters
    num_accepted = 0
    sample_index = 1
    current_state = starting_state
    
    #The placeholders for the results
    trace = np.ones((N+1, dim))
    energy = np.zeros(N+1)
    hamming_distances = np.zeros(N+1)
    
    #Insert the initial values
    trace[0,:] = starting_state
    energy[0] = energy_function(starting_state, pvals, dim)
    hamming_distances[0] = dim*hamming(starting_state, baseline_state)
    
    start_time = time()
    while sample_index <= N:
        #Sampling a random generator
        gen = (np.random.randint(0, dim), np.random.randint(0,dim))
        
        #The probability of accepting the proposed state
        target_ratio = prob_ratio(gen, current_state, pvals)
        prob_acc = np.minimum(1, target_ratio)
        
        #The acceptation-rejection step
        if np.random.rand(1) <= prob_acc:
            #Apply the generator to obtain the new state
            current_state = apply_gen(current_state, gen)
            
            #Update the number of accepted proposals
            num_accepted += 1
            
        
        #Insert the M configuration into the trace
        trace[sample_index,:] = current_state
        #Insert the energy of the target density
        energy[sample_index] = energy_function(current_state, pvals, dim)
        #Insert the Hamming distance to the ground-truth M
        hamming_distances[sample_index] = dim*hamming(current_state, baseline_state)
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

#The pointwise informed proposal sampler
def pw_sampler(dim, #Dimension of the state space
               N, #The number of sampled steps
               pvals, #The p values for each binary component
               g, #The balancing function
               starting_state, #The starting state
               baseline_state, #The baseline state from which the Hamming distance is taken
               print_rate=100
                ):
    #Initialisation of the parameters
    num_accepted = 0
    sample_index = 1
    current_state = starting_state
    
    #The placeholders for the results
    trace = np.ones((N+1, dim))
    energy = np.zeros(N+1)
    hamming_distances = np.zeros(N+1)
    
    #Insert the initial values
    trace[0,:] = starting_state
    energy[0] = energy_function(starting_state, pvals, dim)
    hamming_distances[0] = dim*hamming(starting_state, baseline_state)
    
    #Calculate the initial rates and the normalising constant
    rates = calculate_rates(current_state, pvals, dim, g)
    sum_rates = rates.sum()
    
    start_time = time()
    while sample_index <= N:
        #Sampling a proposed generator
        component = np.searchsorted(rates.cumsum(), np.random.rand(1)*rates.sum())[0]
        
        #Finding the respective generator
        i = (component // dim) 
        j = (component % dim) 
        
        gen = i,j
        
        #Applying the proposed state
        proposed_state = apply_gen(current_state, gen)
        
        #Updating the rates and the normalization constant
        proposed_rates = update_rates(proposed_state, rates, sum_rates, gen, pvals, g)
        proposed_sum_rates = proposed_rates.sum()
        
        #Calculating the acceptance probability
        ratio_states = prob_ratio(gen, current_state, pvals)
        ratio_proposed = np.log(ratio_states) + np.log(proposed_rates[component]) + np.log(sum_rates)
        ratio_proposed -= (np.log(rates[component]) + np.log(proposed_sum_rates))
        prob_acc = np.minimum(0, ratio_proposed)
        
        #The acceptation-rejection step
        if np.log(np.random.rand(1)) <= prob_acc:
            #Changing the rates and normalizing constant
            rates = proposed_rates
            sum_rates = proposed_sum_rates
            #Performing the accepted step
            current_state = proposed_state
            
            num_accepted += 1
            
        
        #Insert the M configuration into the trace
        trace[sample_index,:] = current_state
        #Insert the energy of the target density
        energy[sample_index] = energy_function(current_state, pvals, dim)
        #Insert the Hamming distance to the ground-truth M
        hamming_distances[sample_index] = dim*hamming(current_state, baseline_state)
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

#The Zanella sampler
def zanella_sampler(dim, #Dimension of the state space
                    target_time, #The process sampling time
                    thin_rate, #How often to record the state
                    pvals, #The p values for each binary component
                    g, #The balancing function
                    starting_state, #The starting state
                    baseline_state, #The baseline state from which the Hamming distance is taken
                    print_rate=100
                    ):
    #Initialisation of the parameters
    process_time = 0
    num_iterations = 0
    sample_index = 1
    N = int(target_time/thin_rate)
    current_state = starting_state

    #Placeholders for the results
    trace = np.ones((N, dim))
    hamming_distances = np.zeros(N)
    energy = np.zeros(N)

    #Insert the initial values
    trace[0, :] = starting_state
    energy[0] = energy_function(starting_state, pvals, dim)
    hamming_distances[0] = dim*hamming(starting_state, baseline_state)

    #Calculate all the initial rates and the normalising constant
    jump_rates = calculate_rates(current_state, pvals, dim, g)
    gamma = jump_rates.sum()

    start_time = time()
    while process_time < target_time or sample_index < N:
        
        #Adding the results into the placeholder arrays
        while process_time > sample_index*thin_rate:
            #Break if all the samples have been collected
            if sample_index >= int(target_time/thin_rate):
                break
            #Insert current M into the trace
            trace[sample_index, :] = current_state
            #Insert the target density energy
            energy[sample_index] = energy_function(current_state, pvals, dim)
            #Insert the Hamming distance
            hamming_distances[sample_index] = dim*hamming(current_state, baseline_state)
            
            sample_index += 1

        #Sample the time advance
        time_advance = np.random.exponential(size=1, scale=float(1/gamma))
        process_time += time_advance

        #Pick the index of the generator
        component = np.searchsorted((jump_rates).cumsum(), np.random.rand(1)*gamma)[0]
        
        #Finding the respective generator
        i = (component // dim) 
        j = (component % dim) 
        
        gen = i,j

        #Perform the actual move
        current_state = apply_gen(current_state, gen)
        
        #Updating the rates and the gammas
        jump_rates = update_rates(current_state, jump_rates, gamma, gen, pvals, g)
        gamma = (jump_rates).sum()
        
        num_iterations += 1

        if num_iterations % print_rate == 0:
            progressBar(process_time, target_time)
            
    runtime = time() - start_time
    print("Runtime: ", np.round(runtime, 2))

    return trace, energy, hamming_distances, num_iterations, runtime

#The Tabu sampler
def tabu_sampler(dim, #Dimension of the state space
                target_time, #The process sampling time
                thin_rate, #How often to record the state
                pvals, #The p values for each binary component
                g, #The balancing function
                starting_state, #The starting state
                baseline_state, #The baseline state from which the Hamming distance is taken
                print_rate=100):
    #Initialisation of the parameters
    process_time = 0
    num_iterations = 0
    num_flips = 0
    sample_index = 1
    N = int(target_time/thin_rate)
    current_state = starting_state

    #Placeholders for the results
    trace = np.ones((N, dim))
    hamming_distances = np.zeros(N)
    energy = np.zeros(N)
    alpha = np.ones(dim**2)

    #Insert the initial values
    trace[0, :] = starting_state
    energy[0] = energy_function(starting_state, pvals, dim)
    hamming_distances[0] = dim*hamming(starting_state, baseline_state)

    #Calculate all the initial rates and the normalising constant
    rates = calculate_rates(starting_state, pvals, dim, g)
    gamma_jump = (rates*alpha).sum()
    gamma_flip = (rates*(np.ones(dim**2) -alpha)).sum()

    start_time = time()
    while process_time < target_time or sample_index < N:
        
        #Adding the results into the placeholder arrays
        while process_time > sample_index*thin_rate:
            #Break if all the samples have been collected
            if sample_index >= int(target_time/thin_rate):
                break
            #Insert current M into the trace
            trace[sample_index, :] = current_state
            #Insert the target density energy
            energy[sample_index] = energy_function(current_state, pvals, dim)
            #Insert the Hamming distance
            hamming_distances[sample_index] = dim*hamming(current_state, baseline_state)
            
            sample_index += 1

        #Sample the time advance
        gamma = np.maximum(gamma_jump, gamma_flip)
        time_advance = np.random.exponential(size=1, scale=float(1/gamma))
        process_time += time_advance

        #The probability of taking a step
        move_probability = gamma_jump/gamma

        if np.random.rand(1) <= move_probability:
            #Pick the index of the generator
            component = np.searchsorted((rates*alpha).cumsum(), np.random.rand(1)*gamma_jump)[0]
            
            #Finding the respective generator
            i = (component // dim) 
            j = (component % dim) 
            
            gen = i,j

            #Removing this generator from the set of forward ones
            alpha[component] = 0

            #Perform the actual move
            current_state = apply_gen(current_state, gen)
            
            #Updating the rates and the gammas
            rates = update_rates(current_state, rates, gamma_jump, gen, pvals, g)
            gamma_jump = (rates*alpha).sum()
            gamma_flip = (rates*(np.ones(dim**2)-alpha)).sum()
            
        else:
            #Change the possible generators that can be used
            num_flips += 1
            gamma_jump, gamma_flip = gamma_flip, gamma_jump
            alpha = np.ones(dim**2) - alpha
        num_iterations += 1

        if num_iterations % print_rate == 0:
            progressBar(process_time, target_time)
            
    runtime = time() - start_time
    print("Average excursion length: ", np.floor(num_iterations/(num_flips+1)))
    print("Runtime: ", np.round(runtime, 2))


    return trace, energy, hamming_distances, alpha, num_iterations, runtime