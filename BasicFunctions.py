import numpy as np
import sys
import numba
import matplotlib.pyplot
from DatabaseClass import create_databases


#Remark: the parameters that are fixed are implemented as having a default
#This includes the following parameters:
#lmbd, p_match, l, beta, p_cat, x, y

#Some global parameters
# lmbd = 40
# l = 15 #The number of categories for each record
# p_cat = np.array([0.05, 0.15, 0.4, 0.2, 0.2]) #The pval vector for each category
# beta = 0.01
# p_match, x, y, M, M_reverse = create_databases(lmbd, l, p_cat, beta)
# N_1 = len(x)
# N_2 = len(y)

# #Printing some database information
# if __name__ == "__main__":
#     print("The golden truth p_match: ", p_match)
#     print("The first record of the x database:")
#     print(x[0, :])
#     print("The first record of the y database:")
#     print(y[0,:])
#     print("The golden truth M vector: ")
#     print(M)

#Function that calculates the log energy of the target density
def energy_function(M, N_1, lmbd, p_match, beta, p_cat, l, x, y):
    energy_target = 0
    for i in np.arange(0,N_1):
        if M[i] > 0:
            energy_target += np.log(4*p_match) - np.log(lmbd*(1-p_match)**2)
            for s in np.arange(0,l):
                energy_target += np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[i,s])])*(x[i,s] == y[int(M[i]-1),s]))
    return energy_target


#The function that determines the type of move that is being performed
def move_type(M, M_reverse, i, j):
    i_accent, j_accent = 0, 0
    if M[i-1] == 0 and M_reverse[j-1] == 0:
            move = 1
    elif M[i-1] == j:
            move = 2
    elif M[i-1] == 0 and M_reverse[j-1] != 0:
            move = 3
            i_accent = M_reverse[j-1]
    elif M[i-1] != 0 and M_reverse[j-1] == 0:
            move = 4
            j_accent = M[i-1]
    else:
            move = 5
            i_accent, j_accent = M_reverse[j-1], M[i-1]
    return move, i_accent, j_accent


#The function that returns the probability ratio of the target density
def prob_ratio(i, j, move, lmbd, p_match, l, beta, p_cat, x, y, i_accent=0, j_accent=0):
    ans = 0
    if move == 1:
        ans += np.log(4*p_match) - np.log(lmbd)-2*np.log(1-p_match)
        for s in np.arange(0,l):
            ans += np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[i-1,s])])*(x[i-1,s] == y[j-1,s]))
    elif move == 2:
        ans -= np.log(4*p_match) - np.log(lmbd)-2*np.log(1-p_match)
        for s in np.arange(0,l):
            ans -= np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[i-1,s])])*(x[i-1,s] == y[j-1,s]))
    elif move == 3:
        for s in np.arange(0,l):
            ans += np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[i-1,s])])*(x[i-1,s] == y[j-1,s]))
            ans -= np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[int(i_accent)-1,s])])*(x[int(i_accent)-1,s] == y[j-1,s]))
    elif move == 4:
        for s in np.arange(0,l):
            ans += np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[i-1,s])])*(x[i-1,s] == y[j-1,s]))
            ans -= np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[i-1,s])])*(x[i-1,s] == y[int(j_accent)-1,s]))
    else:
        for s in np.arange(0,l):
            ans += np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[i-1,s])])*(x[i-1,s] == y[j-1,s]))
            ans += np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[int(i_accent)-1,s])])*(x[int(i_accent)-1,s] == y[int(j_accent)-1,s]))
            ans -= np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[int(i_accent)-1,s])])*(x[int(i_accent)-1,s] == y[j-1,s]))
            ans -= np.log(beta*(2-beta) + (1-beta)**2/(p_cat[int(x[i-1,s])])*(x[i-1,s] == y[int(j_accent)-1,s]))
    return np.exp(ans)

#The function that calculates the locally-balanced rates for the samplers
def rates(M, M_reverse, g, lmbd, p_cat, l,
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

#The function that updates the locally-balanced rates for the samplers
def update_rates(i, j, current_rates, M, M_reverse, g, N_1, N_2,
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

#Generate the random starting state (for fixed lambda and p_match)
def starting_state(lmbd, p_match, N_1, N_2):
    num_matched = lmbd*p_match 
    M, M_reverse = np.zeros(N_1), np.zeros(N_2)
    for i in np.arange(0, num_matched):
        record = np.random.randint(1, N_2+1)
        index = np.random.randint(0, N_1)
        if M[index] == 0 and M_reverse[record-1] == 0:
            M[index] = record
            M_reverse[record-1] = index + 1
    return M, M_reverse

#Function that generates an entirely random state (M vector)
@numba.njit()
def random_state(N_1, N_2):
    num_matched = np.random.randint(low=0, high=min(N_1, N_2)+1)
    M, M_reverse = np.zeros(N_1), np.zeros(N_2)
    for i in np.arange(0, num_matched):
        record = np.random.randint(1, N_2+1)
        index = np.random.randint(0, N_1)
        if M[index] == 0 and M_reverse[record-1] == 0:
            M[index] = record
            M_reverse[record-1] = index + 1
    return M, M_reverse

#The function that for a function returns the array with the jumping ratios
def jumping_heuristic(N, N_1, N_2, M_samples, M_reverse_samples, g, lmbd, p_match, p_cat, l, beta, x, y):
    #The placeholde array
    values = np.zeros(N)
    
    for i in np.arange(0, N):
        M, M_reverse = M_samples[i,:], M_reverse_samples[i,:]
        rates = rates(M, M_reverse, g, lmbd, p_cat, l, beta, p_match, x, y, N_1, N_2)
        values[i] = rates.sum()
    
    return np.log(values)
    
#Calculating the autocorrelation for each for a list of lags
@numba.njit()
def autocorr(x, lags):
    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]
    return abs(np.array(corr))

#The function that applies the generator
def apply_gen(i, j, M, M_reverse, move, i_accent=0, j_accent=0):
    if move == 1:
        M[i-1] = j
        M_reverse[j-1] = i
    elif move == 2:
        M[i-1] = 0
        M_reverse[j-1] = 0
    elif move == 3:
        M[i-1] = j
        M_reverse[j-1] = i
        M[i_accent-1] = 0
    elif move == 4:
        M[i-1] = j
        M_reverse[j-1] = i
        M_reverse[j_accent-1] = 0
    else:
        M[i-1] = j
        M_reverse[j-1] = i
        M[i_accent-1] = j_accent
        M_reverse[j_accent-1] = 1
    return M, M_reverse

#Progressbar function copied from the implementation by Power and Goldman
def progressBar(num_moves, N, bar_length=40):
        percent = float(num_moves) / float(N)
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()
        


    
               
