import numpy as np
import random as rand
import numba

#The global parameters
lmbd = 1E1
l = 15 #The number of categories for each record
p_match = 0.2
p_cat = np.array([0.05, 0.15, 0.4, 0.2, 0.2]) #The pval vector for each category
beta = 0.01
p_vals = [p_match, (1-p_match)/2, (1-p_match)/2]

#Sampling the number of entries
N = np.random.poisson(lam=lmbd, size=1)
struc = np.random.multinomial(n=N, pvals=p_vals)

#Function for sampling the singleton records
def single_sample(N, l, pvals):
    res = np.ones((N,l))
    for i in range(N):
        record = np.ones(l)
        for j in range(l):
            record[j] = np.argmax(np.random.multinomial(1, pvals))
        res[i,:] = record
    return res

#Function for sampling the mixed records
def mixed_sample(N, l, pvals, beta):
    resx, resy = np.ones((N,l)), np.ones((N,l))
    for i in range(N):
        recordx = np.ones(l)
        recordy = np.ones(l)
        for j in range(l):
            v = np.argmax(np.random.multinomial(1, pvals))
            u = np.random.uniform(size=2)
            for k in range(2):
                if u[k] <= beta:
                    if k == 0:
                        recordx[j] = np.argmax(np.random.multinomial(1, pvals))
                    else:
                        recordy[j] = np.argmax(np.random.multinomial(1, pvals))
                else:
                    if k == 0:
                        recordx[j] = v
                    else:
                        recordy[j] = v
        resx[i,:], resy[i,:] = recordx, recordy
    return resx, resy

# if __name__ == "__main__":
#     #Combining all the sampled database entries to obtain the final database
#     #Sampling the singletons for x and y
#     single_x = single_sample(struc[1])
#     single_y = single_sample(struc[2])

#     #Sampling the matched identities
#     matchx, matchy = mixed_sample(struc[0])

#     #Combining to obtain the resulting databases
#     x = np.concatenate((single_x, matchx), axis=0)
#     y = np.concatenate((single_y, matchy), axis=0)

#     #Shuffling the rows
#     np.random.shuffle(x)
#     np.random.shuffle(y)

#     #Saving the databases
#     # np.save('database_x.npy', x)
#     # np.save('database_y.npy', y)
    
#     print(x)

##Writing a function to create the database
def create_databases(lmbd, l, p_cat, beta):
    """
    Function that sample the two databases that are to be used based on the following
    parameters.
    Args:
        lmbd (integer): the lambda parameter for the Poisson total number of entities
        l (integer): the number of fields per record
        p_cat (list): the probability for being in a certain category
        beta (float): distortion probability
        
    """
    #Sampling p_match from the uniform prior
    p_match = float(np.random.uniform(size=1))
    

    p_vals = [p_match, (1-p_match)/2, (1-p_match)/2] #The array for the matching probabilities
    N = np.random.poisson(lam=lmbd, size=1) #Sampling the total number of entities
    struc = np.random.multinomial(n=N, pvals=p_vals) #The number of matchings, x_singletons, y_singletons
    
    #Sampling the singletons for x and y
    single_x = single_sample(struc[1], l, p_cat)
    single_y = single_sample(struc[2], l, p_cat)
    
    #Sampling the matched identities
    matchx, matchy = mixed_sample(struc[0], l, p_cat, beta)
    
    #Shuffling the indices
    index_x = np.arange(0, struc[0] + struc[1], 1)
    index_y = np.arange(0, struc[0] + struc[2], 1)
    np.random.shuffle(index_x)
    np.random.shuffle(index_y)
    
    M = np.zeros(struc[0] + struc[1])
    M_reverse = np.zeros(struc[0] + struc[2])
    
    #Where the matchings will take place
    for i in range(struc[0]):
        x_place = np.where(index_x == i)[0][0]
        y_place = np.where(index_y == i)[0][0]
        M[x_place] = y_place + 1
        M_reverse[y_place] = x_place + 1
    
    #Now shuffling the x and y databases according to the shuffled indices
    x_sorted = np.concatenate((matchx, single_x), axis=0)
    y_sorted = np.concatenate((matchy, single_y), axis=0)
    
    x, y = np.zeros(shape=(struc[0] + struc[1],15)), np.zeros(shape=(struc[0] + struc[2], 15))
    
    #Putting the x records in place
    for i in range(struc[0]+struc[1]):
        x[i,:] = x_sorted[index_x[i], :]
    
    #Also putting the y records in place
    for i in range(struc[0]+struc[2]):
        y[i,:] = y_sorted[index_y[i],:]
    return p_match, x, y, M, M_reverse

#Test run the function
if __name__ == "__main__":
    p_match, x, y, M, M_reverse = create_databases(lmbd, l, p_cat, beta)
    print("The matching probability (p_match) = ",p_match)
    print("The x database:")
    print(x)
    print("The y database:")
    print(y)
    print("The ground truth matching:")
    print(M)
    print("The (reversed) ground truth matching:")
    print(M_reverse)
    
    
    

    


        




        
