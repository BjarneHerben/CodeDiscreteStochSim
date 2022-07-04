# CodeDiscreteStochSim
All the code used for my Bachelor thesis: "Efficient stochastic simulation on discrete spaces".

The project was about the incorporation of local information about the target distribution into two different
MCMC sampling schemes. First, the incorporation of local information in the proposal kernel for the Metropolis-Hastings
algorithm implemented and studied. This is the approach proposed in a paper written by Giacomo Zanella (2017), 
titled: "Informed proposals for local MCMC in discrete spaces". The second approach: using the same class of balancing 
functions as introduced in the paper written by Zanella to calculate the rates for a Markov Jump process, this approach was
proposed in a paper written by Sam Power and Jacob Vorstrup Goldman (2019), 
titled: "Accelerated sampling on Discrete-Spaces with Non-Reversible Markov Processes". From the second paper the Zanella process
and the Tabu sampler were implemented.

All the samplers from the two studied papers were implemented for three examples: independent binary components (toy example),
weighted permutations (varying irregularity of the target distribution), and the Bayesian record linkage problem.

The respectively named maps contain all the python scripts (used to write the functions for the samplers, helper functions, etc.)
and the Jupyter notebook files used to generate the results. 

For the independent binary component and the weighted permutation example, the included files refer to the following:
"GelmanRubinDiagnostic.py": the Gelman-Rubin diagnostic and the sampling from the overdispersed starting distribution implemented
"...Help.py": all the basic functions: log energy of the target density, ratio pi(y)/pi(x), generating a random starting state, etc.
"...Plots.ipynb": the Jupyter notebook file used to generate the results as visible in the thesis
"....Sampler.py": all the considered samplers: random walk MH, pointwise informed proposal MH, Zanella process, and the Tabu sampler

The Bayesian Record Linkage directory files are a bit more cumbersome. Firstly, all the different samplers are implemented in seperate files.
The files that are different compared to the other two examples:
"DatabaseClass.py": generating the databases for which the possible matchings are considered
"BRLPlots.ipynb": file used to compare the effective sample size for different values of beta
"BRLTracePlots.ipynb": self-explanatory, used to obtain the trace plots
"BRLGelmanRubin.ipynb": used to obtain the Gelman-Rubin results as presented in the thesis
"BRLProbs.ipynb": used to compare the difference in estimated matching probabilities for two independent runs
"BalancingHeuristic.ipynb": a brief simulation of the choice of balancing function for the Bayesian record linkage problem
as presented in the paper by Power and Goldman (2019).
