############################################################################################
# Written for AIFS
# MODULE 4A - Genetic Algorithms - Process Optimization
# Copyright 2023 Carla Becker. All rights reserved.
############################################################################################

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import rc
from Pi_sintering_PYTHON import Pi, voight, ivoight, frob_norm

## ## Process Optimization ## ##

## Definitions
# Pi(Lambda)          anonymous function for evaluating fitness
#                          Pi.m should be in the working directory
# K,          1 x 1,  number of design strings to preserve and breed
# TOL,        1 x 1,  cost function threshold to stop evolution
# G,          1 x 1,  maximum number of generations
# S,          1 x 1,  total number of design strings per generation
# dv,         1 x 1,  number of design variables per string
# PI,         G x S,  cost of sth design in the gth generation
# Orig,       G x S,  indices of sorted strings before sorting
#                     e.g. Orig(10, 1) = 34 means that the 1st ranked 
#                          string in generation 10 was in position 34, 
#                          visualize using familyTree.m
# Lambda,     dv x S, array of most recent design strings
# g,          1 x 1,  generation counter
# PI_best,    1 x g,  minimum cost across strings and generations
# PI_avg,     1 x g,  average cost across strings and generations
# PI_par_avg, 1 x g,  average cost across strings and generations

def sort(pi):
    new_pi = np.sort(pi, axis=0)
    ind = np.argsort(pi, axis=0)
    return [new_pi, ind]

def reorder(Lambda, ind):
    temp = np.zeros((S,dv))
    for i in range(0, len(ind)):
        temp[i,:] = Lambda[int(ind[i]),:]
    Lambda = temp
    return Lambda

## Givens
P = 6
K = 6
TOL = 1e-6
G = 3
S = 20
dv = 2

J3min = 0
J3max = 1e7
D0min = 0.5
D0max = 0.9

# For construction of random genetic strings
scale_factor = np.array([J3max - J3min, D0max - D0min]) # dv x 1
offset = np.array([J3min, D0min])                       # dv x 1

# Initialize
PI = np.ones((G, S))
Orig = np.ones((G, S))
Lambda = np.random.rand(S, dv)*scale_factor + offset

## First generation
g = 0
cost = Pi(Lambda)

[new_cost, ind] = sort(cost)     # order in terms of decreasing cost    
PI[g, :] = new_cost.reshape(1,S) # log the initial population costs NEED TO RESHAPE?????
Orig[g,:] = ind.reshape(1,S)     # log the indices before sorting
Lambda = reorder(Lambda, ind)    # order in terms of decreasing cost

# Store values for performance tracking
PI_best = 1e10*np.ones(G)
PI_avg = 1e10*np.ones(G)
PI_par_avg = 1e10*np.ones(G)
top_performers = Lambda[1:4,:]
top_costs = new_cost[1:4]

# Update performance trackers
PI_best[0] = np.min(new_cost)
PI_avg[1] = np.mean(new_cost)
MIN = np.min(new_cost)   

## All later generations
while (MIN > TOL) and (g < G):

    print('g=' + str(g))
        
    # Mating 
    parents = Lambda[0:P,:]
    kids = np.zeros((K, dv))
    
    for p in list(range(0,P,2)): # p = 0, 2, 4, 6,...      
        if P % 2:
            print('P is odd. Choose an even number of parents.')
            break
        phi1 = np.random.rand()
        phi2 = np.random.rand()
        kids[p,:]   = phi1 * parents[p,:] + (1 - phi1) * parents[p+1,:]
        kids[p+1,:] = phi2 * parents[p,:] + (1 - phi2) * parents[p+1,:]
        
    # Update Lambda (with parents)
    new_strings = np.random.rand(S-K-P, dv)*scale_factor + offset
    Lambda = np.vstack((parents, kids, new_strings)) # concatenate vertically
    
    # Evaluate fitness of new population
    cost = Pi(Lambda)
 
    # Evaluate fitness of parent population
    par_cost = cost[0:P]
        
    [new_cost, ind] = sort(cost)     
    PI[g, :] = new_cost.reshape(1,S)        
    Orig[g,:] = ind.reshape(1,S) 
    Lambda = reorder(Lambda, ind)    # order in terms of decreasing cost
    
    # Update performance trackers
    PI_best[g] = np.min(new_cost)
    PI_avg[g] = np.mean(new_cost)
    PI_par_avg[g] = np.mean(par_cost)
    
    if np.min(new_cost) < MIN:
        MIN = np.min(new_cost)

    g = g + 1

top_costs = new_cost[0:4]
top_performers = Lambda[0:4,:]

## ## DELIVERABLES ## ##
  
# Problem 1.1: Plot the cost function over time, demonstrate convergence
# Hint: Use a loglog or semilog plot

fig1, ax1 = plt.subplots()
ax1.semilogy(np.arange(0,g), PI_best[0:g])
ax1.semilogy(np.arange(0,g), PI_avg[0:g])
ax1.semilogy(np.arange(0,g), PI_par_avg[0:g])
plt.xlabel('Generations',  fontsize=20)
plt.ylabel('Cost', fontsize=20)
plt.title('Convergence of Cost Function', fontsize=20)
plt.legend(['Best', 'Overall Mean', 'Parent Mean'], fontsize=15)
# plt.savefig('2a_PYTHON.png')

# Problem 1.2: Report your 4 best-performing designs
print('Top Performing Strings')
print('S1')
print(top_performers[0,0])
print(top_performers[0,1])
print('S2')
print(top_performers[1,0])
print(top_performers[1,1])
print('S3')
print(top_performers[2,0])
print(top_performers[2,1])
print('S4')
print(top_performers[3,0])
print(top_performers[3,1])

print('Costs of Top Performing Strings')
print('Pi1 | Pi2 | Pi3 | Pi4')
print(top_costs)

# Problem 1.3: Plots of the densification parameter and temperature versus time for the best-performing design
# Comment: How well does the design achieve the desired system behavior?
# Please see the plotting code in sintering_PYTHON.py file.