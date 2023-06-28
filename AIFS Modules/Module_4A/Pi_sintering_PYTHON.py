############################################################################################
# Written for AIFS
# MODULE 4A - Genetic Algorithms - Process Simulation Functions
# Copyright 2023 Carla Becker, Tarek Zohdi. All rights reserved.
############################################################################################

import numpy as np
import copy
import time
import math
import os
import csv
import matplotlib.pyplot as plt
from matplotlib import rc
from numba import jit
from functools import partial

## Helper functions
# For converting between standard and voight notation FOR STRAINS
toptri = np.array([[0, 1, 1],\
                   [0, 0, 1],\
                   [0, 0, 0]]) # for obtaining upper right triangle of 3x3 matrix
mat69 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],\
                  [0, 0, 0, 0, 1, 0, 0, 0, 0],\
                  [0, 0, 0, 0, 0, 0, 0, 0, 1],\
                  [0, 1, 0, 0, 0, 0, 0, 0, 0],\
                  [0, 0, 0, 0, 0, 1, 0, 0, 0],\
                  [0, 0, 1, 0, 0, 0, 0, 0, 0]])
mat96 = np.array([[1, 0, 0, 0,   0,   0],\
                  [0, 0, 0, 1/2, 0,   0],\
                  [0, 0, 0, 0,   0,   1/2],\
                  [0, 0, 0, 1/2, 0,   0],\
                  [0, 1, 0, 0,   0,   0],\
                  [0, 0, 0, 0,   1/2, 0],\
                  [0, 0, 0, 0,   0,   1/2],\
                  [0, 0, 0, 0,   1/2, 0],\
                  [0, 0, 1, 0,   0,   0]])
half_voight = np.array([[1], [1], [1], [1/2], [1/2], [1/2]])

def voight(mat33):
    mat91 = np.reshape(np.multiply(mat33, 2*toptri+np.identity(3)), [9,1]) # element-wise multiplication
    mat61 = np.matmul(mat69, mat91)
    return mat61

def ivoight(mat61):
    mat91 = np.matmul(mat96, mat61)
    mat33 = mat91.reshape(3,3)
    return mat33

def frob_norm(mat33):
    hadamard = np.multiply(mat33, mat33)
    frob_norm = np.sqrt(hadamard.sum())
    return frob_norm

def get_dnew(p_sigd_dn_arr, A1, dt):

    p = p_sigd_dn_arr[0]
    sigmadn = p_sigd_dn_arr[1]
    dn = p_sigd_dn_arr[2]

    if -p<=0 and abs(p)>sigmadn and dn>0:
        ddotn = -A1*(abs(p)/sigmadn - 1)
    elif dn<0:
        dn = 0
        ddotn = 0
    else:
        ddotn = 0
    dnew = dn + ddotn*dt

    dnew = float(dnew)
    ddotn = float(ddotn)
    dnew_ddotn = np.array([dnew, ddotn])

    return dnew_ddotn 

def get_epdotn(devsig_sigy, A2):

    devsiglast = devsig_sigy[0:3,:]
    sigmayn = devsig_sigy[3,0]
    
    frobn = frob_norm(devsiglast)
    if frobn > sigmayn:
        lambda_dotn = A2*(frobn/sigmayn - 1)  # rate parameter for evolution of plastic strain, [unitless], 1x1
        epdotn = lambda_dotn*devsiglast/frobn # Eulerian plastic strain rate, [unitless], 3x3
    else:
        epdotn = np.zeros([3,3])
    return epdotn

def set_max_theta(thetanew_max_theta):
    thetanew = thetanew_max_theta[0]
    max_theta = thetanew_max_theta[1]
    if thetanew > max_theta:
        max_theta = thetanew
    return max_theta

## Sintering Cost Function
def Pi(Lambda):

    S = Lambda.shape[0]
    J3 = Lambda[:,0].reshape(S,1)
    D0 = Lambda[:,1].reshape(S,1)
    
    # Cost Function Parameters
    theta_des = 650 # desired peak temperature, [K], 1x1
    w1 = 1000       # weight for temperature term in cost function, [unitless], 1x1
    w2 = 100        # weight for densification term in cost function, [unitless], 1x1

    # Material variables (what someone would measure in the lab)
    rho0 = 2350     # material density in reference configuration, [kg/m3], 1x1
    kappa = 85e9    # dense material bulk modulus, [Pa], 1x1
    mu = 22e9       # dense material shear modulus, [Pa], 1x1
    beta = 2e-5     # coefficient of thermal expansion, [1/K], 1x1
    C = 850         # specific heat capacity, [J/K/kg], 1x1
    sigmac0 = 1e5   # dense material electrical conductivity, [S/m], 3x3
    sigmad0 = 1.2e6 # reference densification threshold, [Pa], 1x1
    sigmay0 = 38e6  # reference yield strength, [Pa], 1x1
    a = 0.8         # Joule heating absorption coefficient, [unitless], 1x1
    
    # Elastic Strain
    c1 = kappa + 4/3*mu # modulus for elasticty tensor, [Pa], 1x1
    c2 = kappa - 2/3*mu # modulus for elasticty tensor, [Pa], 1x1
    EE0 = np.array([[c1, c2, c2, 0,  0,  0],\
                    [c2, c1, c2, 0,  0,  0],\
                    [c2, c2, c1, 0,  0,  0],\
                    [0,  0,  0,  mu, 0,  0],\
                    [0,  0,  0,  0,  mu, 0],\
                    [0,  0,  0,  0,  0,  mu]]) # compliance/elasticity tensor, [Pa], 6x6
    
    # Sintering settings (what someone would set on equipment)  
    theta0 = 300 # initial temperature, [K], 1x1 
    divq0 = -30  # divergence of heat flux, [W/m2], 3x1
    
    # "Fudge" Factors
    A1 = 2.5e-4 # densification rate parameter, [unitless]
    A2 = 1.5    # plasticity rate parameter, [unitless]
    P1 = 1e-3   # elasticity thermal parameter, [unitless], 1x1
    P2 = 1e-3   # conductivity thermal parameter, [unitless], 1x1
    P3 = 1e-3   # densification thermal parameter, [unitless], 1x1
    P4 = 1e-3   # yield strength thermal parameter, [unitless], 1x1
    d0 = D0     # initial densification parameter, [unitless], 1x1
    
    # Kinematic variables
    dt = 1e-4               # time step, [s], 1x1
    T = 1.0                 # simulation time, [s], 1x1
    nt = int(T/dt)          # of time points, [unitless], 1x1
    t = np.linspace(0,T,nt) # time points, [s], 1xnt

    # Reset Values
    p = np.zeros([S,1])
    max_theta = np.zeros([S,1])
    dn = D0
    thetan = theta0*np.ones([S,1])
    Ep33n = np.zeros([S,3,3]) # Plastic strain, [unitless], Sx3x3
    sig33n = np.zeros([S,3,3]) # Cauchy stress, [Pa], Sx3x3
        
    ## Time Stepping the Governing Equation for Rate of Thermal Energy Storage
    for n in range(len(t)-1):
        
        # Read-in values from previous time step
        Eplast = Ep33n
        siglast = sig33n
        
        # Effects from Thermal Softening
        softn = np.exp(-(thetan - theta0)/theta0)                         # thermal softening parameter, [unitless], Sx1
        temp = np.multiply((1-dn), softn**P1)
        EEn = np.tensordot(np.multiply((1-dn), softn**P1), EE0, axes=0) # effective elasticity tensor, [Pa], Sx6x6
        EEn = EEn[:,0,:,:]                                                # reshape, get rid of redundant axis
        sigmacn = sigmac0*np.multiply((1-dn), softn**P2)                  # effective electrical conductivity, [S/m], Sx1
        sigmadn = sigmad0*softn**P3                                       # densification stress threshold, [Pa], Sx1
        sigmayn = sigmay0*softn**P4                                       # yield stress, [Pa], Sx1
        
        # Stress Analysis
        Bn = np.array([[0, 0, 0],\
                       [0, 0, 0],\
                       [0, 0, -0.15*(n+1)*dt]])           # du/dX as a function of time, [unitless], 3x3
        Fn = np.identity(3) + Bn                          # deformation gradient as a function of time, [unitless], 3x3
        Jacn = np.linalg.det(Fn)                          # Jacobian, [unitless], 1x1
        E33n = 1/2*(np.matmul(Fn.T, Fn) - np.identity(3)) # Green-Lagrange strain, [unitless], 3x3
        E61n = voight(E33n)                               # Green-Lagrange strain, [unitless], 6x1
    
        # Use previous time step to calculate plastic strain
        p = -1/3*np.array(list(map(np.trace, siglast)))         # hydrostatic pressure, [Pa], 1xS
        p = p.reshape(S,1)
        sphsiglast = np.tensordot(-p, np.identity(3), axes=0) # volumetric (or spherical) stress, [Pa], Sx3x3
        sphsiglast = sphsiglast[:,0,:,:]
        devsiglast = siglast - sphsiglast                       # deviatoric stress, [Pa], Sx3x3

        p_sigd_dn_arr = np.concatenate((p, sigmadn, dn), axis=1) # Sx3
        dnew_ddotn = np.array(list(map(partial(get_dnew, A1=A1, dt=dt), p_sigd_dn_arr)))
        dnew = dnew_ddotn[:,0]
        dnew = dnew.reshape(S,1)
        ddotn = dnew_ddotn[:,1]
        ddotn = ddotn.reshape(S,1)

        # Plastic Strain
        sigmay_33n = np.tensordot(sigmayn, np.identity(3), axes=0) # Sx3x3
        sigmay_33n = sigmay_33n[:,0,:,:]
        devsig_sigy = np.concatenate((devsiglast, sigmay_33n), axis=1)
        epdotn = np.array(list(map(partial(get_epdotn, A2=A2), devsig_sigy))) # Sx3x3

        Epdotn = Jacn * np.matmul(np.linalg.inv(Fn), np.matmul(epdotn, np.linalg.inv(Fn.T))) # Lagrangian plastic strain rate, [unitless], Sx3x3
        Ep33n = Eplast + Epdotn*dt                                                           # Lagrangian plastic strain, [unitless], Sx3x3
        Ep61n = np.array(list(map(voight, Ep33n)))

        # Thermal Strain
        B_dtheta = beta* (thetan - theta0)
        Et33n = np.tensordot(B_dtheta, np.identity(3), axes=0) # thermal strain, [unitless], Sx3x3
        Et33n = Et33n[:,0,:,:]
        Et61n = np.array(list(map(voight, Et33n)))
        
        # Second-Piola Kirchoff stress
        # Combine the Green-Lagrange, thermal, and plastic Strains to find the elastic strain
        Ee61n = E61n - Ep61n - Et61n # elastic strain, [unitless], Sx6x1
        S61n = np.matmul(EEn, Ee61n) # Second-Piola Kirchoff stress, [Pa], Sx6x1
        S33n = np.array(list(map(ivoight, S61n)))
        
        # Calculate the Cauchy stress and hydrostatic pressure  for the current time step
        sig33n = 1/Jacn * np.matmul(Fn, np.matmul(S33n, Fn.T)) # Cauchy stress, [Pa], 3x3
        
        # Electrical analysis
        Hn = a/sigmacn * J3 * J3 # Joule heating power, [W/m3], 1x1
        
        # Governing Equation
        half_Ee61 = np.multiply(Ee61n, half_voight)                         # adjust for extra factors of 2 in Voight multiplication, Sx6x1
        W = 1/2 * np.multiply(half_Ee61, np.matmul(EE0, Ee61n)).sum(axis=1) # stored elastic energy, Sx1
        B_trS = beta * np.array(list(map(np.trace, S33n)))                  # Sx1
        B_trS = B_trS.reshape(S,1)

        thetadotn_num = np.reshape(np.multiply(S33n, Epdotn).sum(axis=(1,2)), [S,1]) + np.multiply(ddotn, softn**P1)*W - divq0 + Jacn*Hn # Sx1
        thetadotn_den = rho0*C - B_trS - theta0*P1*np.multiply((1-dn), softn**P1)*W # Sx1
        thetadotn = np.divide(thetadotn_num, thetadotn_den)
        thetanew = thetan + thetadotn*dt # Sx1
        
        # Update deliverable values and measures of error
        dn = dnew
        thetan = thetanew

        thetanew_max_theta = np.concatenate((thetanew, max_theta), axis=1)
        max_theta = np.array(list(map(set_max_theta, thetanew_max_theta)))
        max_theta = max_theta.reshape(S,1)
            
    # Cost function calculation
    cost = w1*((max_theta - theta_des)/theta_des)**2 + w2*dn

    return cost
