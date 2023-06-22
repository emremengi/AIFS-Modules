###########################################################################################
# Written for AIFS.
# Utilities Functions for students to use
# Copyright 2023 Tarek Zohdi, Emre Mengi, Omar Betancourt, Payton Goodrich. All rights reserved. 
###########################################################################################

#%% Importing Packages

# Packages needed for optimization code
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
plt.rcParams.update({'font.size': 18})

# Additional packages needed for drone code
import math
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
from matplotlib import animation
from matplotlib import rc

# Additional packages needed for lidar code
from scipy.stats import randint
import pandas as pd
from IPython.display import display, HTML
    
# %% Solar Farm Utility Functions

################################## Solar Simulation Function ####################################################

def solarSim(Lam, Nr,     c, w1, w2, domLim, Ptot, Gm, Gp, Pmin, thetaR,R1max,R2max,R3max,p1min,p2min,
            p3min,sReg,hReg):

    ARay = (2*sReg)**2 # Area covered by solar rays

    Pr = Ptot*(ARay)/Nr*np.ones([Nr,1]) # Power per light ray (assuming uniform power distribution among rays)

    Ptol = Pmin*Pr #Power tolerance at which we stop considering a ray

    # Extracting design parameters from design vector
    thetaX = Lam[0]
    thetaY = Lam[1]
    thetaZ = Lam[2]
    
    ng = Lam[3]
    ns = Lam[4]
    
    R1 = Lam[5]
    R2 = Lam[6]
    R3 = Lam[7]
    
    p1 = Lam[8]
    p2 = Lam[9]
    p3 = Lam[10]
    
    h0 = -Lam[11]
        
    # Define rotation matrices based on angles of solar panel
    RotX = np.array([[1,0,0],[0, np.cos(-thetaX),np.sin(-thetaX)],[0,-np.sin(-thetaX),np.cos(-thetaX)]])
    RotY = np.array([[np.cos(-thetaY),0,np.sin(-thetaY)],[0, 1,0],[-np.sin(-thetaY),0,np.cos(-thetaY)]])
    RotZ = np.array([[np.cos(-thetaZ),np.sin(-thetaZ),0],[-np.sin(-thetaZ),np.cos(-thetaZ),0],[0,0,1]])

    ## Determine material properties of solar panel & ground ##

    nhat = ns # Calculate refractive index for panel
    nhatG = ng # Calculate refractive index for ground

    # Define solar region based on defined domain limit
    xMin = -sReg
    xMax = sReg
    yMin = -sReg
    yMax = sReg

    ## Initial Ray Positions
    pos = np.array([(xMax-xMin)*np.random.rand(Nr)+ xMin, (yMax-yMin)*np.random.rand(Nr)+ yMin,hReg*np.ones(Nr)]) # Initial Ray Positions
    pos = pos.T

    # Initial ray velocities based on incoming angle w.r.t. e1-axis
    vel = np.array([np.zeros(Nr), -c*np.sin(thetaR)*np.ones(Nr), -c*np.cos(thetaR)*np.ones(Nr)]) # Initial Ray Velocities
    vel = vel.T

    # Initialize position of moving and impacted rays
    posTot = list()
    gTot = list()
    sTot = list()

    dt = 0.05*(hReg/c) # Time step based on velocity and initial height of rays

    sAbs = 0 # Initialize power absorbed by solar panel
    gAbs = 0 # Initialize power absorbed by ground

    Active = np.array([True for i in range(Nr)]) # Logical array to indicate what rays are still in flight

    ts = 0 # Initialize number of time steps

    while np.any(Active) and ts < 300: # While any of the rays are still traveling

        ts = ts + 1 # Iterate Number of time steps

        pos[Active,:] += dt*vel[Active,:] # Update new posiions of rays
        
        # translate and rotate light rays domain to check envelope equation
        Rot = np.matmul(RotX,RotY)
        Rot = np.matmul(Rot,RotZ)
        Rot = np.hstack([Rot,[[0],[0],[0]]])
        Rot = np.vstack([Rot, [0,0,0,1]])
        Rpos = np.hstack([deepcopy(pos),np.ones([pos.shape[0],1])])
#         Rpos = deepcopy(pos)
        Rpos[:,2] += h0
        Rpos = Rpos.T
        
        Rpos = np.matmul(Rot,Rpos)
        Rpos = Rpos.T

        # Check envelope equation
        rayContour = ((np.abs(Rpos[:,0]))/R1)**p1 + ((np.abs(Rpos[:,1]))/R2)**p2 + ((np.abs(Rpos[:,2]))/R3)**p3
        
        inPanel = list(np.where(rayContour <= 1)) # Indices of rays that are located inside boundary of solar panel
        
        impGround = np.where(pos[:,2] <= 0) # Determine which rays have impacted the ground

        indAct = np.where(Active) # Determine indices of active rays

        # Indices of arrays that are within solar panel region and impacted panel
        solarRem = list(set(inPanel[0][:]) & set(indAct[0][:]))

        # Indices of arrays out of solar region and impacted ground
        groundRem = list(set(impGround[0][:]) & set(indAct[0][:])) 

        sRem = np.array(solarRem) # Indices of rays impacting solar panel
        gRem = np.array(groundRem)# Indices of rays impacting ground

        if sRem.size > 0: # If any ray has impacted the solar panel
                        
            # Calculate gradient of surface based on rotated but NOT translated frame
            gradF = np.array([(Rpos[solarRem,0]*p1*(np.abs(Rpos[solarRem,0])**(p1-2)))/(np.abs(R1)**p1), \
                               (Rpos[solarRem,1]*p2*(np.abs(Rpos[solarRem,1])**(p2-2)))/(np.abs(R2)**p2),
                               ((Rpos[solarRem,2])*p3*(np.abs((Rpos[solarRem,2]))**(p3-2)))/(np.abs(R3)**p3)])


            # Normalize gradient to get normal vector
            normal = -gradF/np.linalg.norm(gradF,2,0)

            # Calculate angle of incidence using velocity of solar rays and normal vector to panel surface
            thetai = np.array(np.arccos(np.sum(vel[solarRem,:]*normal.T,1)/(c)))
            thetai = np.reshape(thetai,[thetai.size,1])

            # Calculate reflectivity of impacted rays on solar panel
            
            term1 = np.abs(np.cos(thetai))
            term2 = (nhat**2 - np.sin(thetai)**2)**0.5
            
            Reflect = np.array(np.abs(np.sin(thetai)/nhat) <= 1.)* \
                np.array(0.5*(((nhat*term1 - term2)/(nhat*term1 + term2))**2 + \
                         ((term1 - term2)/(term1 + term2))**2)) + np.array(np.abs(np.sin(thetai)/nhat) > 1.)*1.


            vel[solarRem,:] -= 2*(c*normal.T*np.cos(thetai)) # Update velocity of reflected rays

            sAbs += np.sum(Pr[solarRem] - Reflect*Pr[solarRem],0) # Calculate power absored by solar panel

            Pr[solarRem] *= Reflect # Update power remaining in impacted rays



        if gRem.size > 0: # If any ray has impacted the ground

            normalG = np.array([[0,0,-1]]) # Define constant normal vector (assuming flat ground)

            # Calculate angle of incidence using velocity of solar rays and normal vector to ground
            thetai = np.array(np.arccos(np.sum(vel[groundRem,:]*normalG,1)/c))
            thetai = np.reshape(thetai,[thetai.size,1])
            
            # Calculate reflectivity of impacted rays on ground
            
            term1 = np.abs(np.cos(thetai))
            term2 = (nhatG**2 - np.sin(thetai)**2)**0.5
            
            gReflect = np.array(np.abs(np.sin(thetai)/nhatG) <= 1.)* \
                np.array(0.5*(((nhatG*term1 - term2)/(nhatG*term1 + term2))**2 + \
                         ((term1 - term2)/(term1 + term2))**2)) + np.array(np.abs(np.sin(thetai)/nhatG) > 1.)*1.
            
            vel[groundRem,:] -= 2*(c*np.cos(thetai)*normalG) # Update velocity of reflected rays

            gAbs += np.sum(Pr[groundRem] - gReflect*Pr[groundRem],0) # Calculate power absored by ground

            Pr[groundRem] *= gReflect # Update power remaining in impacted rays

        # Determine which rays have left the domain or have reduced below the threshold power level
        powRem = np.array(np.where(Pr < Ptol))
        domRem = np.array(np.where((np.abs(pos[:,0]) > domLim) | (np.abs(pos[:,1]) > domLim) | (np.abs(pos[:,2]) > 3*domLim)))
    
        if domRem.size > 0: # If rays have left the domain
            Active[domRem] = False # Set impacted rays to false

        if powRem.size > 0: # If rays have reduced below power tolerance
            Active[powRem] = False # Set impacted rays to false

        # Save positions of active and not active rays for plotting 
        posTot.append(pos[Active,:])

        if np.size(gTot) == 0:
            gTot.append(pos[list(set(impGround[0][:])),:])
        else:
            gTot.append(np.vstack([np.array(gTot[-1]),pos[list(set(impGround[0][:])),:]]))

        if np.size(sTot) == 0:
            sTot.append(pos[list(set(inPanel[0][:])),:])
        else:
            sTot.append(np.vstack([np.array(sTot[-1]),pos[list(set(inPanel[0][:])),:]]))


    # Calculate cost function
    alpha = (Ptot*(ARay) - sAbs)/(Ptot*(ARay)) # ratio of power lost by solar panel
    G = gAbs/Ptot # Ratio of power absorbed by the ground
    gamma = (G >= Gp)*np.abs(G-Gp) + (G <= Gm)*np.abs(G-Gm) # ground absorption parameter
    Pi = w1*alpha + w2*gamma #+ (ts >= 1000)*1000 # Calculate cost function
    
    

    return(Pi, posTot, gTot, sTot, ts, alpha, gamma)

################################# Genetic Algorithm Function #######################################################

def myGA(S,G,P,K,theta1SB, theta2SB, theta3SB, ngSB, nsSB, p1SB, p2SB, p3SB, R1SB, R2SB, R3SB, h0SB, 
         Nr,    c, sReg, hReg, numLam, w1, w2, domLim, Ptot, Gm, Gp, Pmin, thetaR,R1max,R2max,R3max,p1min,p2min,p3min):
    
    # Initialize all variables to be saved
    Min = np.zeros(G) # Minimum cost for each generation
    PAve = np.zeros(G) # Parent average for each generation
    Ave = np.zeros(G) # Total population average for each generation
    
    Pi = np.zeros(S) # All costs in an individual generation
    alpha = np.zeros(S) # All solar panel losses ratios in each generation
    gamma = np.zeros(S) # All ground absorption ratios in each generation
    
    alphaMin = np.zeros(G) #solar panel losses ratio associated with best cost for each generation
    gammaMin = np.zeros(G) # ground absorption ratio associated with best cost for each generation
    
    alphaPAve = np.zeros(G) # solar panel losses ratio for top parents for each generation
    gammaPAve = np.zeros(G) # Aground absorption ratio value for top parents for each generation
    
    alphaAve = np.zeros(G) # Agerage solar panel losses ratio for whole population for each generation
    gammaAve = np.zeros(G) # Average ground absorption ratio for whole population for each generation
    
    # Generate initial random population
    Lam = np.array(np.vstack((((theta1SB[1] - theta1SB[0])*np.random.rand(1,S) + theta1SB[0], \
                             (theta2SB[1] - theta2SB[0])*np.random.rand(1,S) + theta2SB[0], \
                             (theta3SB[1] - theta3SB[0])*np.random.rand(1,S) + theta3SB[0], \
                             (ngSB[1] - ngSB[0])*np.random.rand(1,S) + ngSB[0], \
                             (nsSB[1] - nsSB[0])*np.random.rand(1,S) + nsSB[0], \
                             (R1SB[1] - R1SB[0])*np.random.rand(1,S) + R1SB[0], \
                             (R2SB[1] - R2SB[0])*np.random.rand(1,S) + R2SB[0], \
                             (R3SB[1] - R3SB[0])*np.random.rand(1,S) + R3SB[0], \
                             (p1SB[1] - p1SB[0])*np.random.rand(1,S) + p1SB[0], \
                             (p2SB[1] - p2SB[0])*np.random.rand(1,S) + p2SB[0], \
                             (p3SB[1] - p3SB[0])*np.random.rand(1,S) + p3SB[0], \
                             (h0SB[1] - h0SB[0])*np.random.rand(1,S) + h0SB[0]))))
        
    # Initially, calculate cost for all strings. After, only calculate new strings since top P already calculated
    start = 0 
    
    for i in range(G): # Loop through generations
        
        # Calculate fitness of unknown design string costs
        for j in range(start,S): # Evaluate fitness of strings
            Pi[j], _, _, _, _, alpha[j], gamma[j] = solarSim(Lam[:,j], Nr,    c, w1, w2, domLim, Ptot, Gm, Gp, Pmin, thetaR,R1max,R2max,R3max,p1min,p2min, p3min,sReg,hReg)
            
        
        # Sort cost and design strings based on performance
        ind = np.argsort(Pi)
        Pi = np.sort(Pi)
        Lam = Lam[:,ind]
        alpha = alpha[ind]
        gamma = gamma[ind]
        
        # Generate offspring radnom parameters and indices for vectorized offspring calculation
        phi = np.random.rand(numLam,K)
        ind1 = range(0,K,2)
        ind2 = range(1,K,2)
             
        # Concatonate original parents children, and new random strings all together into new design string array
        Lam = np.hstack((Lam[:,0:P], phi[:,ind1]*Lam[:,ind1] + (1-phi[:,ind1])*Lam[:,ind2],
                      phi[:,ind2]*Lam[:,ind2] + (1-phi[:,ind2])*Lam[:,ind1]))
        
        newLam = np.array(np.vstack((((theta1SB[1] - theta1SB[0])*np.random.rand(1,S-P-K) + theta1SB[0], \
                             (theta2SB[1] - theta2SB[0])*np.random.rand(1,S-P-K) + theta2SB[0], \
                             (theta3SB[1] - theta3SB[0])*np.random.rand(1,S-P-K) + theta3SB[0], \
                             (ngSB[1] - ngSB[0])*np.random.rand(1,S-P-K) + ngSB[0], \
                             (nsSB[1] - nsSB[0])*np.random.rand(1,S-P-K) + nsSB[0], \
                             (R1SB[1] - R1SB[0])*np.random.rand(1,S-P-K) + R1SB[0], \
                             (R2SB[1] - R2SB[0])*np.random.rand(1,S-P-K) + R2SB[0], \
                             (R3SB[1] - R3SB[0])*np.random.rand(1,S-P-K) + R3SB[0], \
                             (p1SB[1] - p1SB[0])*np.random.rand(1,S-P-K) + p1SB[0], \
                             (p2SB[1] - p2SB[0])*np.random.rand(1,S-P-K) + p2SB[0], \
                             (p3SB[1] - p3SB[0])*np.random.rand(1,S-P-K) + p3SB[0], \
                             (h0SB[1] - h0SB[0])*np.random.rand(1,S-P-K) + h0SB[0]))))
        
        Lam = np.hstack((Lam, newLam))        
        
        # Save all requested values
        Min[i] = Pi[0]
        PAve[i] = np.mean(Pi[0:P])
        Ave[i] = np.mean(Pi)
        
        alphaMin[i] = alpha[0]
        gammaMin[i] = gamma[0]
        
        alphaPAve[i] = np.mean(alpha[0:P])
        gammaPAve[i] = np.mean(gamma[0:P])
        
        alphaAve[i] = np.mean(alpha)
        gammaAve[i] = np.mean(gamma)
        
        # Update start to P such that only new string cost values are calculated
        start = P
        
        # Print miminum value of cost for debugging (should monotonically decrease over generations)
        print(Min[i])
        
    return(Lam, Pi, Min, PAve, Ave, alphaMin, gammaMin, alphaPAve, gammaPAve, alphaAve, gammaAve)    

################################# GA Plotter Function ########################################

def PlotSolarGAStatistics(G, Min, PAve, Ave, alphaMin, gammaMin):
    
    # Plot cost evolution over all generations
    fig1 = plt.figure(figsize=(12,5))
    plt.semilogy(range(0,G),Min,label = 'Best Cost')
    plt.semilogy(range(0,G),PAve,label = 'Average Parents Cost')
    plt.semilogy(range(0,G),Ave,label = 'Average Cost')
    plt.xlabel('Generation')
    plt.ylabel('Min or Ave cost')
    plt.title('Cost Evolution')
    plt.legend()
    plt.show()
    

    # Plot cost parameter statistics over all generations
    fig2 = plt.figure(figsize=(12,5))
    plt.plot(range(0,G),alphaMin,label = r'$\alpha$')
    plt.plot(range(0,G),gammaMin,label = r'$\gamma$')
    plt.plot(range(0,G),Min,label = 'Total')
    plt.xlabel('Generation')
    plt.ylabel('Cost Parameter Value')
    plt.title('Best Cost Parameter Evolution')
    plt.legend(loc = 'lower right')
    plt.show()

################################# Solar Plotter Function ######################################## 

def PlotSolarAnimation(bestLam, Nr, c, w1, w2, domLim, Ptot, Gm, Gp, Pmin, thetaR,R1max,R2max,R3max,p1min,p2min,\
            p3min,sReg,hReg):
    def drawframe(n): # Function to create plot
        actRay.set_xdata(posTot[scale*n][:,0])
        actRay.set_ydata(posTot[scale*n][:,1])
        actRay.set_3d_properties(posTot[scale*n][:,2])
        
        gRay.set_xdata(gTot[scale*n][:,0])
        gRay.set_ydata(gTot[scale*n][:,1])
        gRay.set_3d_properties(gTot[scale*n][:,2])

        sRay.set_xdata(sTot[scale*n][:,0])
        sRay.set_ydata(sTot[scale*n][:,1])
        sRay.set_3d_properties(sTot[scale*n][:,2])

        Title.set_text('Solution Animation')
        return(actRay,gRay,sRay)

    _, posTot, gTot, sTot, ts, _, _ = solarSim(bestLam,Nr, c, w1, w2, domLim, Ptot, Gm, Gp, Pmin, thetaR,R1max,R2max,R3max,p1min,p2min,p3min,sReg,hReg)
    

    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim((-2*sReg,2*sReg))
    ax.set_ylim((-2*sReg,2*sReg))
    ax.set_zlim((0,hReg))
    ax.view_init(elev=40., azim=-40)

    Title = ax.set_title('')

    actRay, = ax.plot([],[],[],'r.',ms = 3)
    gRay, = ax.plot([],[],[],'g.',ms = 3)
    sRay, = ax.plot([],[],[],'b.',ms = 3)

    ax.legend()

    scale = 1 # Adjust this for the number of time steps to skip per frame during animation

    anim = animation.FuncAnimation(fig, drawframe, frames= int(ts/scale), interval=30, blit=True)

    rc('animation', html='jshtml',embed_limit = 35)

    return(anim)