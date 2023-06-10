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

# %% Define Function Plotter function

def FunctionPlotter(generalDict,consts):
    plt.figure(figsize = (15,7))
    plt.plot(generalDict['x'],generalDict['piA'](generalDict['x'],consts)['Pi'],label = "$\Pi_a$")
    plt.plot(generalDict['x'],generalDict['piB'](generalDict['x'],consts)['Pi'],label = "$\Pi_b$")
    plt.title("Given Functions")
    plt.xlabel('x')
    plt.ylabel("$\Pi_{a~or~b}(x)$")
    plt.legend()
    plt.show()

# %% Define piA Newton Optimizer

def PiANewtonSolver(generalDict,newtonDict,consts,myNewton):
    
    for i in range(newtonDict['x0'].size): # Loop through all given x0 values
        # call Newton for all initial guesses for PiA
        solA, itsA, histA = myNewton(newtonDict['gradA'], newtonDict['hessA'], newtonDict['x0'][i], newtonDict['TOL'], newtonDict['maxit'])
    
    
        # Plot convergence
        plt.figure(figsize = (15,7))
        plt.plot(generalDict['x'],generalDict['piA'](generalDict['x'],consts)['Pi'],label = "$\Pi_a$")
        plt.plot(newtonDict['x0'][i],generalDict['piA'](newtonDict['x0'],consts)['Pi'][i],'r*',ms = 10,label = "$x_0$ =  %f" %newtonDict['x0'][i])
        plt.plot(histA,generalDict['piA'](histA,consts)['Pi'],'.-',label = "Newton Convergence",ms = 10)
        plt.plot(solA,generalDict['piA'](solA,consts)['Pi'],'b*',ms = 10,label = 'sol = %f' % solA)
        plt.title("Convergence of Newton's Method in %d Iterations" % itsA)
        plt.xlabel('$x$')
        plt.ylabel('$\Pi_a(x)$')
        plt.legend()
        plt.show()
        
# %% Define piB Newton Optimizer

def PiBNewtonSolver(generalDict,newtonDict,consts,myNewton):
    
    for i in range(newtonDict['x0'].size): # Loop through all given x0 values
    
        # call Newton for all initial guesses for PiB
        solB, itsB, histB = myNewton(newtonDict['gradB'], newtonDict['hessB'], newtonDict['x0'][i], newtonDict['TOL'], newtonDict['maxit'])
        
        
        # Plot convergence
        plt.figure(figsize = (15,7))
        plt.plot(generalDict['x'],generalDict['piB'](generalDict['x'],consts)['Pi'],label = "$\Pi_a$")
        plt.plot(newtonDict['x0'][i],generalDict['piB'](newtonDict['x0'],consts)['Pi'][i],'r*',ms = 10,label = "$x_0$ =  %f" %newtonDict['x0'][i])
        plt.plot(histB,generalDict['piB'](histB,consts)['Pi'],'.-',label = "Newton Convergence",ms = 10)
        plt.plot(solB,generalDict['piB'](solB,consts)['Pi'],'b*',ms = 10,label = 'sol = %f' % solB)
        plt.title("Convergence of Newton's Method in %d Iterations" % itsB)
        plt.xlabel('$x$')
        plt.ylabel('$\Pi_b(x)$')
        plt.legend()
        plt.show()
        
# %% Define piB GA Optimizer

def PiBGASolver(generalDict,geneticDict,consts,myGA):
    
    # Call genetic algorithm function ... 
    lamHist, Lam, bestLam, Pi, Min, PAve, Ave  = myGA(geneticDict['S'], geneticDict['G'], geneticDict['P'], geneticDict['SB'], generalDict['piB'],consts)


    plt.figure(figsize = (15,7))
    plt.plot(generalDict['x'],generalDict['piB'](generalDict['x'],consts)['Pi'],label = "$\Pi_b$")
    plt.plot(lamHist[0,0],generalDict['piB'](lamHist[0,0],consts)['Pi'],'r*',ms = 10,label = "$x_0$ =  %f" % lamHist[0,0])
    plt.plot(lamHist[:,0],generalDict['piB'](lamHist[:,0],consts)['Pi'],'.-',label = "GA Convergence",ms = 10)
    plt.plot(lamHist[-1,0],generalDict['piB'](lamHist[-1,0],consts)['Pi'],'b*',ms = 10,label = 'sol = %f' % Pi[0])
    plt.title("Convergence of Genetic Algorithm in %d Generations" % geneticDict['G'])
    plt.xlabel('$x$')
    plt.ylabel('$\Pi_b(x)$')
    plt.legend()
    plt.show()

    plt.figure(figsize = (15,7))
    plt.plot(range(geneticDict['G']),Min)
    plt.title("Evolution of Best Cost per Generation")
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.show()

    plt.figure(figsize = (15,7))
    plt.plot(range(geneticDict['G']),PAve)
    plt.title("Evolution of Average Parent Cost per Generation")
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.show()

    plt.figure(figsize = (15,7))
    plt.plot(range(geneticDict['G']),Ave)
    plt.title("Evolution of Average Cost per Generation")
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.show()
    
# %% Drone Utility Functions

def PlotInitialDroneSystem(droneConsts):
    
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(droneConsts['obs'][:,0],droneConsts['obs'][:,1],droneConsts['obs'][:,2],color = 'r',label = 'Obstacles')
    ax.scatter(droneConsts['tar'][:,0],droneConsts['tar'][:,1],droneConsts['tar'][:,2],color = 'g',label = 'Targets')
    ax.scatter(droneConsts['pos'][:,0],droneConsts['pos'][:,1],droneConsts['pos'][:,2],color = 'k',label = 'Agents')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.view_init(elev=20., azim=80)
    ax.set_title('Initial System Configuration')
    ax.legend()
    
def droneSim(Lam,consts):

    # Assigning simulation constants to associated variables
    Nt = consts['Nt'] # Number of targets
    Nm = consts['Nm'] # Number of drones
    No = consts['No'] # Number of obstacles
    dt = consts['dt'] # Time step size (sec)
    tf = consts['tf'] # Largest time limit (sec)
    w1 = consts['w1'] # Weight of mapping in net cost
    w2 = consts['w2'] # Weight of time usage in net cost
    w3 = consts['w3'] # Weight of agent losses in net cost
    pos = consts['pos'] # Initial Agent Positions (m)
    vel = consts['vel'] # Initial Agent Velocities (m/s)
    tar = consts['tar'] # Initial Target Positions (m)
    obs = consts['obs'] # Obstacle Positions (m)
    xmax = consts['xmax'] # x bound of domain
    ymax = consts['ymax'] # y bound of domain
    zmax = consts['zmax'] # z bound of domain
    agent_sight = consts['agent_sight'] # maximum target mapping distance (m)
    crash_range = consts['crash_range'] # agent collision distance (m)
    Ai = consts['Ai'] # agent characteristic area (m^2)
    Cdi = consts['Cdi'] # agent coefficient of drag
    mi = consts['mi'] # agent mass (kg)
    va = consts['va'] # Air velocity (m/s)
    ra = consts['ra'] # Air Density (kg/m^3)
    Fp = consts['Fp'] # Propolsion force magnitude (N)
    
    # Assigning design string to associated variables   
    Wmt = Lam[0] 
    Wmo = Lam[1] 
    Wmm = Lam[2] 

    wt1 = Lam[3]
    wt2 = Lam[4]
    
    wo1 = Lam[5]  
    wo2 = Lam[6]
    
    wm1 = Lam[7]  
    wm2 = Lam[8]
    
    a1 = Lam[9]
    a2 = Lam[10] 

    b1 = Lam[11]
    b2 = Lam[12]

    c1 = Lam[13]
    c2 = Lam[14]
    
    Nt0 = Nt # Saving initial number of targets for cost calculation
    Nm0 = Nm # Saving initial number of agents for cost calculation

    ts = int(np.ceil(tf/dt)) # Max Number of time steps

    c = 0 # counter for actual number of time steps

    # Initialize agent and target position arrays for plotting
    posTot = list()
    tarTot = list()

    posTot.append(deepcopy(np.array(pos))) # array to save all positions of agents at every time step
    tarTot.append(deepcopy(np.array(tar))) # array to save all positions of agents at every time step

    for i in range(ts): # Loop through all time
        
        if Nt <= 0 or Nm <= 0: # If all targets or agents have crashed, stop simulating
            break

        c = c + 1 # Keep track of time step

        # Initialize distance arrays between targets, agents, and obstacles
        dmt = np.array(np.zeros((3,Nm,Nt)))
        dmm = np.array(np.zeros((3,Nm,Nm)))
        dmo = np.array(np.zeros((3,Nm,No)))


        # Loop through agents
        for j in range(Nm):
                        
            dmt[:,j,:] = np.array((tar - pos[j,:]).T) # distance b/w agent j and all targets

            dmm[:,j,:] = np.array((pos - pos[j,:]).T) # distance b/w agent j and all other agents
            
            dmm[:,j,j] = float('inf') # Marker so we are not considering distance between agent and itself

            dmo[:,j,:] = np.array((obs - pos[j,:]).T) # distance b/w agent j and all obstacles


        # Calculate magnitude of distances between all objects
        magdmt = np.array(np.linalg.norm(dmt,2,0))
        magdmm = np.array(np.linalg.norm(dmm,2,0))
        magdmo = np.array(np.linalg.norm(dmo,2,0))

        # Determine which targets have been mapped
        tar_map = np.array(np.where(magdmt < agent_sight)) #produces 2 1-D arrays: 1st array:row, 2nd array:column

        # Determined which agents have crashed into one another
        mm_crash = np.array(np.where(magdmm < crash_range))

        # Determine which agents have crashed into obstacles
        mo_crash = np.array(np.where(magdmo < crash_range))

        # Determine which agents have moved outside domain in each dimension
        x_crash = np.array(np.where(np.abs(pos[:,0]) > xmax))
        y_crash = np.array(np.where(np.abs(pos[:,1]) > ymax))
        z_crash = np.array(np.where(np.abs(pos[:,2]) > zmax))

        # Combine all domain crashes together (unique since there could be overlap)
        dom_crash = np.unique(np.hstack((x_crash[0,:], y_crash[0,:], z_crash[0,:])))

        # Generate index arrays to determine which targets to remove and which agents to remove
        tarRem = np.array(np.unique(tar_map[1,:])) #access the column indices for tar_map --> target indices
        ageRem = np.unique(np.hstack([mm_crash[0,:], mo_crash[0,:], dom_crash])) #access the row indices for mm/mo --> agent/obstacle indices
        
        # use -inf as marker to distinguish between agent j-j marker and remove agent marker
        magdmm[magdmm == float('inf')] = float('-inf')
        dmm[dmm == float('inf')] = float('-inf')

        if (tarRem.size > 0 or ageRem.size > 0): # Only remove objects if targets mapped or agents crash

            # Determine new number of targets and new number of agents
            Nt = Nt - np.size(tarRem)
            Nm = Nm - np.size(ageRem)
            
            if Nt <= 0 or Nm <= 0: # If all targets or agents have crashed, stop simulating
                break
  
            if tarRem.size > 0: # If statement for target mapping
            
                # Use +inf as remove target marker
                magdmt[:,tarRem] = float('inf')
                dmt[:,:,tarRem] = float('inf')
                tar[tarRem,:] = float('inf')
                
            if ageRem.size > 0:
                
                # Use +inf as remove agent marker
                magdmt[ageRem,:] = float('inf')
                dmt[:,ageRem,:] = float('inf')
                magdmm[ageRem,:] = float('inf')
                magdmm[:,ageRem] = float('inf')
                dmm[:,ageRem,:] = float('inf')
                dmm[:,:,ageRem] = float('inf')
                magdmo[ageRem,:] = float('inf')
                dmo[:,ageRem,:] = float('inf')
                pos[ageRem,:] = float('inf')
                vel[ageRem,:] = float('inf')

            # Remove and reshape arrays to account for removal of all targets and agents
            magdmt = np.array(np.reshape(magdmt[magdmt != float('inf')],[Nm,Nt]))
            dmt = np.array(np.reshape(dmt[dmt != float('inf')],[3,Nm,Nt]))
            magdmo = np.array(np.reshape(magdmo[magdmo != float('inf')],[Nm,No]))
            dmo = np.array(np.reshape(dmo[dmo != float('inf')],[3,Nm,No]))
            magdmm = np.array(np.reshape(magdmm[magdmm != float('inf')],[Nm,Nm]))
            dmm = np.array(np.reshape(dmm[dmm != float('inf')],[3,Nm,Nm]))
            pos = np.array(np.reshape(pos[pos != float('inf')],[Nm,3]))
            vel = np.array(np.reshape(vel[vel != float('inf')],[Nm,3]))
            tar = np.array(np.reshape(tar[tar != float('inf')],[Nt,3]))

        tarTot.append(tar[:]) # save new target positions
        
        # Remove and reshape array for j-j agent interactions which we ignore 
        magdmm = np.array(np.reshape(magdmm[magdmm != float('-inf')],[Nm,Nm-1]))
        dmm = np.array(dmm[dmm != float('-inf')])
        dmm = np.reshape(dmm,[3,Nm,Nm-1])

        # Calculate unit normal vector between all objects
        nmt = dmt / magdmt[np.newaxis,:,:]
        nmm = dmm / magdmm[np.newaxis,:,:]
        nmo = dmo / magdmo[np.newaxis,:,:]

        
        # Calculate scaled direction vectors between objects
        nhatmt = (wt1*np.exp(-a1*magdmt) - wt2*np.exp(-a2*magdmt))
        nhatmt = nhatmt[np.newaxis,:,:]*nmt

        nhatmm = (wm1*np.exp(-c1*magdmm) - wm2*np.exp(-c2*magdmm))
        nhatmm = nhatmm[np.newaxis,:,:]*nmm

        nhatmo = (wo1*np.exp(-b1*magdmo) - wo2*np.exp(-b2*magdmo))
        nhatmo = nhatmo[np.newaxis,:,:]*nmo

        # Sum up all iteraction vectors for each agent
        Nmt = np.sum(nhatmt,2)
        Nmm = np.sum(nhatmm,2)
        Nmo = np.sum(nhatmo,2)

        # Calculate the total force vectors for each agent
        Ntot = (Wmt*Nmt.T + Wmm *Nmm.T + Wmo*Nmo.T)

        # Obtain magnitude of force vector
        nDum = np.linalg.norm(Ntot,2,1)

        # Normalize force vector for each agent
        nstar = Ntot / nDum[:,np.newaxis]

        # Calculate drag force on all agents
        Fd = 0.5*ra*Cdi*Ai*((va-vel).T*np.linalg.norm(va - vel,2,1)).T

        # Calculate the total force on all agents
        Ftot = Fp*nstar + Fd

        # Update the velocity for each agent using forward euler
        vel += dt*Ftot/mi

        # Update the position for each agent using forward euler
        pos += (vel*dt)

        # Save the position of each agent
        posTot.append(pos[:])
     
    # Calculate Mstar, Tstar, Lstar for cost calculation
    Mstar = (Nt/Nt0)
    Tstar = ((c*dt)/tf)
    Lstar = ((Nm0 - Nm)/Nm0)
      
    # Calculate the cost for this simulation
    Pi = w1*Mstar + w2*Tstar + w3*Lstar
    
    outputs = {
        'Pi': Pi, # Cost for given input design string
        'posTot': posTot, # time dependent agent positions for this design
        'tarTot': tarTot, # time dependent target positions for this design
        'c': c, # Number of time steps for this simulation to complete
        'Mstar': Mstar, # Fraction of targets remaining
        'Tstar': Tstar, # Fraction of total time used
        'Lstar': Lstar # Fraction of total agents remaining
    }
    
    return(outputs)


def PlotDroneGAStatistics(G,Min,PAve,Ave,lamHist,droneConsts):
    
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
    
    # Obtain cost parameter statistics for best performer for each generation
    MstarMin = np.zeros(G)
    TstarMin = np.zeros(G)
    LstarMin = np.zeros(G)
    
    for i in range(G):
        output = droneSim(lamHist[i,:],droneConsts)
        
        MstarMin[i] = output['Mstar']
        TstarMin[i] = output['Tstar']
        LstarMin[i] = output['Lstar']

    # Plot cost parameter statistics over all generations
    fig2 = plt.figure(figsize=(12,5))
    plt.plot(range(0,G),MstarMin,label = 'M*')
    plt.plot(range(0,G),TstarMin,label = 'T*')
    plt.plot(range(0,G),LstarMin,label = 'L*')
    plt.xlabel('Generation')
    plt.ylabel('Cost Parameter Value')
    plt.title('Best Cost Parameter Evolution')
    plt.legend(loc = 'lower right')
    plt.show()
    
    
def PlotDroneAnimation(bestLam,droneConsts):
    def drawframe(n): # Function to create plot
        dots1.set_xdata(obs[:,0])
        dots1.set_ydata(obs[:,1])
        dots1.set_3d_properties(obs[:,2])
        
        dots2.set_xdata(tarTot[n][:,0])
        dots2.set_ydata(tarTot[n][:,1])
        dots2.set_3d_properties(tarTot[n][:,2])
        
        dots3.set_xdata(posTot[n][:,0])
        dots3.set_ydata(posTot[n][:,1])
        dots3.set_3d_properties(posTot[n][:,2])

        Title.set_text('Solution Animation: Time = {0:4f}'.format(n*dt))
        return(dots1,dots2,dots3)

    output = droneSim(bestLam,droneConsts)
    
    obs = droneConsts['obs'] # Obstacle positions for all time steps
    tarTot = output['tarTot'] # Time dependent target positions
    posTot = output['posTot'] # Time dependent agent positions
    dt = droneConsts['dt'] # Time step size (sec)
    c = output['c'] # Total number of time steps


    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_xlim((-droneConsts['xmax'],droneConsts['xmax']))
    ax.set_ylim((-droneConsts['ymax'],droneConsts['ymax']))
    ax.set_zlim((-droneConsts['zmax'],droneConsts['zmax']))
    ax.view_init(elev=20., azim=80)

    Title = ax.set_title('')


    dots1, = ax.plot([],[],[],'r.',ms = 10,label = 'Obstacles')
    dots2, = ax.plot([],[],[],'g.',ms = 10,label = 'Targets')
    dots3, = ax.plot([],[],[],'k.',ms = 10,label = 'Agents')

    ax.legend()

    anim = animation.FuncAnimation(fig, drawframe, frames=c, interval=50, blit=True)

    rc('animation', html='jshtml',embed_limit = 35)

    return(anim)