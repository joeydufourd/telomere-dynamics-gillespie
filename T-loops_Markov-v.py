import numpy
import math
import matplotlib.pyplot as plt
import time
#Species = [['LB',X[0]],['LR',X[1]],['T',X[2]]]
#math.log is a natural logarithm (ln)
#Parameters
X = [46, 46, 0]  #Knowing Species X[0] sets parameter as 1 to LB

# Total telomeres wanted
Ntot = sum(X)

numspecies = 3
numreactions = 5
numsim=3
t = 0
tmax = 3600  #A day 86400 an hour 3600

# Stoichiometry matrix: [ΔLB, ΔLR, ΔT]
smatrix = [
    [-1, 1, 0],  #k LB->LR
    [1, -1, 0],  #k LR-> LB
    [0, -1, 1],  #k LR-> T
    [1, 0, -1],  #k T ->LB
    [0, 1, -1]  #k T->LR
]

#reaction rates
k0 = 0.1   # LB -> LR (drains LB slowly)
k1 = 0.00001  # LR -> LB
k2 = 1.0   # LR -> T
k3 = 0.00001  # T -> LB (off)
k4 = 4.0   # T -> LR  (4 * k2)

def propensity(X):
    R = numpy.array([k0 * X[0], k1 * X[1], k2 * X[1], k3 * X[2], k4 * X[2]])
    RTot = R.sum()
    return R, RTot


def Next(X, Xz, Yz, t):  # Reaction per reaction for each species
    R, RTot = propensity(X)
    if RTot == 0 :
        return t, X #no reaction possible

    #stochastic choice of reaction
    V = numpy.random.random() * RTot
    i = numpy.searchsorted(numpy.cumsum(R), V)
    # update species with chosen random reaction
    X = X + numpy.array(smatrix[i])
    #time increment (Gillespie formula)
    tau = -numpy.log(numpy.random.random()) / RTot
    t = t + tau
    #trajectory
    Xz.append(t)
    #Yz.append(X[:]) #X[2] stores only T, here we store all vectors
    Yz.append(numpy.array(X) / Ntot * 100) #store it as percentage
    return t,X


def programme(t, tmax, X, sim):
    all_Xz = []
    all_Yz = []

    for sim in range(numsim):
        t = 0
        Xcurr = X[:]  # copy initial condition
        Xz, Yz = [], []
        print(f"Currently at simulation #{sim}")

        while t < tmax:
            t, Xcurr = Next(Xcurr, Xz, Yz, t)

        all_Xz.append(Xz)
        all_Yz.append(numpy.array(Yz))  # shape (steps, 3)

    # Plot each species with mean and ± std shadow
    species_names = ["Blunt linear", "Resected linear", "T-loop"]
    for species_idx, species_name in enumerate(species_names):
        # Interpolate all simulations on a common time grid
        t_max = max([max(Xz) for Xz in all_Xz])
        num_points = min(1000, max(len(Xz) for Xz in all_Xz))#Adaptive number of points:
        # Use as many points as there are total reaction events, or a fraction of that.
        t_grid = numpy.linspace(0, t_max, num_points)

        # Interpolate each simulation
        Y_interp = []
        for sim in range(numsim):
            Y_sim = numpy.array(all_Yz[sim][:, species_idx])
            t_sim = numpy.array(all_Xz[sim])
            Y_interp.append(numpy.interp(t_grid, t_sim, Y_sim))
        Y_interp = numpy.array(Y_interp)

        # Mean and std values
        mean_Y = numpy.mean(Y_interp, axis=0)
        std_Y = numpy.std(Y_interp, axis=0)

        # Mean and shadow plots
        plt.figure(figsize=(8, 5))
        plt.plot(t_grid, mean_Y, color='blue', label=f"{species_name} mean")
        plt.fill_between(t_grid, mean_Y - std_Y, mean_Y + std_Y, color='blue', alpha=0.3, label='±1 std')
        plt.ylabel(f"{species_name} (%)")
        plt.xlabel("Time (s)")
        plt.title(f"{species_name} over time (mean of {numsim} cells)")
        plt.ylim(0, 100)
        plt.legend()
        plt.show()


programme(t, tmax, X, numsim)  #Launch
