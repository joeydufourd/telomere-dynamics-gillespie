import numpy
import math
import matplotlib.pyplot as plt
import time
from numba import njit
from numba.typed import List
from numba import prange

# Initialize species counts: [LB, LR, T] ; LB is Linear Blunt telomeres, LR Linear resected Telomeres, T Telomere-loops
X = numpy.array([46, 46, 0], dtype=numpy.float64) #forced numba dtype

# Parameters
#start = time.time()
numpy.random.seed(0)
numspecies = 3 #LB/LR/T
numreactions = 5 #Detailed in smatrix
numsim = 2 #number of simulations (=number of cells simulated)
tmax = (3600 * 8) * 2  # Here we use ESCs cell cycle time (8 hours) to render 2 cell cycles
G1_len = 3600 #G1 is approx 1 hour in ESCs
S_len  = 5*3600 #S is approx 5 hours in ESCs
G2_len = 2*3600 #G2 is approx 2 hours in ESCs
cycle_length = G1_len + S_len + G2_len

# Stoichiometry matrix, refer to the diagram for a more visual representation
smatrix = numpy.array([
    [-1, 1, 0],
    [1, -1, 0],
    [0, -1, 1],
    [1, 0, -1],
    [0, 1, -1]
])

# Rate constants for each phase
k_G1  = numpy.array([0.1, 0.00001, 1.0, 0.00001, 4.0])  # G1 less LB/LR more t-loops
k_S   = numpy.array([1, 1, 1.0, 0.00001, 2.0]) # S phase less t-loops to replicate, equilibrium for LB/LR [1.0, 1.0, 0.1, 0.00001, 2.0]
k_G2M = numpy.array([0.1, 0.00001, 1.0, 0.00001, 4.0])  # G2/M back to G1 like

# Gillespie propensity calculations
@njit
def propensity(X, k):
    R = numpy.array([k[0]*X[0], k[1]*X[1], k[2]*X[1], k[3]*X[2], k[4]*X[2]])
    return R, R.sum()

# Gillespie step (choosing which reaction to do at time t)
@njit
def Next(X, t, k):
    R, RTot = propensity(X, k)
    if RTot == 0:
        return t, X
    #Gillespie method, formula and time increment
    V = numpy.random.random() * RTot
    i = numpy.searchsorted(numpy.cumsum(R), V)
    X = X + smatrix[i]
    tau = -numpy.log(numpy.random.random()) / RTot
    return t + tau, X

# Division functions, most divisions result in equal partition of DNA material
@njit
def deterministic_partition(X):
    X = numpy.asarray(X, dtype=numpy.float64)
    Xd = X // 2
    X_rem = X - Xd
    return Xd + X_rem

# Rare non equal division
@njit
def partition_with_rare_noise(X, eps=1e-10):
    if numpy.random.random() < eps:
        return numpy.array([numpy.random.binomial(int(x), 0.5) for x in X], dtype=numpy.float64)
    else:
        return deterministic_partition(X)

# Simulation using time-based scaling
@njit
def simulate_cycle(X0, tmax):
    t_total = 0
    X = X0.copy()
    Xz, Yz = [], []

    while t_total < tmax:
        # Determine position in current cell cycle
        t_in_cycle = t_total % cycle_length

        if t_in_cycle < G1_len:
            k = k_G1
        elif t_in_cycle < G1_len + S_len:
            k = k_S
        else:
            k = k_G2M

        t_inc, X = Next(X, t_in_cycle, k)
        tau = t_inc - t_in_cycle
        t_total += tau

        Xz.append(t_total)
        total = sum(X)
        if total > 0:
            Yz.append((X / total) * 100.0)
        else:
            Yz.append(X.copy())

        # Division at end of each cycle
        if t_in_cycle + tau >= cycle_length:
            X = partition_with_rare_noise(X)

    return Xz, Yz, X

@njit(parallel=True)
def parallel_sim(X, numsim):
    all_Xz = List()
    all_Yz = List()

    for sim in prange(numsim):
        #print(f"Currently at simulation #{sim+1}")
        Xz, Yz, Xfinal = simulate_cycle(X, tmax)
        #Xz=numpy.array(Xz)
        #Yz = numpy.array(Yz)
        all_Xz.append(Xz)
        all_Yz.append(Yz)
    return all_Xz, all_Yz


# Main program

def programme(X, numsim):
    all_Xz, all_Yz = parallel_sim(X,numsim)
    all_Yz = [numpy.array(y) for y in all_Yz]
    all_Xz = [numpy.array(x) for x in all_Xz]
    # Plot
    species_names = ["Blunt linear", "Resected linear", "T-loop"]
    for species_idx, species_name in enumerate(species_names):
        t_max = max([max(Xz) for Xz in all_Xz])
        num_points = min(1000, max(len(Xz) for Xz in all_Xz))
        t_grid = numpy.linspace(0, t_max, num_points)

        # Interpolation for better data visualisation
        Y_interp = []
        for sim in range(numsim):
            Y_sim = numpy.array(all_Yz[sim])[:, species_idx]
            t_sim = numpy.array(all_Xz[sim])
            Y_interp.append(numpy.interp(t_grid, t_sim, Y_sim))
        Y_interp = numpy.array(Y_interp)

        mean_Y = numpy.mean(Y_interp, axis=0)
        std_Y = numpy.std(Y_interp, axis=0)

        plt.figure(figsize=(10,6))
        plt.plot(t_grid, mean_Y, color='blue', label=f"{species_name} mean")
        plt.fill_between(t_grid, mean_Y - std_Y, mean_Y + std_Y, color='blue', alpha=0.3, label='±1 std')

        # Phase shading and division lines
        cycle = cycle_length
        num_cycles = int(t_max // cycle) + 1
        for n in range(num_cycles):
            g1_start = n*cycle
            g1_end = g1_start + G1_len
            s_end = g1_start + G1_len + S_len
            g2m_end = g1_start + cycle_length
            plt.axvspan(g1_start, g1_end, facecolor="lightgreen", alpha=0.2, label="G1" if n==0 else "")
            plt.axvspan(g1_end, s_end, facecolor="orange", alpha=0.2, label="S" if n==0 else "")
            plt.axvspan(s_end, g2m_end, facecolor="lightblue", alpha=0.2, label="G2/M" if n==0 else "")
            plt.axvline(g2m_end, color="red", linestyle="--", alpha=0.7, label="Division" if n==0 else "")

        plt.ylabel(f"{species_name} (%)")
        plt.xlabel("Time (s)")
        plt.title(f"{species_name} over time (mean of {numsim} cells)")
        plt.ylim(0, 100)
        plt.xlim(0,tmax)
        plt.legend()
        plt.show()
    #end = time.time() # njit improved Execution time: 102.9640 seconds, 8 times improvement (5sim) ; parallel improves to Execution time: 70.7025 seconds (5sim)
    print(f"Execution time: {end - start:.4f} seconds")

# Run !
programme(X, numsim)
