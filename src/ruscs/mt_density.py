import numpy as np
import random
from .atomic_mt import mt_process_update

def mt_density(rate,
               alpha,
               delta,
               maxiter,
               neighborz,
               kmax,
               initial_infected=1,
               n=1,
               C_max=10,
               step_size=1,
               show_prints=False):
    """
    Executes MT on a network.
    Args:
        rate: Infection rate
        alpha: Recovery rate
        delta: Stifler to ignorant rate
        maxiter: Maximum number of iterations
        neighborz: Neighbors of each node
        kmax: Maximum degree of the network
        initial_infected: Number of initially infected nodes (randomly selected). Default is 1.
        n: Seed for random number generator
        C_max: Maximum coverage to reach
        step_size: Step size for recording data
        show_prints: Show debug prints
    """
    np.random.seed(n)
    random.seed(n)
    rand_size = 10000000
    network_size = len(neighborz)
    infected = list(np.random.choice(network_size, initial_infected, replace=False) + 1)
    density_y = np.zeros(maxiter + 1)
    active_y = np.zeros(network_size, dtype=int)
    active_y[:len(infected)] = infected
    active_z = np.zeros(network_size, dtype=int)
    inactive = set(range(1, network_size + 1)) - set(infected) # x
    ever_infected = set(infected)
    C = len(ever_infected) / network_size
    Ny = len(infected)
    My = sum([len(neighborz[i-1]) for i in infected])
    Nz = 0 # Number of stiflers
    time = 0
    density_y[0] = Ny/network_size
    index = 1
    last_record_time = 0.0
    time_vector  = np.zeros(maxiter + 1)

    density_z = np.zeros(maxiter + 1)
    density_z[0] = Nz/network_size

    rand = np.random.rand(rand_size)
    rn_index = random.randint(0, rand_size-1)

    if delta > 0:
        mt_modified = True
        print("Model MT modified: Single absorbing state")
    else:
        mt_modified = False
        print("Model MT standard: Multiple absorbing states")

    while index < maxiter:

        # Random number preparation
        if rn_index >= rand_size-10000:
            rand = np.random.rand(rand_size)
            rn_index = random.randint(0, rand_size-1)
        
        # Update time step
        dyn_R = delta*Nz + (rate + alpha)*My
        time += 1/dyn_R

        # Infection or recovery
        s = delta * Nz / dyn_R

        # Call process update
        
        rn_index, Nz, Ny, My, inactive, active_z, active_y, ever_infected, Nz_new, Ny_new = mt_process_update(
                rand,
                rn_index,
                s,
                Nz,
                inactive,
                active_z,
                show_prints, # show_prints,
                kmax,
                neighborz,
                active_y,
                Ny,
                rate,
                alpha,
                ever_infected,
                My
            )
        if show_prints:
            print("Nx: ", len(inactive), " Ny: ", Ny, " Nz: ", Nz)

        while time - last_record_time >= step_size:
            if index > maxiter:
                print("Reached time end of simulation.")  # Exit the loop if index exceeds maxiter
            index += 1
            density_y[index] = Ny/network_size
            density_z[index] = Nz/network_size
            time_vector[index] = time
            last_record_time += step_size

        # compute coverage
        C = len(ever_infected) / network_size
        if C >= C_max:
            print("Reached coverage. Endemic realization.")
            break
        # MT modified
        if mt_modified:
            if len(inactive) == network_size:
                print("All nodes are ignorant. Reached absorbing state.")
                break
        else:
            # Standard MT
            if Ny == 0:
                print("No spreaders left. Reached absorbing state.")
                break

        
    return time_vector[:index], density_y[:index], density_z[:index]