import numpy as np
import random
from .atomic_mt import mt_process_update_pulse
import copy

def add_new_spreaders(index, infected_vector_value_i, new_added, active_y, active_z, Ny, network_size, newconnections, index_new_connections, kmax_dynamic, kmax, neighborz, use_always_infected, always_infected, qs, M):
    new_infected = infected_vector_value_i
    new_added = new_added + new_infected
    infected_nodes = [i for i in range(network_size+1, network_size + 1 + int(new_infected))]
    active_y = np.insert(active_y, Ny, infected_nodes)
    active_z = np.insert(active_z, [network_size], [0]*int(new_infected))
    Ny = Ny + int(new_infected)
    Ny_new = int(new_infected)
    if use_always_infected:
        always_infected.update(infected_nodes)
    
    # New conections as input
    for i in range(int(new_infected)):
        neighborz.append(newconnections[index_new_connections])
        for node in newconnections[index_new_connections]:
            neighborz[node-1].append(infected_nodes[i])
            if len(neighborz[node-1]) > kmax_dynamic:
                kmax_dynamic = len(neighborz[node-1])
        network_size += 1
        index_new_connections += 1
    
    My = sum(len(neighborz[i-1]) for i in active_y[:Ny])
    
    index_backup = None
    backup_active_y = None
    backup_My = None
    backup_active_z = None
    if qs:
        index_backup = 0
        backup_active_y = np.zeros((M, network_size), dtype=int)
        backup_active_y[0] = np.copy(active_y)
        backup_My = np.zeros(M, dtype=int)
        backup_My[0] = np.copy(My)
        backup_active_z = np.zeros((M, network_size), dtype=int)
        backup_active_z[0] = np.copy(active_z)
    return active_y, active_z, Ny, Ny_new, network_size, index_new_connections, kmax_dynamic, My, index_backup, backup_active_y, backup_My, backup_active_z, new_added

   
def mt_density_dynamic_qs_pulse_choice(
    infected_vector,
    rate,
    alpha,
    delta,
    maxiter,
    neighbors,
    kmax,
    new_connections,
    condition_infection_first_spreading = False,
    use_always_infected = False,
    qs = False,
    M = 100,
    initial_infected=1,
    n=1,
    C_max=0.5,
    step_size=1.,
    rand_size = 10000000,
    return_timeseries=True,
    return_symbolic=False,
    realization_type=0,
    show_prints_process=False,
    show_prints=True,
    return_backup_indices = False
):
    """
    Executes MT on a network with 1 initially infected node to obtain the lifetime
    of a finite realization. A realization is considered finite when the number of 
    infected nodes reaches 0 before the coverage C reaches C_max.
    It also returns the class of nodes over time.

    Args:
        infected_vector: Vector of new infections at each time step
        rate: Infection rate vector of length maxiter
        alpha: Recovery rate
        delta: Stifler to ignorant rate
        maxiter: Maximum number of iterations to run the simulation
        neighbors: Neighbors of each node
        kmax: Maximum degree of the network
        use_always_infected (optional): Flag parameter to keep new spreaders active always
        qs (optional): Use QS algorithm,
        M (optional): Size of QS memory
        n (optional): Seed for random number generator
        C_max (optional): Maximum coverage to reach
        return_only_final_values (optional): Don't return densities
        return_symbolic (optional): Return symbolic values
        realization_type (optional): Type of realization (finite (0) or endemic (1))
        show_prints (optional): Show prints
        step_size (optional): Time step size 
        rand_size (optional): Size of random number vector
    Value:
        if return_timeseries
            time_vector: Time vector of simulation
            density_y: Density of infected nodes over time
            density_new_y
            if return_symbolic:
                symbolic_y: Infected nodes over time
                symbolic_z: Stifler nodes over time
            realization_type: Type of realization (finite or endemic)
            reached_maxiter: bool
            n: random seed number
        else:
            time: final time value
            dy: final value of spreaders density
            dz: final value of stiflers density
            realization_type
            reached_maxiter
            r: number of realizations containing the last wanted type
            n
    """
    if isinstance(rate, (int, float)):
        rate = [rate]*maxiter
    else:
        if len(rate) != maxiter:
                raise ValueError("Rate vector must be of length maxiter.")

    if not condition_infection_first_spreading:
        if len(infected_vector) != maxiter:
            raise ValueError("New spreaders vector must be of length maxiter.")

        indices_infected = np.where(infected_vector != 0)[0]
        length_infected = len(indices_infected)
    else:
        indices_infected = []
        length_infected = sum(infected_vector)
    first_infection_step = True

    # Initialize parameter initial infected
    if isinstance(initial_infected, (int, float)):
        initial_infected_is_vector = False
    else:
        initial_infected_is_vector = True

    np.random.seed(n)
    random.seed(n)
    reached_coverage = False
    reached_maxiter = False
    r = 0
    rand = np.random.rand(rand_size)
    rn_index = np.random.randint(1, rand_size - 1)

    # Define dynamic rate

    if delta > 0:
        mt_modified = True
        if show_prints:
            print("Model MT modified: Single absorbing state")
    else:
        mt_modified = False
        if show_prints:
            print("Model MT standard: Multiple absorbing states")

    while True:
        index = 0
        index_backup = 0
        neighborz = copy.deepcopy(neighbors)
        newconnections = copy.deepcopy(new_connections)
        network_size = len(neighborz)
        initial_nodes = network_size
        if initial_infected_is_vector:
            infected = copy.deepcopy(initial_infected)
        else:
            infected = np.random.choice(network_size, initial_infected, replace=False) + 1
        rn_index += 1
        active_y = np.zeros(network_size, dtype=int)
        active_y[:len(infected)] = infected
        active_z = np.zeros(network_size, dtype=int)
        inactive = set(range(1, network_size + 1)) - set(infected)
        ever_infected = set(infected)
        C = len(ever_infected) / network_size
        Ny = len(infected)
        Ny_new = len(infected)
        My = sum(len(neighborz[i-1]) for i in infected)
        Nz = 0
        Nz_new = 0
        time = 0.0
        last_record_time = 0.0
        time_first_backup_before_pulse = -1
        time_first_backup_after_pulse = 1
        kmax_dynamic = kmax
        last_infected_index = None
        if return_backup_indices:
            backup_indices = []
        else:
            backup_indices = None
        # always infected 
        if use_always_infected:
            always_infected = set()
        else:
            always_infected = None
        # QS objects for backup
        if qs:
            backup_active_y = np.zeros((M, network_size), dtype=int)
            backup_active_y[0] = np.copy(active_y)
            backup_My = np.zeros(M, dtype=int)
            backup_My[0] = My
            backup_active_z = np.zeros((M, network_size), dtype=int)

        if return_timeseries:
            if return_symbolic:
                symbolic_y = [None] * (maxiter + 1)
                symbolic_y[0] = infected
                symbolic_z = [None] * (maxiter + 1)
                symbolic_z[0] = np.empty(0)

            density_y = np.zeros(maxiter + 1)
            density_new_y = np.zeros(maxiter + 1)
            time_vector = np.zeros(maxiter + 1)
            density_z = np.zeros(maxiter + 1)
            density_new_z = np.zeros(maxiter + 1)
            filtered_density_y = np.zeros(maxiter + 1)

            density_y[0] = Ny / network_size
            density_new_y[0] = Ny_new / network_size
            density_z[0] = Nz / network_size
            density_new_z[0] = Nz / network_size
            filtered_density_y[0] = np.sum(active_y[:Ny] < 1001) / network_size

        index_new_connections = 0
        new_added = 0
        while index < (maxiter):
            # if show_prints:
            #     print(f"Nx, Ny, Nz: {len(inactive)}, {Ny}, {Nz}")

            

            # Random number preparation
            if rn_index >= rand_size - 10_000:
                rand = np.random.rand(rand_size)
                rn_index = np.random.randint(1, rand_size)

            # Update time step
            dyn_R = delta * Nz + (rate[index] + alpha) * My
            time += 1 / dyn_R

            # Infection or recovery
            s = delta * Nz / dyn_R
            (
                rn_index, Nz, Ny, My, inactive, active_z, 
                active_y, ever_infected, Nz_new, Ny_new
            ) = mt_process_update_pulse(
                rand,
                rn_index,
                s,
                Nz,
                inactive,
                active_z,
                show_prints_process,
                kmax_dynamic,
                neighborz,
                active_y,
                Ny,
                rate[index],
                alpha,
                ever_infected,
                My,
                always_infected = always_infected,
                use_always_infected = use_always_infected,
                Nz_new = Nz_new,
                Ny_new = Ny_new,
                qs = qs
            )
            true_my = sum(len(neighborz[i-1]) for i in active_y[:Ny])
            if true_my != My:
                print("ERROR: Inconsistency in My")
                print("My: ", My, " True My: ", true_my)
                break
            if len(inactive) + Ny + Nz != network_size:
                print("ERROR: Inconsistency in network size")
                print("Nx: ", len(inactive), " Ny: ", Ny, " Nz: ", Nz)
                print("sum classes: ", len(inactive) + Ny + Nz)
                break
            if show_prints_process:
                print("index: ", index)
                print("Ny_new: ", Ny_new, " Nz_new: ", Nz_new)
                print("Nx: ", len(inactive), " Ny: ", Ny, "My: ", My, " Nz: ", Nz)
                
            # Save values for backup
            if qs and Ny > 0:
                if index_backup < M:
                    backup_active_y[index_backup] = np.copy(active_y)
                    backup_My[index_backup] = np.copy(My)
                    backup_active_z[index_backup] = np.copy(active_z)
                    index_backup += 1
                else:
                    if random.random() <= 0.1: # Randomly replace values in backup
                        rand_k = random.randint(0, M - 1)
                        backup_active_y[rand_k] = np.copy(active_y)
                        backup_My[rand_k] = np.copy(My)
                        backup_active_z[rand_k] = np.copy(active_z)
            # Save values at intervals determined by step_size
            k = 0
            while time - last_record_time >= step_size:  # Save values inside computed time
                index += 1
                first_infection_step = True
                if index >= maxiter:
                    break  # Exit the loop if index exceeds maxiter
                if return_timeseries:
                    density_y[index] = np.copy(Ny / network_size)
                    density_new_y[index] = np.copy(Ny_new / network_size)
                    filtered_density_y[index] = np.sum(active_y[:Ny] < 1001) / network_size
                    Ny_new = 0
                    time_vector[index] = np.copy(time)
                    density_z[index] = np.copy(Nz / network_size)
                    density_new_z[index] = np.copy(Nz_new / network_size)
                    Nz_new = 0

                    if return_symbolic:
                        symbolic_y[index] = np.copy(active_y[:Ny])
                        symbolic_z[index] = np.copy(active_z[:Nz])
                ###
                # Update density
                if condition_infection_first_spreading: # Use only once
                    if last_infected_index is None and Ny > 1:
                        if show_prints:
                            print("Adding new spreaders, Ny: ", Ny)
                        time_first_backup_after_pulse = -1
                        active_y, active_z, Ny, Ny_new, network_size, index_new_connections, kmax_dynamic, My, index_backup, backup_active_y, backup_My, backup_active_z, new_added = add_new_spreaders(
                            index, sum(infected_vector), new_added, active_y, active_z, Ny, network_size, newconnections, index_new_connections, kmax_dynamic, kmax, neighborz, use_always_infected, always_infected, qs, M
                        )
                        first_infection_step = False
                        last_infected_index = index
                else:
                    if index in indices_infected and first_infection_step:
                        if show_prints:
                            print("Adding new spreaders")
                        time_first_backup_after_pulse = -1
                        active_y, active_z, Ny, Ny_new, network_size, index_new_connections, kmax_dynamic, My, index_backup, backup_active_y, backup_My, backup_active_z, new_added = add_new_spreaders(
                            index, infected_vector[index], new_added, active_y, active_z, Ny, network_size, newconnections, index_new_connections, kmax_dynamic, kmax, neighborz, use_always_infected, always_infected, qs, M
                        )
                        last_infected_index = index
                        
                        first_infection_step = False
                ###
                last_record_time += step_size

            # compute coverage
            C = len(ever_infected) / network_size
            if C >= C_max:
                if show_prints:
                    print("Reached coverage. Endemic realization.")
                reached_coverage = True
                break

            # Type of realization
            if qs:
                if Ny == 0:
                    if return_backup_indices:
                        backup_indices.append(index)
                    if time_first_backup_before_pulse < 0:
                        time_first_backup_before_pulse = float(time.copy())
                    if time_first_backup_after_pulse < 0:
                        time_first_backup_after_pulse = float(time.copy())
                    if show_prints:
                        print("Restoring backup")

                    m = min(index_backup, M)
                    rand_m = random.randint(0, m-1)
                    active_y = np.copy(backup_active_y[:m][rand_m])
                    active_z = np.copy(backup_active_z[:m][rand_m])
                    My = np.copy(backup_My[:m][rand_m])
                    Ny = len(np.trim_zeros(active_y, 'b'))
                    Nz = len(np.trim_zeros(active_z, 'b'))
                    inactive = set(range(1, network_size + 1)) - set(active_y) - set(active_z)
            if mt_modified:
                if Ny < 1 and Nz < 1:
                    if show_prints:
                        print("Index", index)
                        print("Ny", Ny, "Nz", Nz, "Nx", len(inactive))
                        print("network_size", network_size)
                        print("Ny and Nz are 0. All nodes are ignorant. Reached absorbing state.")
                    break
            else:
                if Ny < 1:
                    if show_prints:
                        print("No infected nodes. Finite realization (reached absorbing state).")
                    break

        n += 1
        r += 1

        # Specify realization and break
        if realization_type == 0:
            if Ny < 1:
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Ending finite realization.")
                break
        elif realization_type == 1:
            if reached_coverage:
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Reached coverage. Ending endemic realization.")
                break
            elif index >= maxiter:
                reached_maxiter = True
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Reached max iterations. Ending endemic realization.")
                break
        else:
            if Ny < 1:
                realization_type = 0
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Ending finite realization.")
                break
            elif reached_coverage:
                realization_type = 1
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Ending endemic realization.")
                break
            elif index >= maxiter:
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Reached max iterations")
                reached_maxiter = True
                realization_type = 2
                break
            else:
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Unknown realization type")
                realization_type = 3
                break
    # Make variables None if not returned
    if length_infected < 1:
        neighborz = None
    if not return_symbolic:
        symbolic_y = None
        symbolic_z = None
    
    if return_timeseries:
        return (time_vector[:(index)], #0
                density_y[:(index)], #1
                density_z[:(index)], #2
                density_new_y[:(index)], #3
                density_new_z[:(index)], #4
                symbolic_y[:(index)] if return_symbolic else None, # 5
                symbolic_z[:(index)] if return_symbolic else None, #6
                realization_type, #7
                reached_maxiter, #8
                r, #9
                n, #10
                time_first_backup_before_pulse, #11
                time_first_backup_after_pulse, #12
                network_size, #13
                neighborz, #14
                backup_indices, #15
                time, # 16
                last_infected_index, #17
                filtered_density_y[:(index)]) #18
    else:
        return (time, 
                Ny / network_size, 
                Nz / network_size, 
                realization_type, 
                reached_maxiter, 
                r,
                n,
                time_first_backup_before_pulse, 
                time_first_backup_after_pulse,
                network_size,
                neighborz,
                backup_indices)