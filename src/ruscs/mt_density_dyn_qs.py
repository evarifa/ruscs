import numpy as np
import random
from .atomic_mt import mt_process_update

   
def mt_density_dynamic_qs(
    rate,
    alpha,
    delta,
    maxiter,
    neighbors,
    kmax,
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
    return_backup_indices=False
):
    """
    Executes MT on a network with 1 initially infected node to obtain the lifetime
    of a finite realization. A realization is considered finite when the number of 
    infected nodes reaches 0 before the coverage C reaches C_max.
    It also returns the class of nodes over time.

    Args:
        rate: Infection rate vector of length maxiter
        alpha: Recovery rate
        delta: Stifler to ignorant rate
        maxiter: Maximum number of iterations to run the simulation
        neighbors: Neighbors of each node
        kmax: Maximum degree of the network
        qs (optional): Use QS algorithm,
        M (optional): Size of QS memory
        n (optional): Seed for random number generator
        C_max (optional): Maximum coverage to reach
        return_only_final_values (optional): Don't return densities
        return_stiflers (optional): Return density of stiflers
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
            if return_stiflers:
                density_z: Density of stiflers over time
                density_new_z
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


    np.random.seed(n)
    random.seed(n)
    network_size = len(neighbors)
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
        My = sum(len(neighbors[i-1]) for i in infected)
        Nz = 0
        Nz_new = 0
        time = 0.0
        last_record_time = 0.0
        # QS objects for backup
        if return_backup_indices:
            backup_indices = []
        else:
            backup_indices = None
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
            density_y[0] = Ny / network_size
            density_new_y[0] = Ny_new / network_size

            density_z = np.zeros(maxiter + 1)
            density_new_z = np.zeros(maxiter + 1)
            density_z[0] = Nz / network_size
            density_new_z[0] = Nz / network_size

        while index < maxiter:
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
            ) = mt_process_update(
                rand,
                rn_index,
                s,
                Nz,
                inactive,
                active_z,
                show_prints_process,
                kmax,
                neighbors,
                active_y,
                Ny,
                rate[index],
                alpha,
                ever_infected,
                My,
                Nz_new,
                Ny_new,
                qs
            )
            true_my = sum(len(neighbors[i-1]) for i in active_y[:Ny])
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
            if show_prints_process:
                print("Ny_new: ", Ny_new, " Nz_new: ", Nz_new)
                print("Nx: ", len(inactive), " Ny: ", Ny, " Nz: ", Nz)
            # Save values at intervals determined by step_size
            k = 0
            while time - last_record_time >= step_size:  # Save values inside computed time
                index += 1
                if index >= maxiter:
                    break  # Exit the loop if index exceeds maxiter
                if return_timeseries:
                    density_y[index] = np.copy(Ny / network_size)
                    density_new_y[index] = np.copy(Ny_new / network_size)
                    Ny_new = 0
                    time_vector[index] = np.copy(time)
                    density_z[index] = np.copy(Nz / network_size)
                    density_new_z[index] = np.copy(Nz_new / network_size)
                    Nz_new = 0

                    if return_symbolic:
                        symbolic_y[index] = np.copy(active_y[:Ny])
                        symbolic_z[index] = np.copy(active_z[:Nz])
                # Save values for backup
                if qs:
                    if index < M:
                        backup_active_y[index] = np.copy(active_y)
                        backup_My[index] = np.copy(My)
                        backup_active_z[index] = np.copy(active_z)
                    else:
                        if random.random() <= 0.01: # Randomly replace values in backup
                            krand_M = random.randint(0, M - 1)
                            backup_active_y[k] = np.copy(active_y)
                            backup_My[k] = np.copy(My)
                            backup_active_z[k] = np.copy(active_z)

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
                    if show_prints:
                        print("Restoring backup")
                    m = min(index, M)
                    rand_m = random.randint(0, m-1)
                    active_y = np.copy(backup_active_y[:m][rand_m])
                    active_z = np.copy(backup_active_z[:m][rand_m])
                    My = np.copy(backup_My[:m][rand_m])
                    Ny = len(np.trim_zeros(active_y, 'b'))
                    Nz = len(np.trim_zeros(active_z, 'b'))
                    inactive = set(range(1, network_size + 1)) - set(active_y) - set(active_z)
            else:
                if mt_modified:
                    if len(inactive) == network_size:
                        if show_prints:
                            print("All nodes are ignorant. Reached absorbing state.")
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
            elif reached_coverage:
                realization_type = 1
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Ending endemic realization.")
            elif index >= maxiter:
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Reached max iterations")
                reached_maxiter = True
                realization_type = 2
            else:
                if show_prints:
                    print(f"Iterations: {index}")
                    print("Unknown realization type")
                realization_type = 3
            break
    if return_timeseries:
        return (time_vector[:index], #0
                density_y[:index], #1
                density_z[:index], #2
                density_new_y[:index], #3
                density_new_z[:index], #4
                symbolic_y[:(index)] if return_symbolic else None, # 5
                symbolic_z[:(index)] if return_symbolic else None, #6
                realization_type, #7
                reached_maxiter, #8
                r, #9
                n, #10
                backup_indices, #11
                time) #12
    else:
        return (time, #0
                Ny / network_size, #1
                Nz / network_size, #2
                realization_type, #3
                reached_maxiter, #4
                r, #5
                n) #6