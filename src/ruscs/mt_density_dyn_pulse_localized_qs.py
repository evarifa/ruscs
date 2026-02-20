import numpy as np
import random
import copy
from typing import List, Optional, Set, Tuple, Sequence, Union
from .atomic_mt import mt_process_update_pulse

def add_new_spreaders(index: int, infected_vector_value_i: int, new_added: int, active_y: np.ndarray, active_z: np.ndarray, Ny: int, network_size: int, newconnections: List[List[int]], index_new_connections: int, kmax_dynamic: int, kmax: int, neighborz: List[List[int]], use_always_infected: bool, always_infected: Set[int], qs: bool, M: int) -> Tuple[np.ndarray, np.ndarray, int, int, int, int, int, int, Optional[int], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], int]:
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

def generate_newconnections(total_new_infected: int, active_y: np.ndarray, Ny: int, initial_nodes: int, network_size: int, kmax: int, newconnections: Optional[List[List[int]]] = None) -> List[List[int]]:
    """
    Extend or create the newconnections list (each sublist with fixed length kmax).
    If newconnections already exists, the function appends additional connection lists.
    
    For the first connection (if none exist):
      - Use as many active nodes (from active_y) as possible (up to kmax).
    For subsequent connections:
      - Include a growing number of new infected node IDs.
        The number of new infected nodes is at most half of kmax.
      - The remaining slots are filled with a random selection of active nodes.
      - If still slots remain, they are filled with random nodes.
    
    Args:
        total_new_infected (int): Number of new connection sublists to add.
        active_y (np.array): Array of active infected nodes.
        Ny (int): Current count of active infected nodes.
        initial_nodes (int): Initial number of nodes before new infections.
        network_size (int): Current network size.
        kmax (int): Fixed length of each sublist.
        newconnections (list, optional): Existing list of connection sublists.
    
    Returns:
        newconnections (list): Updated list of connection sublists.
    """
    if newconnections is None:
        newconnections = []
    
    existing = len(newconnections)
    total_required = int(existing + total_new_infected)
    
    # Extend the existing newconnections until we have added total_new_infected additional sublists.
    for i in range(existing, total_required):
        # For the very first sublist (if none exist).
        if i == 0 and existing < 1:
            candidate_active = list(active_y[:min(Ny, kmax)])
        else:
            # For subsequent connections, include a growing number of new infected nodes.
            # new_count is at most half of kmax and increases with each added connection.
            new_count = min(kmax // 2, i - existing + 1)
            candidate_new = [network_size + j - 1 for j in range(1, new_count + 1)]
            # The remaining slots go to active (old) nodes.
            old_count = kmax - new_count
            if Ny > 0 and old_count > 0:
                candidate_old = random.sample(list(active_y[:Ny]), min(old_count, Ny))
            else:
                candidate_old = []
            candidate_active = candidate_new + candidate_old
        
        # Remove duplicates if any.
        chosen_nodes = list(dict.fromkeys(candidate_active))
        # Fill remaining slots with random nodes.
        available = list(set(range(1, initial_nodes + i)) - set(chosen_nodes))
        num_needed = kmax - len(chosen_nodes)
        if len(available) >= num_needed:
            chosen_nodes.extend(random.sample(available, num_needed))
        else:
            chosen_nodes.extend(available)
            # In case there are still not enough nodes, fill by randomly choosing from available.
            while len(chosen_nodes) < kmax:
                chosen_nodes.append(random.choice(available))
        
        newconnections.append(chosen_nodes[:kmax])
    
    return newconnections

def mt_density_dynamic_qs_pulse_loc(
    infected_vector: Sequence[int],
    rate: Union[float, Sequence[float]],
    alpha: float,
    delta: float,
    maxiter: int,
    neighbors: Sequence[Sequence[int]],
    kmax: int,
    condition_infection_first_spreading: bool = False,
    use_always_infected: bool = False,
    qs: bool = False,
    M: int = 100,
    initial_infected: int = 1,
    n: int = 1,
    C_max: float = 0.5,
    step_size: float = 1.0,
    rand_size: int = 10000000,
    return_timeseries: bool = True,
    return_symbolic: bool = False,
    realization_type: int = 0,
    show_prints_process: bool = False,
    show_prints: bool = True,
    return_backup_indices: bool = False
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

    Returns:
        If return_timeseries is True, returns a tuple with elements:
            0 time_vector: np.ndarray (1D float array) -- recorded times
            1 density_y: np.ndarray (1D float array) -- infected density over time
            2 density_z: np.ndarray (1D float array) -- stifler density over time
            3 density_new_y: Optional[np.ndarray] -- new infected density over time or None
            4 density_new_z: Optional[np.ndarray] -- new stifler density over time or None
            5 symbolic_y: Optional[List[np.ndarray]] -- list (length <= maxiter) of arrays of infected node ids or None
            6 symbolic_z: Optional[List[np.ndarray]] -- list of arrays of stifler node ids or None
            7 realization_type: int
            8 reached_maxiter: bool
            9 r: int -- number of realizations run
            10 n: int -- random seed / counter
            11 time_first_backup_before_pulse: float
            12 time_first_backup_after_pulse: float
            13 network_size: int
            14 neighborz: Optional[List[List[int]]] -- adjacency list or None
            15 backup_indices: Optional[List[int]] -- indices of backups or None
            16 time: float -- last simulated time
            17 last_infected_index: Optional[int]
            18 filtered_density_y: np.ndarray (1D float array)

        If return_timeseries is False, returns a tuple with the final summary values in similar order,
        where array entries are replaced by scalar final values or None as indicated in the code.
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
        newconnections = []
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

                        # New code here:
                        total_new_infected = int(sum(infected_vector))
                        newconnections = generate_newconnections(total_new_infected, active_y, Ny, initial_nodes, network_size, kmax)

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


                        newconnections = generate_newconnections(infected_vector[index], active_y, Ny, initial_nodes, network_size, kmax, newconnections = copy.deepcopy(newconnections))

                        active_y, active_z, Ny, Ny_new, network_size, index_new_connections, kmax_dynamic, My, index_backup, backup_active_y, backup_My, backup_active_z, new_added = add_new_spreaders(
                            index, infected_vector[index], new_added, active_y, active_z, Ny, network_size, copy.deepcopy(newconnections), index_new_connections, kmax_dynamic, kmax, neighborz, use_always_infected, always_infected, qs, M
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
                        time_first_backup_before_pulse = np.copy(time)
                    if time_first_backup_after_pulse < 0:
                        time_first_backup_after_pulse = np.copy(time)
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
    
    return (time_vector[:(index)] if return_timeseries else time, #0
            density_y[:(index)] if return_timeseries else Ny/network_size, #1
            density_z[:(index)] if return_timeseries else Nz/network_size, #2
            density_new_y[:(index)] if return_timeseries else None, #3
            density_new_z[:(index)] if return_timeseries else None, #4
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
            time if return_timeseries else None, # 16
            last_infected_index if return_timeseries else None,  #17
            filtered_density_y[:(index)]) if return_timeseries else None #18