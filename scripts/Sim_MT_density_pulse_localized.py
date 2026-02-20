import numpy as np
import matplotlib.pyplot as plt
import logging
import pickle
import yaml
import time
import copy
import os
import argparse

from ruscs.utils import InlineListDumper, init_logging
from ruscs.init_network import initialize_network_RRG_list
from ruscs.mt_density_dyn_pulse_localized_qs import mt_density_dynamic_qs_pulse_loc


def main():

    parser = argparse.ArgumentParser(description="MT density dynamics with localized pulse simulation.")
    parser.add_argument("--output", type=str, default="output/density_pulse_localized/",
                        help="Path to the output directory (default: output/density_pulse_localized/)")
    parser.add_argument("--name", type=str, default="density_pulse_localized",
                        help="Name prefix for log files (default: density_pulse_localized)")
    args = parser.parse_args()

    # Add custom representation to the Dumper
    yaml.add_representer(dict, InlineListDumper.represent_dict, Dumper=InlineListDumper)
    yaml.add_representer(list, InlineListDumper.represent_list, Dumper=InlineListDumper)

    output = args.output
    os.makedirs(output, exist_ok=True)

    # Create density vector
    maxiter = 1500
    steps_wait = 500
    steps_increase = 200
    # y_range = [2.16, 59]
    y_range = [0.05, 0.3]
    density_vector = np.zeros(maxiter, dtype=int)
    density_vector[500] = 50
    save_symbolic = True

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(density_vector)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Defined Additional Density")
    plt.savefig(output+"density_vector.png", bbox_inches='tight')
    plt.close()

    rate_vector = np.zeros(maxiter)
    rate_vector = rate_vector+0.05

    # possible dynamic rate
    # rate_vector = linear_increase(total_length = maxiter, y_min = y_range[0], y_max = y_range[1], steps_wait = steps_wait, steps_increase = steps_increase)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(rate_vector)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Defined Rate")
    plt.savefig(output+"rate_vector.png", bbox_inches='tight')
    plt.close()

    # Simulation parameters

    simulation_params = {
        "density_vector" : density_vector.tolist(),
        "steps_wait": steps_wait,
        "steps_increase": steps_increase,
        "y_range": y_range,
        "rates": rate_vector.tolist(),               
        "alfa": 0.5,    
        "delta": 0.89,
        "step_size": 0.2,   
        "coverage": 10,
        "N_vals": 1000, # network size
        "k_network": 10, # number of neighbours
        "initial_infected": 1,
        "realization_type": "any",
        "niter": 2, # 1000, # number of simulations
        "maxiter": maxiter, # maximum number of iterations for each simulation,
        "existent_network": False,
        "density_pulse": True,
        "use_always_infected": False
    }

    N = simulation_params['N_vals']
    maxiter = simulation_params['maxiter']
    use_always_infected = simulation_params['use_always_infected']
    if use_always_infected:
        print("Using always infected")
    else:
        print("Using normal spreaders")


    init_logging(output=output, name=args.name)
    logging.info(f'Number of nodes: {N}')
    logging.info(f'Number of iterations: {simulation_params["niter"]}')

    N = int(N)
    
    # Existent network
    if simulation_params['existent_network']:
        print("Loading network.")
        kmax = simulation_params['k_network']
        with open(f'{output}initial_network_N{N}_k{kmax}.pkl', 'rb') as file:
            neighborz = pickle.load(file)
        neighborz = [neighbors.tolist() for neighbors in neighborz]

    else:
        print("Creating network")
        neighborz, kmax = initialize_network_RRG_list(N, simulation_params['k_network'])
        with open(f'{output}initial_network_N{N}_k{kmax}.pkl', 'wb') as file:
            pickle.dump(neighborz, file)

    print('kmax', kmax)
    print('Length neighborz: ', len(neighborz))

    logging.info(f'K_av: {kmax}')
    n = 1
    
    time_list = []
    density_list = []
    simulation_time = []
    data_points = []
    reached_maxiter = []
    N_iter_needed = []
    seed_number = []
    realization_type_list = []
    if save_symbolic:
        symbolic_y_list = []
        symbolic_z_list = []
    time_backup_before = []
    time_backup_after = []
    new_networksize = []
    neighbours_list = []
    backup_indices = []
    filtered_density_list = []
    for i in range(simulation_params['niter']):
        print(i)
        neighborz_copy = copy.deepcopy(neighborz)
        start_time = time.time()
        res = mt_density_dynamic_qs_pulse_loc(
            infected_vector = density_vector,
            rate = rate_vector, 
            alpha = simulation_params['alfa'], 
            delta = simulation_params['delta'], 
            maxiter = maxiter, 
            neighbors = neighborz_copy, 
            kmax = kmax, 
            use_always_infected = use_always_infected,
            qs = True,
            M = 100,
            initial_infected = simulation_params['initial_infected'], 
            n = n,
            C_max = simulation_params['coverage'],
            return_timeseries = True,
            return_symbolic = save_symbolic,
            realization_type = simulation_params['realization_type'],
            step_size = simulation_params['step_size'], 
            show_prints_process = False,
            show_prints = False,
            return_backup_indices = True
        )
        if res[8] == False:
            print("Not reached maxiter")

        end_time = time.time()
        n = res[10]

        density_data = np.array([res[1], res[2], res[3], res[4]])
        density_list.append(density_data)
        time_list.append(res[0])
        if save_symbolic:
            symbolic_y_list.append(res[5])
            symbolic_z_list.append(res[6])

        simulation_time.append(end_time-start_time)
        data_points.append(len(res[0]))
        realization_type_list.append(res[7])
        reached_maxiter.append(res[8])
        N_iter_needed.append(res[9])
        seed_number.append(res[10])
        time_backup_before.append(res[11])
        time_backup_after.append(res[12])
        new_networksize.append(res[13])
        neighbours_list.append(copy.deepcopy(res[14]))
        backup_indices.append(res[15])
        filtered_density_list.append(res[18])

        # Every 100 iterations, save symbolic data and clear the lists
        if save_symbolic and (i + 1) % 100 == 0:
            with open(output + f"symbolic_dynamic_y_block_{(i + 1) // 100}.pkl", "wb") as file:
                pickle.dump(symbolic_y_list, file)
            with open(output + f"symbolic_dynamic_z_block_{(i + 1) // 100}.pkl", "wb") as file:
                pickle.dump(symbolic_z_list, file)
            # Clear the lists to free up memory
            symbolic_y_list = []
            symbolic_z_list = []

    # Define output metadata variables
    output_metadata = {
        "output_file": output,  # Path to the output file
        "run_time": simulation_time, # Total simulation runtime (s)
        "data_points": data_points, # Number of data points generated
        "realization_type": realization_type_list, # 0 finite / 1 endemic
        "reached_maxiter": reached_maxiter, # Reached max iterations
        "N_iter_needed": N_iter_needed, # Number of iterations needed for specific type
        "seed_number": seed_number,
        "time_backup_before": str(time_backup_before),
        "time_backup_after": str(time_backup_after),
        "new_networksize" : new_networksize
    }

    yaml_data = {
        "simulation_params": simulation_params, 
        "output_metadata": output_metadata
    }
    # Write data to a YAML file

    with open(output+f"simulation_config.yaml", "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, Dumper = InlineListDumper, sort_keys=False)
    with open(output+f"time_vector.pkl", 'wb') as file:
        pickle.dump(time_list, file)
    with open(output+f"densities_dynamic.pkl", 'wb') as file:
        pickle.dump(density_list, file)
    with open(output+f"neighbours_list.pkl", "wb") as file:
        pickle.dump(neighbours_list, file)
    with open(output+f"backup_indices.pkl", "wb") as file:
        pickle.dump(backup_indices, file)
    with open(output+f"filtered_density_list.pkl", "wb") as file:
        pickle.dump(filtered_density_list, file)
    
    # Save the remaining symbolic data if any are left
    if save_symbolic and symbolic_y_list:
        with open(output + f"symbolic_dynamic_y_block_final.pkl", "wb") as file:
            pickle.dump(symbolic_y_list, file)
        with open(output + f"symbolic_dynamic_z_block_final.pkl", "wb") as file:
            pickle.dump(symbolic_z_list, file)

if __name__ == "__main__":
    main()
