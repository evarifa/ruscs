import numpy as np
import random
import logging
import pickle
import yaml
import os
import copy
import argparse

from ruscs.utils import InlineListDumper, init_logging
from ruscs.init_network import initialize_network_RRG_list
from ruscs.mt_density_dyn_pulse_choice_qs import mt_density_dynamic_qs_pulse_choice

# Function to save data
def save_intermediate_results(output, r, iteration, time_list_i = None, density_list_i = None, symbolic_y_list_i = None, symbolic_z_list_i = None):
    r = round(r,4)
    if time_list_i:
        with open(f'{output}time_vector_{r}_{iteration}.pkl', 'wb') as file:
            pickle.dump(time_list_i, file)
    if density_list_i:
        with open(f'{output}densities_dynamic_{r}_{iteration}.pkl', 'wb') as file:
            pickle.dump(density_list_i, file)
    if symbolic_y_list_i:
        with open(f'{output}symbolic_dynamic_y_{r}_{iteration}.pkl', 'wb') as file:
            pickle.dump(symbolic_y_list_i, file)
    if symbolic_z_list_i:
        with open(f'{output}symbolic_dynamic_z_{r}_{iteration}.pkl', 'wb') as file:
            pickle.dump(symbolic_z_list_i, file)
    return

# @profile
def main():
    parser = argparse.ArgumentParser(description="MT density lifetime pulse simulation (near-seed).")
    parser.add_argument("--output", type=str, default="output/lifetime_pulse_nearseed/",
                        help="Path to the output directory (default: output/lifetime_pulse_nearseed/)")
    parser.add_argument("--name", type=str, default="lifetime_pulse_nearseed",
                        help="Name prefix for log files (default: lifetime_pulse_nearseed)")
    args = parser.parse_args()

    # Add custom representation to the Dumper
    yaml.add_representer(dict, InlineListDumper.represent_dict, Dumper=InlineListDumper)
    yaml.add_representer(list, InlineListDumper.represent_list, Dumper=InlineListDumper)

    output = args.output
    os.makedirs(output, exist_ok=True)

    init_logging(output=output, name=args.name)

    # Read average lifetime
    pulse_av_lifetime = False
    pulse = True
    if pulse:
        if pulse_av_lifetime:
            file = f'{output}lifetime_finite.txt'
            print("Reading file", file)
            data = np.loadtxt(file)
            av_lifetimes = data[:, 1]
            rates = data[:, 0]
        else:
            index_pulse = 0
            rates = np.arange(0.005, 0.22, 0.005)
            print("Index pulse", index_pulse)
    else:
        print("No pulse")
        rates = np.arange(0.005, 0.22, 0.005)
    print("Length rates", len(rates))

    # Simulation parameters
    alfa = 0.5
    delta = 1.0
    step_size = 0.1
    niter = 10 # 1000
    N = 100 # 1000
    maxiter = 10000 # 100000
    k_network = 3 # 10

    simulation_params = {
        "rates": rates.tolist(),               
        "alfa": alfa,    
        "delta": delta,
        "step_size": step_size,   
        "coverage": 0.8,
        "N_vals": N, # network size
        "k_network": k_network, # number of neighbours
        "initial_infected": 1,
        "realization_type": "any",
        "niter": niter, # number of simulations
        "maxiter": maxiter, # maximum number of iterations for each simulation,
        "repetitions": 1,
        "existent_network": False,
        "density_pulse": False,
        "use_always_infected": False,
        "pulse_number": 1,
        "pulse_av_lifetime" : pulse_av_lifetime,
        "condition_infection_first_spreading": True,
        "save_intermediate_results": False
    }


    logging.info(f'Number of nodes: {N}')
    logging.info(f'Number of iterations: {simulation_params["niter"]}')

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

    
    logging.info(f'Total rates {rates}')
    logging.info(f'Number of nodes: {N}')
    logging.info(f'Number of iterations: {niter}')
    print("Simulation parameters", simulation_params)
    N = int(N)
    logging.info(f'K_av: {kmax}')

    n = 1

    seed_number_array = np.zeros((len(rates), niter)) 
    new_networksize_array = np.zeros((len(rates), niter))
    last_infected_index_array = np.zeros((len(rates), niter))

    # new connections
    new_connections = 0

    with open(f'{output}lifetime_dz_finite_N{N}_k{kmax}_MT.txt', 'a') as f:
        for r_i in range(len(rates)):
            r = rates[r_i]
            rate_vector = np.zeros(maxiter)
            rate_vector = rate_vector+r
            logging.info(f'rate: {r}')
            density_vector = np.zeros(maxiter)
            if pulse and not simulation_params['condition_infection_first_spreading']:
                if pulse_av_lifetime:
                    index_pulse = int(av_lifetimes[r_i]*0.5/step_size)
                else:
                    index_pulse = index_pulse
                logging.info(f'Density pulse at: {index_pulse}')
                density_vector[index_pulse] = simulation_params['pulse_number']
                print(f'Spreaders Added: {density_vector[index_pulse]}')
            elif simulation_params['condition_infection_first_spreading']:
                density_vector = [simulation_params['pulse_number']]
                print(f'Spreaders Added After Second Spreader when Ny > 1: {density_vector}')


            # Initialize arrays
            time_niter = np.zeros(niter)
            r_niter = np.zeros(niter)
            dz_niter = np.zeros(niter)
            time_pulse = np.zeros(niter)

            time_list_i = []
            density_list_i = []
            symbolic_y_list_i = []
            symbolic_z_list_i = []

            for i in range(niter):
                added_spreaders = False
                k = 0
                while not added_spreaders:
                    k += 1
                    # Create initial infected
                    initial_infected = [random.randint(1, N)]
                    # Create new connections
                    if simulation_params['pulse_number'] > 0:
                        new_connections = []
                        new_i = 0
                        for new_node in range(N, N + int(sum(density_vector))):
                            # Randomly choose 10 existing nodes to connect to
                            chosen_nodes = random.sample(range(1, N+1+new_i), kmax)
                            new_connections.append(chosen_nodes)
                            new_i += 1
                        # Substitute the last element of the list with the initial infected node
                        new_connections[0][-1] = initial_infected[0]
                    else:
                        new_connections = []
                    res = mt_density_dynamic_qs_pulse_choice(
                        infected_vector = density_vector,
                        rate = rate_vector, 
                        alpha = alfa, 
                        delta = delta, 
                        maxiter = maxiter, 
                        neighbors = copy.deepcopy(neighborz), 
                        kmax = kmax, 
                        new_connections = copy.deepcopy(new_connections),
                        condition_infection_first_spreading = simulation_params['condition_infection_first_spreading'],
                        use_always_infected = False,
                        qs = False,
                        M = 100,
                        initial_infected = copy.deepcopy(initial_infected), 
                        n = n,
                        C_max = simulation_params['coverage'],
                        return_timeseries = True,
                        return_symbolic = True,
                        realization_type = 0,
                        step_size = step_size, 
                        show_prints_process = False,
                        show_prints = False
                    )
                    n = res[10]
                    if res[13]-N == int(simulation_params['pulse_number']):
                        added_spreaders = True
                    if res[8]:
                        print("Reached maxiter!")
                # print("Time", res[16])
                # print("new network size", res[13])
                time_niter[i] = res[16]
                r_niter[i] = res[9]
                dz_niter[i] = res[2][-1]
                time_pulse[i] = res[17]
                

                seed_number_array[r_i, i] = res[10]
                new_networksize_array[r_i, i] = res[13]
                last_infected_index_array[r_i, i] = res[17]

                density_data = np.array([res[1], res[2], res[3], res[4]])
                density_list_i.append(density_data)
                time_list_i.append(res[0])
                symbolic_y_list_i.append(res[5])
                symbolic_z_list_i.append(res[6])
            
            # Save results
            if simulation_params['save_intermediate_results']:
                save_intermediate_results(output, r, i+1, time_list_i = time_list_i, density_list_i = density_list_i, symbolic_y_list_i = symbolic_y_list_i, symbolic_z_list_i = symbolic_z_list_i)
            time_list_i = []
            density_list_i = []
            symbolic_y_list_i = []
            symbolic_z_list_i = []

            av_t = np.sum(time_niter)/niter
            av_r = np.sum(r_niter)/niter
            av_dz = np.sum(dz_niter)/niter
            av_pulse = np.sum(time_pulse*step_size)/niter

            logging.info(f'av_t: {av_t}')
            logging.info(f'av_r: {av_r}')
            logging.info(f'av_dz: {av_dz}')
            logging.info(f'av_pulse: {av_pulse}')

            f.write(f'{r} {av_t} {av_r} {av_dz} {av_pulse}\n')

    # Define output metadata variables
    output_metadata = {
        "output_file": output,  # Path to the output file
        "seed_number": seed_number_array.tolist(),
        "new_networksize" : new_networksize_array.tolist(),
        "last_infected_index": last_infected_index_array.tolist()
    }

    yaml_data = {
        "simulation_params": simulation_params, 
        "output_metadata": output_metadata
    }
    with open(output+f"simulation_config.yaml", "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, Dumper = InlineListDumper, sort_keys=False)

if __name__ == "__main__":
    main()
