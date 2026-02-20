# Core function for the process update 
def mt_process_update(
    rand,
    rn_index,
    s,
    Nz,
    inactive,
    active_z,
    show_prints,
    kmax,
    neighbors,
    active_y,
    Ny,
    rate,
    alpha,
    ever_infected,
    My,
    Nz_new = None,
    Ny_new = None,
    qs = False
):
    """
    Update the process of the MT model.
    """
    if rand[rn_index] < s:
        # stifler (z) becomes ignorant (x)
        rn_index += 1
        node_index = int(rand[rn_index]*Nz)
        rn_index += 1
        inactive.add(active_z[node_index])
        if show_prints:
            print(f"Stifler (z, {active_z[node_index]}) becomes ignorant (x) !!")
        Nz -= 1
        active_z[node_index] = active_z[Nz]
        if qs:
            active_z[Nz] = 0
    else:
        # infection ?
        rn_index += 1
        node_deg = 0
        while rand[rn_index] >= node_deg / kmax:
            rn_index += 1
            node_index = int(rand[rn_index]*Ny)
            rn_index += 1
            node_deg = len(neighbors[active_y[node_index] - 1])
        rn_index += 1
        rand_node_index = int(rand[rn_index]*node_deg)
        rn_index += 1
        rand_node = neighbors[active_y[node_index] - 1][rand_node_index]

        if rand_node in inactive:
            l = rate / (rate + alpha)
            if rand[rn_index] < l:
                # X learns the rumour and becomes spreader (Y)
                rn_index += 1
                node_deg = len(neighbors[rand_node - 1])
                inactive.remove(rand_node)
                active_y[Ny] = rand_node
                My += node_deg
                Ny += 1
                ever_infected.add(rand_node)
                if show_prints:
                    print(f"X {rand_node} learns the rumour and becomes spreader (Y)")
                if Ny_new is not None:
                    Ny_new += 1
        else:
            l = alpha / (rate + alpha)
            # Y learns the rumour and becomes stifler (Z)
            if rand[rn_index] < l:
                rn_index += 1
                My -= node_deg
                Ny -= 1
                active_z[Nz] = active_y[node_index]
                active_y[node_index] = active_y[Ny]
                if qs:
                    active_y[Ny] = 0
                Nz += 1
                if show_prints:
                    print(f"Y {active_z[Nz-1]} learns the rumour and becomes stifler (Z)")
                if Nz_new is not None:
                    Nz_new += 1
    return rn_index, Nz, Ny, My, inactive, active_z, active_y, ever_infected, Nz_new, Ny_new

def mt_process_update_pulse(
    rand,
    rn_index,
    s,
    Nz,
    inactive,
    active_z,
    show_prints,
    kmax,
    neighbors,
    active_y,
    Ny,
    rate,
    alpha,
    ever_infected,
    My,
    always_infected = None,
    use_always_infected = False,
    Nz_new = None,
    Ny_new = None,
    qs = False
):
    """
    Update the process of the MT model.
    """
    if rand[rn_index] < s:
        # stifler (z) becomes ignorant (x)
        rn_index += 1
        node_index = int(rand[rn_index]*Nz)
        rn_index += 1
        inactive.add(active_z[node_index])
        if show_prints:
            print(f"Stifler (z, {active_z[node_index]}) becomes ignorant (x) !!")
        Nz -= 1
        active_z[node_index] = active_z[Nz]
        if qs:
            active_z[Nz] = 0
    else:
        # infection ?
        rn_index += 1
        node_deg = 0
        while rand[rn_index] >= node_deg / kmax:
            rn_index += 1
            node_index = int(rand[rn_index]*Ny)
            rn_index += 1
            node_deg = len(neighbors[active_y[node_index] - 1])
        rn_index += 1
        rand_node_index = int(rand[rn_index]*node_deg)
        rn_index += 1
        rand_node = neighbors[active_y[node_index] - 1][rand_node_index]

        if rand_node in inactive:
            l = rate / (rate + alpha)
            if rand[rn_index] < l:
                # X learns the rumour and becomes spreader (Y)
                rn_index += 1
                node_deg = len(neighbors[rand_node - 1])
                inactive.remove(rand_node)
                active_y[Ny] = rand_node
                My += node_deg
                Ny += 1
                ever_infected.add(rand_node)
                if show_prints:
                    print(f"Node {rand_node} (x) learns the rumour and becomes spreader (y)")
                if Ny_new is not None:
                    Ny_new += 1
        elif use_always_infected and active_y[node_index] in always_infected:
            pass
        else:
            l = alpha / (rate + alpha)
            # Y learns the rumour and becomes stifler (Z)
            if rand[rn_index] < l:
                rn_index += 1
                My -= node_deg
                Ny -= 1
                active_z[Nz] = active_y[node_index]
                active_y[node_index] = active_y[Ny]
                if qs:
                    active_y[Ny] = 0
                Nz += 1
                if show_prints:
                    print(f"Y {active_z[Nz-1]} learns the rumour and becomes stifler (Z)")
                if Nz_new is not None:
                    Nz_new += 1
    return rn_index, Nz, Ny, My, inactive, active_z, active_y, ever_infected, Nz_new, Ny_new