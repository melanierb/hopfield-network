import numpy as np
import HopfieldNetwork as fct
import utils as plot


def main():
    # Definition of parameters
    num_patterns = 50
    pattern_size = 2500
    num_perturb = 1000
    max_iter_synch = 20
    max_iter_asynch = 30000
    soft_conver_iter = 10000
    pattern_num = np.random.randint(0, num_patterns - 1)

    # Creating random patterns
    patterns = fct.generate_patterns(num_patterns, pattern_size)

    # Creating the checkerboard
    patterns = plot.generate_checkerboard(patterns, pattern_num)

    # Creating the hebbian/ storkey weight matrix
    weight_heb = fct.hebbian_weights(patterns)
    weight_sto = fct.storkey_weights(patterns)

    # Storing the random patterns in a Hopfield Network
    hopfield_networks = patterns.copy()
    hopfield_networks[pattern_num] = fct.perturb_patterns(patterns[pattern_num], num_perturb)

    # Running the synchronous dynamical system
    historic_state_heb = fct.dynamics(hopfield_networks[pattern_num], weight_heb, max_iter_synch)
    historic_state_sto = fct.dynamics(hopfield_networks[pattern_num], weight_sto, max_iter_synch)

    # Running the asynchronous dynamical system
    historic_state_asynch_heb = fct.dynamics_async(hopfield_networks[pattern_num], weight_heb, max_iter_asynch,
                                                   soft_conver_iter)
    historic_state_asynch_sto = fct.dynamics_async(hopfield_networks[pattern_num], weight_sto, max_iter_asynch,
                                                   soft_conver_iter)

    # Evaluating the energy function
    plot.time_energy(historic_state_heb, weight_heb)
    plot.time_energy(historic_state_sto, weight_sto)
    plot.time_energy(historic_state_asynch_heb, weight_heb)
    plot.time_energy(historic_state_asynch_sto, weight_sto)

    # Visualization of the evolution of our state
    plot.visualization(patterns, historic_state_sto, 'storkey_sync')
    plot.visualization(patterns, historic_state_asynch_sto, 'storkey_aync')
    plot.visualization(patterns, historic_state_heb, 'hebbian_sync')
    plot.visualization(patterns, historic_state_asynch_heb, 'hebbian_async')