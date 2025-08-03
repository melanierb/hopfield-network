import numpy as np
from HopfieldNetworkClass import DataSaver, Patterns, HopfieldNetwork

def main():
    # Definition of parameters
   num_patterns = 50
   pattern_size = 2500
   num_perturb = 1000
   max_iter_synch = 20
   max_iter_asynch = 30000
   soft_conver_iter = 10000
   skip = 10
   pattern_num = np.random.randint(0, num_patterns - 1)

   rule1 = 'hebbian'
   rule2 = 'storkey'
   
   rule3 = 'hebbian_sync'
   rule4 = 'hebbian_async'
   rule5 = 'storkey_sync'
   rule6 = 'storkey_async'


   # initialize a pattern
   patt = Patterns(num_patterns, pattern_size, pattern_num)

   # get one specific pattern
   pattern = patt.get_pattern(pattern_num)

   # perturb a specific pattern
   patt.perturb_patterns(pattern, num_perturb)

   # get the perturbed pattern
   perturbed_pattern = patt.get_perturbed_pattern()

   # get all the patterns
   patterns = patt.get_patterns()

   # initialize the hopfield network
   HN_heb = HopfieldNetwork(patterns, rule1)
   HN_sto = HopfieldNetwork(patterns, rule2)
   
   saver_heb = DataSaver()
   saver_sto = DataSaver()
   saver_heb_async = DataSaver()
   saver_sto_async = DataSaver()

   #Running the dynamical_sync system
   HN_heb.dynamics(perturbed_pattern, saver_heb, max_iter_synch)
   HN_sto.dynamics(perturbed_pattern, saver_sto, max_iter_synch)

   # Running the dynamical_async system
   HN_heb.dynamics_async(perturbed_pattern, saver_heb_async, max_iter_asynch, soft_conver_iter, skip)
   HN_sto.dynamics_async(perturbed_pattern, saver_sto_async, max_iter_asynch, soft_conver_iter, skip)

   # Evaluating the energy function
   saver_heb.plot_energy()
   saver_sto.plot_energy()
   saver_heb_async.plot_energy()
   saver_sto_async.plot_energy()

   # Visualization of the evolution of states
   saver_heb.visualization(patt, rule3)
   saver_heb_async.visualization(patt, rule4)
   saver_sto.visualization(patt, rule5)
   saver_sto_async.visualization(patt, rule6)