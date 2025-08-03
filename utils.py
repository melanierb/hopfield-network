"""
Module containing matlab code for visualization of the Hopfield Network.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import HopfieldNetwork as fct




def time_energy(historic_state, weights):
    """
    Generates a time-energy plot.

    Parameters
    ----------
    historic_state : list
                     state history until convergence of a given pattern
    weights : array
              weight matrix of the memorized patterns

    Returns
    -------
    Integer or an array of integer.
    
    """
    x = np.arange(len(historic_state))
    y_vector = np.zeros(len(historic_state))

    for i in range(len(historic_state)):
        y = fct.energy(historic_state[i], weights)
        y_vector[i] = y

    plt.plot(x, y_vector, color='r')
    plt.title('Evolution of the Energy as a Function of Time.', loc='center', pad=15, color='black')
    plt.xlabel("time")
    plt.ylabel("energy", labelpad=0)
    plt.show()


def save_video(state_list, out_path): 
    """"
    Creates a video/gif which shows the evolution of the evolution of the states.
    
    Parameters
    ----------
    state_list : list of arrays
                 list of different states of the system
    out_path : str
               name of the path where video/gif is stored
    
    Returns
    -------
    None.
        
    """
    figure = plt.figure()
    images_state = []
    
    for state in state_list:
        images_state.append([plt.imshow(state, "gray")])
        
    ani = anim.ArtistAnimation(fig=figure, artists=images_state, interval=1000, repeat_delay=15)
    ani.save(out_path,fps=1.0, dpi=200)
    
   
def generate_checkerboard(patterns, pattern_num): # helper function
    """
    Generates 10x10 checkerboard 
    
    Parameters
    ----------
    patterns : array
               the initally memorized pattern
               
    pattern num : int
                  pattern that is going to be modified          
    
    Returns
    -------
    numpy array
        reshaped array
    
    """
    # Creating a checkerboard
    unit_matrix = np.ones((10, 10))  # this is the basic matrix which we will duplicate
    unit_matrix[5:19, 0:5] = -1
    unit_matrix[0:5, 5:10] = -1
    checkerboard = np.tile(unit_matrix, (5, 5))
    # Flattening the checkerboard in an array 
    flat_checkerboard = checkerboard.flatten()
    patterns[pattern_num] = flat_checkerboard
    return patterns

    
def visualization(patterns, historic_states, rule):
    """
    Generates a gif of the checkerboard representing the given patterns.
    
    Parameters
    ----------
    patterns : array
               memorized patterns
    historic_states : list
                      contains the historic states of a pattern
    rule : string 
           defines under which name the gif is going to be saved
                   
    Returns
    -------
    None.
    
    """
    sqrt = np.int(np.sqrt(len(patterns[0, :])))
    historic_states=[np.reshape(state, (sqrt, sqrt)) for state in historic_states]
        
    if (rule == 'hebbian_sync'):
        save_video(historic_states, "resultats/image_h_s.mp4")
    elif (rule == 'hebbian_async'):
        save_video(historic_states, "resultats/image_h_a.mp4")
    elif (rule == 'storkey_sync'):
        save_video(historic_states, "resultats/image_s_s.mp4")
    else:
        save_video(historic_states, "resultats/image_s_a.mp4");