#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 16:54:17 2021

@author: renuka
"""

import numpy as np
import HopfieldNetwork as fct
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt


def capacity_curve(num_patterns, match_frac, n, rule):
    '''
    Plots the fraction of retrieved patterns vs number of patterns for a given number of neurons.
    
    Parameters
    ----------
    num_patterns : list
                   number of patterns
    match_frac : float
                 fraction of retrieved patterns
    n : integer
        number of neurons
    rule : string
           hebbian or storkey

    Returns
    -------
    None.

    '''
    fig = plt.figure()
    
    x = num_patterns
    y = match_frac
    plt.plot(x, y, color='b')

    # Title
    if rule == 'hebbian':
        plt.title("Fraction of retrieved patterns vs. nb of patterns for " + str(n) + " neurons. (Hebbian) ")
    else:
        plt.title("Fraction of retrieved patterns vs. nb of patterns for " + str(n) + " neurons. (Storkey) ")

    # Labels
    plt.xlabel("Number of patterns")
    plt.ylabel("Fraction of retrieved patterns")

    plt.show()


def emp_cap_curve(sizes, emp_capacity, theor_capacity, rule):
    '''
    Compares the experimental and theoretical capacity of different networks.
    
    Parameters
    ----------
    nsizes : list
             number of neurons
    emp_capacity : list
                   empirical capacities
    theor_capacity : list
                     theoretical capacities
    rule : string
           hebbian or storkey

    Returns
    -------
    None.

    '''
    fig = plt.figure()
    
    x = sizes
    y1 = emp_capacity
    y2 = theor_capacity
    
    plt.plot(x, y1, color = 'r', label = 'experimental capacity')
    plt.plot(x, y2, color = 'g', label = 'theoretcal capacity')

    # Title
    if rule == 'hebbian':
        plt.title(("Capacity vs. Number of neurons (Hebbian)"), loc='center', pad=15, color='black')
    else:
        plt.title(("Capacity vs. Number of neurons (Storkey)"), loc='center', pad=15, color='black')

    # Labels
    plt.xlabel("Number of Neurons")
    plt.ylabel("Capacity")

    plt.show()


def robust_curve(num_perturb, match_frac, n, rule):
    '''
    Plots the fraction of retrieved patterns vs number of perturbations.
    
    Parameters
    ----------
    num_perturb : list
                  number of perturbations
    match_frac : float
                 fraction of retrieved patterns
    n : integer
        number of neurons
    rule : string
           hebbian or storkey

    Returns
    -------
    None.

    '''
    fig = plt.figure()
    
    x = num_perturb
    y = match_frac
    plt.plot(x, y, color='g')

    # Title
    if rule == 'hebbian':
        plt.title(("Fraction of retrieved patterns vs. Number of perturbations for " + str(n) + " neurons. (Hebbian)"))
    else:
        plt.title(("Fraction of retrieved patterns vs. Number of perturbations for " + str(n) + " neurons. (Storkey)"))

    # Labels
    plt.xlabel("Number of perturbations [%]")
    plt.ylabel("Fraction of retrieved patterns")

    plt.show()
    
    
def max_perturb_curve(sizes, results, num_patterns):
    '''
    Plots the maximum perturbation in percentage of a network.
    
    Parameters
    ----------
    sizes : list of integers
            number of neurons
    results : list
              dictionaries with the results of the experiment
    num_patterns : integer
                   number of patterns

    Returns
    -------
    None.

    '''
    fig = plt.figure()
    
    x = sizes
    y = results
    plt.plot(x, y, color = 'r')
    
    # Title
    if rule == 'hebbian':
        plt.title(("Maximal perturbations for a network with" +str(num_patterns)+ "patterns. (Hebbian)"))
    else:
        plt.title(("Maximal perturbations for a network with" +str(num_patterns)+ "patterns. (Storkey)"))
    
    # Lables
    plt.xlabel("Maximal perturbations [%]")
    plt.ylabel("Number of neurons")
    
    plt.show()
    

def theoretical_cap(size, rule):
    '''
    Calculates the theoretical capacity of a network.

    Parameters
    ----------
    size : interger
           number of neurons
    rule : string
           either hebbian or storkey

    Returns
    -------
    float
        The capacity of the network.

    '''
    if rule == 'hebbian': 
        C = size / (2 * np.log(size))
    else:
        C = size / np.sqrt(2 * np.log(size))
    
    return C
    

def experiment(size, num_patterns, weight_rule, num_perturb, num_trials=10, max_iter=100):
    '''
    Creates a network and applies a number of trials on it to test if the patterns converge.

    Parameters
    ----------
    size : integer
           number of neurons.
    num_patterns : integer
                   number of patterns.
    weight_rule : string
                  either hebbian or storkey.
    num_perturb : integer
                  number of neurons to be perturbed.
    num_trials : integer, optional
                 number of trials for each network. The default is 10.
    max_iter : integer, optional
               Max iteration for the dynamical system. The default is 100.

    Returns
    -------
    results_dict : list of dictionaries
        Each dictionary contains the results for 1 network.

    '''
    match_frac = 0

    for trial in range(num_trials):

        # Creating a network
        network = fct.generate_patterns(num_patterns, size)

        # Computing the weight matrix
        if weight_rule == 'hebbian':
            weights = fct.hebbian_weights(network)
        else:
            weights = fct.storkey_weights(network)

        # Perturbing a pattern
        k = np.random.randint(0, num_patterns)
        perturbed_pattern = fct.perturb_patterns(network[k], num_perturb)

        # Running the dynamical system
        historic_states = fct.dynamics(perturbed_pattern, weights, max_iter)

        if np.allclose(historic_states[-1], network[k]):
            match_frac += 1

    match_frac /= num_trials
    results_dict = {'network_size': size, 'weight_rule': weight_rule, 'num_perturb': num_perturb,
                    'match_frac': match_frac}

    return results_dict


def capacity(sizes, rule):
    '''
    Calculates the empirical capacity of several networks and plots the results in a panda dataframe and in two graphs.

    Parameters
    ----------
    sizes : list of integers
            number of neurons
    rule : string
           either hebbian or storkey

    Returns
    -------
    None.

    '''
    emp_capacities = []
    theor_capacities = []

    for n in sizes:
        results = []
        match_frac = []
        max_capacity = 0  # Patterns that can be retrieved with >= 90% probability
            
        C = theoretical_cap(n, rule) # Calculate the theoretical capacity
        theor_capacities.append(C)
        
        num_patterns = np.linspace(0.5 * C, 2 * C, 10).astype(int)  # Calculate the numbers of patterns
        num_perturb = int(0.2 * n)  # 20% of the pattern size

        for t in num_patterns:
            result = experiment(n, int(t), rule, num_perturb)
            results.append(result)
            match_frac.append(result["match_frac"])

            if result["match_frac"] >= 0.9: 
                max_capacity = t

        emp_capacities.append(max_capacity)

        # Convert capacity results to a pandas dataframe
        df = pd.DataFrame(results)
        print(df.to_markdown())  # let pandas print the table in markdown format

        # Fraction of retrieved patterns vs. number of patterns
        capacity_curve(num_patterns, match_frac, n, rule)
    
    # Comparing empirical to theoretical capacity
    emp_cap_curve(sizes, emp_capacities, theor_capacities, rule)
    

def robustness(sizes, rule):
    '''
    Calculates the robustness of several networks and plots the results in a panda dataframe and in two graphs.

    Parameters
    ----------
    sizes : list of integers
            number of neurons
    rule : either hebbian or storkey

    Returns
    -------
    None.

    '''
    num_patterns = 2
    max_perturbs = []
    
    for n in sizes:
        results = []
        match_fracs = []
        percent_perturb = 0
        perturbations = np.linspace(0, 100, 20)
        
        result = experiment(n, num_patterns, rule, int(percent_perturb * n))

        while result['match_frac'] >= 0.9:
            percent_perturb += 5/100
            result = experiment(n, num_patterns, rule, int(percent_perturb * n))
        
        results.append(result)
        max_perturbs.append(percent_perturb * 100)
        percent_perturb = 0
        
        # Convert results to a panda dataframe
        df = pd.DataFrame(results)
        print(df.to_markdown())
        
        # Fraction of retrieved patterns vs. number of perturbations
        for i in range (0, 100, 5):
            fraction = experiment(n, num_patterns, rule, int(i/100 * n))
            match_fracs.append(fraction["match_frac"])

        robust_curve(perturbations, match_fracs, n, rule)
        
    # Computing the max percentage of perturbation
    max_perturb_curve(sizes, max_perturbs, num_patterns)


# Defining parameters
sizes = [10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500]  # Number of neurons
rule = 'hebbian'

# Capacity
capacity(sizes, rule)

# Robustness
robustness(sizes, rule)