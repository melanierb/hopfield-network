#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 16:37:21 2021

@author: Meli
"""

import HopfieldNetworkClass as fct
from update_cython import update
from update_cython import update_async
import numpy as np

# Definition of parameters
patterns = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
pattern = np.array([1, 1, -1, -1])
perturbed_pattern = np.array([-1, 1, -1, -1])
hebbian_weights = np.array([[0., 0.33333333, -0.33333333, -0.33333333], [0.33333333, 0., -1, 0.33333333],
                            [-0.33333333, -1, 0., -0.33333333], [-0.33333333, 0.33333333, -0.33333333, 0.]])
storkey_weights = np.array([[1.125, 0.25, -0.25, -0.5], [0.25, 0.625, -1., 0.25], [-0.25, -1., 0.625, -0.25],
                            [-0.5, 0.25, -0.25, 1.125]])
historic_states = [np.array([-1, 1, -1, -1]), np.array([1., 1., -1., 1.]), np.array([1., 1., -1., 1.]),
                   np.array([1., 1., -1., 1.])]

# Definition of parameters for benchmark
num_patterns = 50
network_size = 2500
num_perturbations = 1000
max_iter_synch = 20
max_iter_async = 30000
pattern_num = 0

# Initializing the classes
Patterns = fct.Patterns(num_patterns, network_size, pattern_num)
HN = fct.HopfieldNetwork(patterns, "hebbian")
saver = fct.DataSaver()

# Definition of parameters for benchmark
patterns_bench = Patterns.get_patterns()


# Testing perturb_patterns
def test_perturb_patterns():
    import pytest
    pattern = np.array([1, 1, -1, -1])
    with pytest.raises(ValueError):
        Patterns.perturb_patterns(pattern, 5)


def test_perturb_patterns_wrongType():
    import pytest
    pattern = np.array([1, 1, -1, -1])
    with pytest.raises(ValueError):
        Patterns.perturb_patterns(pattern, 3.2)


# Testing pattern_match
def test_pattern_match():
    import pytest
    pattern = np.array([1, 1, -1, -1, 1])
    with pytest.raises(ValueError):
        Patterns.pattern_match(patterns, pattern)


# Testing hebbian_weights
def test_hebbian_weights(benchmark):
    weights = benchmark.pedantic(HN.hebbian_weights, args=(patterns_bench,))
    assert np.allclose(HN.hebbian_weights(patterns), hebbian_weights)


def test_hebbian_weights_wrongType():
    import pytest
    pattern = [-1, 1, -1, 1]
    with pytest.raises(ValueError):
        HN.hebbian_weights(pattern)


def test_hebbian_weights_size():
    weights = HN.hebbian_weights(patterns)
    N = len(patterns[0,:])
    assert np.shape(weights) == np.shape(np.zeros((N,N)))


def test_hebbian_weights_valueRange():
    weights = HN.hebbian_weights(patterns)
    assert (weights <= 1).all() and (weights >= -1).all()


def test_hebbian_weights_symmetric():
    weights = HN.hebbian_weights(patterns)
    assert (weights == np.transpose(weights)).all()


def test_hebbian_weights_diagonal():
    weights = HN.hebbian_weights(patterns)
    assert (weights.diagonal()).all() == 0


# Testing storkey_weights
def test_storkey_weights(benchmark):
    weights = benchmark.pedantic(HN.storkey_weights, args=(patterns_bench,))
    assert np.allclose(HN.storkey_weights(patterns), storkey_weights)


def test_storkey_weights_wrongType():
    import pytest
    pattern = [-1, 1, -1, 1]
    with pytest.raises(ValueError):
        HN.storkey_weights(pattern)


def test_storkey_weights_size():
    # patterns = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    weights = HN.storkey_weights(patterns)
    N = len(patterns[0, :])
    assert np.shape(weights) == np.shape(np.zeros((N, N)))


def test_storkey_weights_type():
    # patterns = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    weights = HN.storkey_weights(patterns)
    assert weights.dtype == float


# Testing update function with hebbian weights
def test_update_hebbianLength():
    new_state = update(perturbed_pattern, hebbian_weights)
    assert len(new_state) == len(pattern)


def test_update_hebbianType():
    new_state = update(perturbed_pattern, hebbian_weights)
    assert type(new_state) == type(pattern)


def test_update_hebbianRange():
    new_state = update(perturbed_pattern, hebbian_weights)
    assert (new_state <= 1).all() and (new_state >= -1).all()


# Testing update with storkey weigths
def test_update_storkeyLength():
    new_state = update(perturbed_pattern, storkey_weights)
    assert len(new_state) == len(pattern)


def test_update_strokeyType():
    new_state = update(perturbed_pattern, storkey_weights)
    assert type(new_state) == type(pattern)


def test_update_storkeyRange():
    new_state = update(perturbed_pattern, storkey_weights)
    assert (new_state <= 1).all() and (new_state >= -1).all()


# Running update for 100 steps
def test_update_profiling():
    for i in range(99):
        update(perturbed_pattern, hebbian_weights)


def test_update_benchmark(benchmark):
    benchmark.pedantic(test_update_profiling, args=())


# Running update_async for 100 steps
def test_update_async_profiling():
    for i in range(99):
        update_async(perturbed_pattern, hebbian_weights)


def test_update_async_benchmark(benchmark):
    benchmark.pedantic(test_update_async_profiling, args=())


# Testing dynamics with hebbian weights
def test_dynamics_converHebbian():
    historic = HN.dynamics(perturbed_pattern,saver,1000)
    assert (historic.bool_conver)


def test_dynamics_notConverHebbian():
    perturbed_pattern = np.array([-1,-1,-1,-1])
    historic = HN.dynamics(perturbed_pattern,saver,1000)
    assert (historic.bool_not_conver)

# Testing dynamics with storkey weights
def test_dynamics_converStorkey():
    historic = HN.dynamics(perturbed_pattern,saver,1000)
    assert(historic.bool_conver)

# Testing dynamics_async with hebbian weights
def test_dynamics_async_converHebbian():
    historic = HN.dynamics_async(perturbed_pattern,saver,100000,100000)
    assert (historic.bool_conver)


# Testing dynamics_async with storkey weights
def test_dynamics_async_converStorkey():
    historic =HN.dynamics_async(perturbed_pattern,saver,100000,100000)
    assert (historic.bool_conver)

# Testing energy with hebbian weights
def test_energy_hebbian():
    energies = []
    for i in range(len(historic_states)):
        energies.append(saver.compute_energy(historic_states[i], hebbian_weights))
    assert energies[0] >= energies[1] >= energies[2] >= energies[3]


def test_energy_hebbian_benchmark(benchmark):
    benchmark.pedantic(test_energy_hebbian, args=())


def test_energy_hebbian_len():
    energies = []
    for i in range(len(historic_states)):
        energies.append(saver.compute_energy(historic_states[i], hebbian_weights))
    assert len(energies) == len(historic_states)


# Testing energy with storkey weights
def test_energy_storkey():
    energies = []
    for i in range(len(historic_states)):
        energies.append(saver.compute_energy(historic_states[i], storkey_weights))
    assert energies[0] >= energies[1] >= energies[2] >= energies[3]


def test_energy_storkey_len():
    energies = []
    for i in range(len(historic_states)):
        energies.append(saver.compute_energy(historic_states[i], storkey_weights))
    assert len(energies) == len(historic_states)
