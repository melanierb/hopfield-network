import doctest
import HopfieldNetwork as fct
import numpy as np


def test_HopfieldNetwork():
    """ integrating the doctests in the pytest framework """
    assert doctest.testmod(fct, raise_on_error=True)


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
patterns_bench = fct.generate_patterns(num_patterns, network_size)
num_perturbations = 1000
max_iter_synch = 20
max_iter_async = 30000


# Testing perturb_patterns
def test_perturb_patterns():
    import pytest
    pattern = np.array([1, 1, -1, -1])
    with pytest.raises(ValueError):
        fct.perturb_patterns(pattern, 5)


def test_perturb_patterns_wrongType():
    import pytest
    pattern = np.array([1, 1, -1, -1])
    with pytest.raises(ValueError):
        fct.perturb_patterns(pattern, 3.2)


# Testing pattern_match
def test_pattern_match():
    import pytest
    pattern = np.array([1, 1, -1, -1, 1])
    with pytest.raises(ValueError):
        fct.pattern_match(patterns, pattern)


# Testing hebbian_weights
def test_hebbian_weights(benchmark):
    weights = benchmark.pedantic(fct.hebbian_weights, args=(patterns_bench,))
    assert np.allclose(fct.hebbian_weights(patterns), hebbian_weights)
    

def test_hebbian_weights_wrongType():
    import pytest
    pattern = [-1, 1, -1, 1]
    with pytest.raises(ValueError):
        fct.hebbian_weights(pattern)


def test_hebbian_weights_size():
    weights = fct.hebbian_weights(patterns)
    N = len(patterns[0, :])
    assert np.shape(weights) == np.shape(np.zeros((N, N)))


def test_hebbian_weights_valueRange():
    weights = fct.hebbian_weights(patterns)
    assert (weights <= 1).all() and (weights >= -1).all()


def test_hebbian_weights_symmetric():
    weights = fct.hebbian_weights(patterns)
    assert (weights == np.transpose(weights)).all()


def test_hebbian_weights_diagonal():
    weights = fct.hebbian_weights(patterns)
    assert (weights.diagonal()).all() == 0


# Testing storkey_weights
def test_storkey_weights(benchmark):
    weights = benchmark.pedantic(fct.storkey_weights, args=(patterns_bench,))
    assert np.allclose(fct.storkey_weights(patterns), storkey_weights)


def test_storkey_weights_wrongType():
    import pytest
    pattern = [-1, 1, -1, 1]
    with pytest.raises(ValueError):
        fct.storkey_weights(pattern)


def test_storkey_weights_size():
    #patterns = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    weights = fct.storkey_weights(patterns)
    N = len(patterns[0, :])
    assert np.shape(weights) == np.shape(np.zeros((N, N)))


def test_storkey_weights_type():
    #patterns = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    weights = fct.storkey_weights(patterns)
    assert weights.dtype == float


# Testing update function with hebbian weights
def test_update_hebbianLength():
    new_state = fct.update(perturbed_pattern, hebbian_weights)
    assert len(new_state) == len(pattern)


def test_update_hebbianType():
    new_state = fct.update(perturbed_pattern, hebbian_weights)
    assert type(new_state) == type(pattern)


def test_update_hebbianRange():
    new_state = fct.update(perturbed_pattern, hebbian_weights)
    assert (new_state <= 1).all() and (new_state >= -1).all()


# Testing update with storkey weigths
def test_update_storkeyLength():
    new_state = fct.update(perturbed_pattern, storkey_weights)
    assert len(new_state) == len(pattern)


def test_update_strokeyType():
    new_state = fct.update(perturbed_pattern, storkey_weights)
    assert type(new_state) == type(pattern)


def test_update_storkeyRange():
    new_state = fct.update(perturbed_pattern, storkey_weights)
    assert (new_state <= 1).all() and (new_state >= -1).all()


# Running update for 100 steps
def test_update_profiling():
    for i in range(99):
        fct.update(perturbed_pattern, hebbian_weights)


def test_update_benchmark(benchmark):
    benchmark.pedantic(test_update_profiling, args=())


# Running update_async for 100 steps
def test_update_async_profiling():
    for i in range(99):
        fct.update_async(perturbed_pattern, hebbian_weights)
        

def test_update_async_benchmark(benchmark):
    benchmark.pedantic(test_update_async_profiling, args=())


# Testing dynamics with hebbian weights
def test_dynamics_converHebbian():
    historic_states = fct.dynamics(perturbed_pattern, hebbian_weights, 1000)
    assert (historic_states[-1] == historic_states[-2]).all()


def test_dynamics_notConverHebbian():
    perturbed_pattern = np.array([-1, -1, -1, -1])
    historic_states = fct.dynamics(perturbed_pattern, hebbian_weights, 1000)
    assert (historic_states[-1] != historic_states[-2]).all()


# Testing dynamics with storkey weights
def test_dynamics_converStorkey():
    historic_states = fct.dynamics(perturbed_pattern, storkey_weights, 1000)
    assert (historic_states[-1] == historic_states[-2]).all()


# Testing dynamics_async with hebbian weights
def test_dynamics_async_converHebbian():
    historic_states = fct.dynamics_async(perturbed_pattern, hebbian_weights, 100000, 100000)
    assert (historic_states[-1] == historic_states[-2]).all()


# Testing dynamics_async with storkey weights
def test_dynamics_async_converStorkey():
    historic_states = fct.dynamics_async(perturbed_pattern, storkey_weights, 100000, 100000)
    assert (historic_states[-1] == historic_states[-2]).all()


# Testing energy with hebbian weights
def test_energy_hebbian():
    energies = []
    for i in range(len(historic_states)):
        energies.append(fct.energy(historic_states[i], hebbian_weights))
    assert energies[0] >= energies[1] >= energies[2] >= energies[3]


def test_energy_hebbian_benchmark(benchmark):
    benchmark.pedantic(test_energy_hebbian, args=())


def test_energy_hebbian_len():
    energies = []
    for i in range(len(historic_states)):
        energies.append(fct.energy(historic_states[i], hebbian_weights))
    assert len(energies) == len(historic_states)


# Testing energy with storkey weights
def test_energy_storkey():
    energies = []
    for i in range(len(historic_states)):
        energies.append(fct.energy(historic_states[i], storkey_weights))
    assert energies[0] >= energies[1] >= energies[2] >= energies[3]


def test_energy_storkey_len():
    energies = []
    for i in range(len(historic_states)):
        energies.append(fct.energy(historic_states[i], storkey_weights))
    assert len(energies) == len(historic_states)