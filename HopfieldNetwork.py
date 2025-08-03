"""
Module containing simulation code of the Hopfield Network.
"""

import numpy as np


def generate_patterns(num_patterns, pattern_size):
    """
    Generates an array of patterns with random values between {-1,1}.

    Parameters
    ----------
    num_patterns : int
                   number of patterns to be generated
    pattern_size : int
                   size of the patterns

    Returns
    -------
    numpy array
        The generated patterns.

    Examples
    --------
    >>> np.size(generate_patterns(3, 50))
    150
    >>> np.shape(generate_patterns(4,5))
    (4, 5)
    >>> generate_patterns(2, 3) >= -1
    array([[ True,  True,  True],
           [ True,  True,  True]])
    >>> generate_patterns(2, 3) <= 1
    array([[ True,  True,  True],
           [ True,  True,  True]])

    """
    return np.random.choice([-1, 1], size=(num_patterns, pattern_size))


def perturb_patterns(pattern, num_perturb):
    """
    Samples a number of elements (num_perturb) of the input pattern (pattern) uniformly and at random and
    changes their sign.

    Parameters
    ----------
    pattern : array
              the memorized pattern
    num_perturb : int
                  number of elements in the pattern to be perturbated

    Returns
    -------
    numpy array
        Perturbed patterns

    Examples
    -------
    >>> perturb_patterns(np.array([1,-1,1,-1,1,-1]), 6)
    array([-1,  1, -1,  1, -1,  1])
    >>> np.shape(perturb_patterns(np.array([1,-1,-1,1,-1]), 4))
    (5,)
    >>> perturb_patterns(np.array([1,-1,-1,1,1,-1]), 6) >= -1
    array([ True,  True,  True,  True,  True,  True])

    """
    if num_perturb > len(pattern):
        raise ValueError("The number of perturbations is bigger than the number of neurons. (perturb patterns)")

    if type(num_perturb) != int:
        raise ValueError("The number of perturbations is of the wrong type, it needs to be an integer. "
                         "(perturb patterns)")

    new_pattern = pattern.copy()
    new_neurons = np.random.choice(len(pattern), (num_perturb), replace=False)
    new_pattern[new_neurons] = -1 * new_pattern[new_neurons]

    return new_pattern


def pattern_match(memorized_patterns, pattern):
    """
    Checks if a pattern matches any of the memorized patterns.

    Parameters
    ----------
    memorized_patterns : array
                         the initially memorized patterns
    pattern : array
              the perturbated pattern

    Returns
    -------
    None
        If no memorized pattern matches.
    int
        Index of the row corresponding to the matching pattern.

    Examples
    -------
    >>> pattern_match(np.array([[-1,-1,1,-1,1,-1], [1,-1,1,-1,1,-1]]),np.array([-1,-1,1,-1,1,-1]))
    0
    >>> pattern_match(np.array([[-1,1,1,1,1,1], [1,1,1,1,1,-1], [1,1,-1,1,1,-1]]),np.array([-1,-1,1,-1,-1,-1]))

    """
    if pattern.size != memorized_patterns[0].size:  # throw exception for incorrect pattern size
        raise ValueError("The given pattern does not have the correct size, it must be of size:",
                         len(memorized_patterns[0, :]), "(pattern match).")

    for i in range(len(memorized_patterns)):
        if (memorized_patterns[i, :] == pattern).all():
            return i

    return None


def hebbian_weights(patterns):
    """
    Applies the hebbian learning rule on given patterns to create the weight matrix.

    Parameters
    ----------
    patterns : array
               the initally memorized pattern

    Returns
    -------
    numpy array
        Weight matrix of the given patterns.

    """
    if type(patterns) != np.ndarray:
        raise ValueError("The given pattern is not of the correct type, needs to be a numpy array (hebbian weights).")

    w = 0
    for i in range(len(patterns[:, 0])):
        w += np.outer(patterns[i, :], patterns[i, :])

    W = w / len(patterns[:, 0])
    np.fill_diagonal(W, 0)
    return W


def storkey_weights(patterns):
    """
    Applies the Storkey learning rule to train the weights of the Hopfield network.

    Parameters
    ----------
    patterns : array
               initially memorized patterns

    Returns
    -------
    numpy array
        Weight matrix of the trained weights of the patterns.

    """
    if type(patterns) != np.ndarray:
        raise ValueError("The given pattern is not of the correct type, needs to be a numpy array (storkey weights).")

    N = np.size(patterns, axis=1)
    weight = np.zeros((N, N))
    H = np.zeros((N, N))

    for i, pattern in enumerate(patterns):
        W = weight.copy()
        p = np.array([pattern] * N).transpose()
        np.fill_diagonal(p, 0)
        np.fill_diagonal(W, 0)
        H = np.dot(W, p)
        pi = np.zeros(N).reshape(N, 1) + pattern
        pj = np.transpose(pi).copy()
        piHji = H * pi
        pjHij = np.transpose(H) * pj
        weight = weight + (1 / N) * (np.outer(pattern, pattern) - piHji - pjHij)

    return weight


def sigma(Wp):  # helper function
    """
    Implements the sign function.

    Parameters
    ----------
    Wp : array or integer
         matrix multiplication of the weights and pattern or one element of this matrix

    Returns
    -------
    integer or an array of integer

    Examples
    -------
    >>> sigma(np.array([4,2.2,-0.8,-1,1,0.01,]))
    array([ 1.,  1., -1., -1.,  1.,  1.])
    >>> sigma(np.array([1,2,-2,4,1.2]))
    array([ 1.,  1., -1.,  1.,  1.])
    >>> sigma(np.array([]))
    array([], dtype=float64)
    >>> np.size(sigma(np.array([1,2,-2,4,1.2])))
    5

    """
    if type(Wp) == type(np.array([])):
        Wp[Wp >= 0] = 1  # all the values in the array that are equal or bigger than 0
        Wp[Wp < 0] = -1  # all the values in the array that are smaller than 0
    else:  # if Wp is a value and not an array
        if Wp < 0:
            Wp = -1
        else:
            Wp = 1
    return Wp


def update(state, weights):
    """
    Applies the update rule to a given state pattern.

    Parameters
    ----------
    state : array
            state of a pattern
    weights : array
              weight matrix of the memorized patterns

    Returns
    -------
    numpy array
        New state of the given state.

    """
    matrix_prod = np.matmul(weights, state)

    return sigma(matrix_prod)


def update_async(state, weights):
    """
    Applies the asynchronous update rule to a given state pattern.

    Parameters
    ----------
    state : array
            state of a pattern
    weights : array
              weight matrix of the memorized patterns

    Returns
    -------
    numpy array
        New state of the given state.

    """
    random_index = np.random.randint(0, len(weights[0]))  # Chooses a random integer between 0 and 1000.
    new_state = state.copy()
    new_state[random_index] = sigma(np.dot(state, weights[random_index]))
    return new_state, random_index


def dynamics(state, weights, max_iter):
    """
    Applies the update function from an initial state until convergence or until a maximum
    number of steps is reached.

    Parameters
    ----------
    state : array
            state of a pattern
    weights : array
              weight matrix of the memorized patterns
    max_iter: int
              maximum number of times the update function is called

    Returns
    -------
    list
        Whole state history until convergence.

    """
    new_state = state.copy()  # array of 0, same size as state
    history_states = [new_state]
    counter_max_iter = 0
    convergence = 0  # false
    old_state = new_state.copy()

    while convergence < 2 and counter_max_iter < max_iter:
        old_state = new_state
        new_state = update(old_state, weights)
        history_states.append(new_state)
        counter_max_iter += 1
        if (old_state == new_state).all():
            convergence += 1
        else:
            convergence = 0

    return history_states


def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """
    Applies the asynchronous update function from an initial state until a maximum number of
    steps is reached.

    Parameters
    ----------
    state : array
            state of a pattern
    weights : array
              weight matrix of the memorized patterns
    max_iter : int
               maximum number of times the update function is called
    convergence_num_iter : int
                           Steps in a row a solution should not change, to say that the algorithm has reached
                           convergence

    Returns
    -------
    list
        Whole state history until convergence.

    """
    new_state = state.copy()  # array of 0, same size as state
    old_state = np.zeros_like(state)
    history_states = [new_state]
    counter_max_iter = 0
    counter_soft_conver = 0
    old_state = new_state.copy()

    while counter_max_iter < max_iter and counter_soft_conver < convergence_num_iter:
        old_state = new_state
        transitory = update_async(old_state, weights)
        new_state = transitory[0]
        counter_max_iter += 1
        if (old_state[transitory[1]] == new_state[transitory[1]]).any():
            counter_soft_conver += 1
        else:
            counter_soft_conver = 0
        if (counter_max_iter % 1000 == 0):
            history_states.append(new_state)

    return history_states


def energy(state, weights):
    """
    Applies the energy function to a state.

    Parameters
    ----------
    state : array
            state of a pattern
    weights : array
              weight matrix of the memorized patterns

    Returns
    -------
    float
        Energy of a state.

    """
    return -np.sum(weights * np.outer(state, state))