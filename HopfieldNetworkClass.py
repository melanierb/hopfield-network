#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 08:17:23 2021

@author: Meli
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim


class Patterns: # Helper class
    def __init__(self, num_patterns=50, pattern_size=2500, pattern_num=0):
        """
        Initializes Patterns.

        Parameters
        ----------
        num patterns : int
                       number of patterns to be generated
        pattern_size : int
                       size of the patterns

        Returns
        -------
        None.

        """
        self.patterns = self.generate_patterns(num_patterns, pattern_size)
        self.patterns = self.generate_checkerboard(self.patterns, pattern_num)

    def generate_patterns(self, num_patterns, pattern_size):
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

        """
        if type(num_patterns) != int or type(pattern_size) != int:
            raise ValueError("The given number of patterns or the pattern"
                             " size is not of the correct type, must be an"
                             " integer. (generate patterns)")

        return np.random.choice([-1, 1], size=(num_patterns, pattern_size))

    def generate_checkerboard(self, patterns, pattern_num):  # helper fmethod
        """
        Generates a 10x10 checkerboard.

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
        unit_matrix = np.ones((10, 10))  # basic matrix which is duplicated
        unit_matrix[5:19, 0:5] = -1
        unit_matrix[0:5, 5:10] = -1
        checkerboard = np.tile(unit_matrix, (5, 5))

        # Flattening the checkerboard in an array
        flat_checkerboard = checkerboard.flatten()
        patterns[pattern_num] = flat_checkerboard

        return patterns

    def get_pattern(self, pattern_num):  # helper method
        """
        Returns a specific pattern of the initialized patterns.

        Parameters
        ----------
        pattern num : int
                      pattern that is going to be returned

        Returns
        -------
        numpy array
            a specific pattern

        """
        return self.patterns[pattern_num]

    def get_patterns(self):  # helper method
        """
        Returns the initialized patterns.

        Returns
        -------
        numpy array
            initialized patterns

        """
        return self.patterns

    def get_patterns_len(self):  # helper method
        """
        Returns the length of the initialized patterns.

        Returns
        -------
        int
            length of the initialized patterns

        """
        return len(self.patterns[0, :])

    def get_perturbed_pattern(self):  # helper method
        """
        Returns the perturbed pattern.

        Returns
        -------
        numpy array
            the perturbed pattern

        """
        return self.perturbed_pattern

    def perturb_patterns(self, pattern, num_perturb):
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

        """
        if num_perturb > len(pattern):
            raise ValueError("The number of perturbations is bigger than the"
                             " number of neurons. (perturb patterns)")

        if type(num_perturb) != int:
            raise ValueError("The number of perturbations is of the wrong"
                             " type, it needs to be an integer."
                             " (perturb patterns")

        new_pattern = pattern.copy()
        new_neurons = np.random.choice(len(pattern), (num_perturb), replace=False)
        new_pattern[new_neurons] = -1 * new_pattern[new_neurons]

        self.perturbed_pattern = new_pattern

    def pattern_match(self, memorized_patterns, pattern):
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

        """
        if pattern.size != memorized_patterns[0].size:  # throw exception for incorrect pattern size
            raise ValueError("The given pattern does not have the correct size, it must be of size:",
                             len(memorized_patterns[0, :]), "(pattern match).")

        for i in range(len(memorized_patterns)):
            if (memorized_patterns[i, :] == pattern).all():
                return i

        return None


class HopfieldNetwork:
    def __init__(self, patterns, rule="hebbian"):
        """
        Initializes HopfieldNetwork.

        Parameters
        ----------
        patterns : array
                   memorized patterns
        rule : string
               defines which weight matrix should be calculated (hebbian or storkey)

        Returns
        -------
        None.

        """
        if rule == "hebbian":  # initalize weights according to rule
            self.weights = self.hebbian_weights(patterns)

        else:
            self.weights = self.storkey_weights(patterns)

    def hebbian_weights(self, patterns):
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
            raise ValueError("The given pattern is not of the correct type, "
                             "needs to be a numpy array (hebbian weights).")

        w = 0
        for i in range(len(patterns[:, 0])):
            w += np.outer(patterns[i, :], patterns[i, :])
        self.weights = w / len(patterns[:, 0])
        np.fill_diagonal(self.weights, 0)

        return self.weights

    def storkey_weights(self, patterns):
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
            raise ValueError("The given pattern is not of the correct type, "
                             "needs to be a numpy array (storkey weights).")

        N = np.size(patterns, axis=1)
        self.weights = np.zeros((N, N))
        H = np.zeros((N, N))

        for i, pattern in enumerate(patterns):
            W = self.weights.copy()
            p = np.array([pattern] * N).transpose()
            np.fill_diagonal(p, 0)
            np.fill_diagonal(W, 0)
            H = np.dot(W, p)
            pi = np.zeros(N).reshape(N, 1) + pattern
            pj = np.transpose(pi).copy()
            piHji = H * pi
            pjHij = np.transpose(H) * pj
            self.weights = self.weights + (1 / N) * (np.outer(pattern, pattern) - piHji - pjHij)

        return self.weights

    def sigma(self,Wp):  # helper function
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
            Wp[Wp >= 0] = 1
            Wp[Wp < 0] = -1
        else:  # if Wp is a value and not an array
            if Wp < 0:
                Wp = -1
            else:
                Wp = 1
        return Wp

    def update(self,state, weights):
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
        matrix_prod = np.matmul(self.weights, state)

        return self.sigma(matrix_prod)

    def update_async(self,state, weights):
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
        random_index = np.random.randint(0, len(self.weights[0]))  # Chooses a random integer between 0 and 1000.
        new_state = state.copy()
        new_state[random_index] = self.sigma(np.dot(state, self.weights[random_index]))
        return new_state, random_index

    def dynamics(self, state, saver, max_iter=20):
        """
        Applies the update function from an initial state until convergence or until a maximum
        number of steps is reached.

        Parameters
        ----------
        state : array
                state of a pattern
        saver : DataSaver
                DataSaver class
        max_iter: int
                  maximum number of times the update function is called.

        Returns
        -------
        None.

        """
        new_state = state.copy()  # array of 0, same size as state
        saver.store_iter(new_state, self.weights)

        counter_max_iter = 0
        convergence = 0  # false
        old_state = new_state.copy()

        while convergence < 2 and counter_max_iter < max_iter:
            old_state = new_state
            new_state = self.update(old_state, self.weights)
            saver.store_iter(new_state, self.weights)
            counter_max_iter += 1
            if (old_state == new_state).all():
                convergence += 1
            else:
                convergence = 0

        if convergence == 0:
            print("The states do not converge")
        else:
            print("The states converge in ", counter_max_iter, " iterations.")

        return saver

    def dynamics_async(self, state, saver, max_iter=30000,
                       convergence_num_iter=1000, skip=100):
        """
        Applies the asynchronous update function from an initial state until a maximum number of
        steps is reached.

        Parameters
        ----------
        state : array
                state of a pattern
        saver : DataSaver
                DataSaver calss
        max_iter : int
                   maximum number of times the update function is called
        convergence_num_iter : int
                               Steps in a row a solution should not change, to say that the algorithm has reached
                               convergence
        skip : int
               defines how many states should be skipped and therefore not saved

        Returns
        -------
        None.

        """
        new_state = state.copy()  # array of 0, same size as state
        old_state = np.zeros_like(state)
        counter_max_iter = 0
        counter_soft_conver = 0
        old_state = new_state.copy()
        saver.store_iter(new_state, self.weights)

        while counter_max_iter < max_iter and counter_soft_conver < convergence_num_iter:
            old_state = new_state
            transitory = self.update_async(old_state, self.weights)
            new_state = transitory[0]
            counter_max_iter += 1
            if (old_state[transitory[1]] == new_state[transitory[1]]).any():
                counter_soft_conver += 1
            else:
                counter_soft_conver = 0

            if (counter_max_iter % skip == 0):
                saver.store_iter(new_state, self.weights)

        return saver


class DataSaver:
    def __init__(self):
        """
        Initialized DataSaver.

        Returns
        ------
        None

        """
        self.data = {"States": [], "Energies": []}  # dictionary with states and corresponding energy

    def bool_conver(self):
        return (self.data["States"][-2] == self.data["States"][-1]).all()

    def bool_not_conver(self):
        return (self.data["States"][-2] != self.data["States"][-1]).all()

    def reset(self):
        """
        Resets the data.

        Returns
        -------
        None.

        """
        self.data.clear()

    def store_iter(self, state, weights):
        """
        Stores the states calculated by dynamics in the HopfieldNetwork and saves the corresponding energy.

        Parameters
        ----------
        state : array
                state of a pattern
        weights : array
                  weight matrix of the memorized patterns

        Returns
        -------
        None.

        """
        self.data["States"].append(state)
        self.data["Energies"].append(self.compute_energy(state.copy(), weights.copy()))

    def compute_energy(self, state, weights):
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

    def get_data(self):
        """
        Returns the data.

        Returns
        -------
        dictionary
            data

        """
        return self.data

    def save_video(self, out_path, img_shape):
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
        images_state.append([plt.imshow(self.data["States"], "gray")])
        ani = anim.ArtistAnimation(fig=figure, artists=images_state, interval=1000, repeat_delay=15)
        ani.save(out_path, fps=1.0, dpi=200)

    def plot_energy(self):
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
        integer or an array of integer

        """
        x = np.arange(len(self.data["States"]))# length of dictionary

        plt.plot(x, self.data["Energies"], color='r')
        plt.title('Evolution of the Energy as a Function of Time.', loc='center', pad=15, color='black')
        plt.xlabel("time")
        plt.ylabel("energy", labelpad=0)
        plt.show()

    def visualization(self, Patterns, rule="hebbian_sync"):
        """
        Generates a gif of the checkerboard representing the given patterns.

        Parameters
        ----------
        Patterns : Patterns
                   Patterns class
        rule : string
               defines under which name the gif is going to be saved

        Returns
        -------
        None.

        """
        sqrt = np.int(np.sqrt(Patterns.get_patterns_len()))

        for state in self.data["States"]:
            historic_modified_patterns = [np.reshape(state, (sqrt, sqrt))]
        if (rule == "hebbian_sync"):
            self.save_video(historic_modified_patterns, "resultats/image_h_s.mp4")
        elif (rule == "hebbian_async"):
            self.save_video(historic_modified_patterns, "resultats/image_h_a.mp4")
        elif (rule == "storkey_sync"):
            self.save_video(historic_modified_patterns, "resultats/image_s_s.mp4")
        else:
            self.save_video(historic_modified_patterns, "resultats/image_s_a.mp4");