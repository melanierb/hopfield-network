# BIO-210-team-19

Note : the ```README.md``` file must be read in light mode. To change github settings, go to [https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-user-account/managing-user-account-settings/managing-your-theme-settings](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-user-account/managing-user-account-settings/managing-your-theme-settings).

This python project simulates the evolution of a Hopfield network. The Hopfield network is a computational model for associative memory, it explains how a neural network can store representations. We implement an iterative process which allows to retrieve one of the stored patterns starting from the representation of a new pattern.

The different parts of the project are separated in different versions v<sub>i</sub>, where i is the number of the week.

## GIT INSTALLATION 
##### WINDOWS USERS

```
1. Go to: https://gitforwindows.org/
2. Download the latest version and start the installer
3. Follow the Git Setup wizard
4. Open the windows command prompt (or Git Bash)
5. Type: git --version -> to verify Git was installed
```
##### MAC USERS

```
1. Navigate to: https://sourceforge.net/projects/git-osx-installer/files/git-2.23.0-intel-universal-
mavericks.dmg/download
2. Download the latest version and start the installer
3. Follow the instructions
4. Open the command prompt "terminal" 
5. Type: git --version -> to verify Git was installed.
```

##### LINUX USERS

```
Git is already pre-installed on Linux, so you do not need to install it.
```


## ANACONDA INSTALLATION 
##### WINDOWS USERS

```
1. Follow the instructions of the following link : https://docs.anaconda.com/anaconda/install/windows/
2. Open Anaconda prompt
3. Type: python --version -> to verify python is installed
```
##### MAC USERS

```
1. Follow the instructions of the following link : https://docs.anaconda.com/anaconda/install/mac-os/
2. Open Anaconda prompt
3. Type: python --version -> to verify python is installed
```

##### LINUX USERS

```
1. Follow the instructions of the following link : https://docs.anaconda.com/anaconda/install/linux/
2. Open Anaconda prompt
3. Type: python --version -> to verify python is installed
```



## Important definitions 
- N : is the number of neurons, they're either firing (1) or non-firing (-1). 
- W : the Hopfield network is fully connected, that means that each neuron is connected to the other neurons via synapses. We store these connections in a weight matrix W, whose elements w<sub>ij</sub> ∈ [-1,1].
-  W is symmetric and no self connections are allowed (diagonal = 0)
- p, patterns : a Hopfield network can store a certain number of network firing patterns p<sup>μ</sup> ∈ {-1,1}, μ ∈ { 1,...,M}.
- M : generally used to represent the number of patterns. 

## Functions 
The functions are stored in a file [HopfieldNetwork.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/HopfieldNetwork.py) and imported in the [main.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/main.py) as ```fct```.
Each function contains a docstring explaining its purpose in the code. Here are a few important equations :
### Hebbian learning rule 
- The network connectivity is described as follows : 
<img src="https://render.githubusercontent.com/render/math?math=w_{i,j} = \frac{1}{M} \sum_{\mu=1}^{M} p_{i}^{\mu} p_{j}^{\mu} \quad i \ne j">


### Retrieving a memorized pattern
- The update rule is defined as follows : 
  
<img src="https://render.githubusercontent.com/render/math?math=p^{(t + 1)} = \sigma(Wp^{(t)})"> 
  
Where the function σ(x) is defined as :
  
 
<img src="https://render.githubusercontent.com/render/math?math=\sigma(x) = -1"> if x > 0
                  
and

<img src="https://render.githubusercontent.com/render/math?math=\sigma(x) = 1"> if x ≤ 0              


### Storkey rule
- The storkey rule is an alternative to the Hebbian learning rule : 

<img src="https://render.githubusercontent.com/render/math?math=w_{ij}^{\mu} = w_{ij}^{\mu-1} + \frac{1}{N}(p_{i}^{\mu}p_{j}^{\mu}-p_{i}^{\mu}h_{ji}^{\mu}-p_{j}^{\mu}h_{ij}^{\mu})">

where

<img src="https://render.githubusercontent.com/render/math?math=h_{ij}^{\mu} = \sum_{k : i \ne k \ne j} w_{ik}^{\mu-1}p_{k}^{\mu}">.

### Energy function
- A Hopfield network has an associated energy function E, where

<img src="https://render.githubusercontent.com/render/math?math=E(t) ≤ E(t'), t<t'">. 

- We define E as : 

<img src="https://render.githubusercontent.com/render/math?math=E = - \frac{1}{2} \sum_{i,j} w_{ij} p_{i} p_{j}">


## Visualization 
We visualize the code using the library ```matplotlib.pyplot```  which we import as ```plt```  and ```matplotlob.animation``` which we import as ```anim```.
We manually defined a pattern which looks like a checkerboard. To visualize it, we generate an mp4 file which we save in a directory [resultats](https://github.com/EPFL-BIO-210/BIO-210-team-19/tree/main/resultats). 

### Plot of the Energy Function
The code for the visualization of the energy function is in the ```time_energy(historic_state, weights)``` function of the [utils.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/utils.py) file. The energy is calculated at a regular interval and plotted using matplotlib.pyplot. Description of the function and parameters are written in the docstring of the function. 
Here is an example of an energy plot : 
```
import numpy as np
import new_HopfieldNetwork as fct
import new_utils as plot

def main():
    patterns = fct.generate_patterns(50, 2500)
    W = fct.hebbian_weights(patterns)
    patterns[0] = fct.perturb_patterns(patterns[0], 1000)
    hopfield_networks = patterns.copy()
    hopfield_networks[0, :] = fct.perturb_patterns(patterns[0, :], 1000)
    historic_state_heb = fct.dynamics(patterns[0], W, 20)
    
    ## Plot of energy function
    plot.time_energy(historic_state_heb, W)
   
main()
```
The output of this code is the following graph : 
<p align="center">
  <img src=plot_energy.png width="350">
</p>

### Plot of the Checkerboard 
The checkerboard is generated in a function ```generate_checkerboard(patterns, k)``` of [utils.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/utils.py). We also call the ```visualization(patterns, historic_modified_patterns, vo)``` function (see Docstring for more details).
The code the same as previously, with the addition of one line in the main function : ```plot.visualization(patterns, historic_state_heb, 'sto_synch')```.
The ouput looks something like the following gif :
<p align="center">
  <img src=results/image_h_a.gif width="350">
</p>

Note : we saved the images as ```.mp4```, for simplicity we put a gif in this file.
    
## Code testing 

We use [doctests](https://docs.python.org/3/library/doctest.html) and [pytest](https://docs.pytest.org/en/6.2.x/contents.html). 

### Doctests
We use the doctests for relatively simple functions.
Here is an example of a doctest in the [HopfieldNetwork.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/HopfieldNetwork.py) file. To test the sigma function, we write the following code in the docstring of the function ```sigma(Wp)```:
```
def sigma(Wp):
    """
    Examples
    --------
    >>> sigma(np.array([4,2.2,-0.8,-1,1,0.01,]))
    array([ 1.,  1., -1., -1.,  1.,  1.])
    >>> sigma(np.array([1,2,-2,4,1.2]))
    array([ 1.,  1., -1.,  1.,  1.])
    >>> sigma(np.array([]))
    array([], dtype=float64)
    >>> np.size(sigma(np.array([1,2,-2,4,1.2])))
    5
    
    """
```
We also add the following code at the end of the [HopfieldNetwork.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/HopfieldNetwork.py) file :
```
if __name__ == "__main__":
    import doctest
    
    doctest.testmod()
```
When running the [HopfieldNetwork.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/HopfieldNetwork.py) file, nothing will be printed as the tests passed.
Now if we change the last line of the docstring ```5``` to ```6```, we get the following error message : 
```
**********************************************************************
File "Desktop/HopfieldNetwork.py", line 34, in __main__.sigma
Failed example:
    np.size(sigma(np.array([1,2,-2,4,1.2])))
Expected:
    6
Got:
    5
**********************************************************************
1 items had failures:
   1 of   4 in __main__.sigma
***Test Failed*** 1 failures.
```
Note : we can also run the tests by entering the following command in the terminal : ```python HopfieldNetwork.py```. This will only print something if the tests fail, but to verify that every test passes, we can enter ```python HopfieldNetwork.py -v``` which will give us more details.

### Pytests
We use the pytests for more complex functions. To check the code coverage of our program we first download the ```coverage``` tool by entering the following command in the terminal : ```conda install -c anaconda coverage```.
We created a [test_HN.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/test_HN.py) file for the pytests. Here is an example of the syntax :
```
import doctest
import HopfieldNetwork as fct
import numpy as np

def test_HopfieldNetwork():
    """ integrating the doctests in the pytest framework """
    assert doctest.testmod(fct, raise_on_error=True)
    
def test_hebbian_weights_size():
    patterns = np.array([[1, 1, -1, -1], [1, 1, -1, 1], [-1, 1, -1, 1]])
    weights = fct.hebbian_weights(patterns)
    N = len(patterns[0, :])
    assert np.shape(weights) == np.shape(np.zeros((N, N)))
```
We can now run the following commands in the terminal : ```coverage run -m pytest``` followed by ```coverage report```.
This will print the following :
```
============================== 22 passed in 3.16s ==============================
(base) user@eduroam-4-200 BIO-210-team-19 % coverage report       
Name                 Stmts   Miss  Cover
----------------------------------------
HopfieldNetwork.py     103     0    100%
test_HN.py             8       0    100%
----------------------------------------
TOTAL                  111     0    100%
```

## Rewriting of code in oriented object
The code was rewritten in oriented object, files with oriented object code end in 'Class.py' (e.g. [mainClass.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/mainClass.py)).

## Capacity of the Hopfield Network
The capacity of the Hopfield network is assessed in a file [experiments.py](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/experiments.py). Further information of how to use the code is given in a markdown file [summary.md](https://github.com/EPFL-BIO-210/BIO-210-team-19/blob/main/summary.md).

## Summary of libraries :
- numpy : imported as np
- matplotlib.pyplot : imported as plt
- matplotlib.animation : imported as anim
- timeit : from timeit we import default_timer as timer 
- Cython 


## License
© 2021 GitHub, Inc.


EPFL © [Melanie Buechler](https://github.com/melanierb), [Maria Cherchouri](https://github.com/mariach13), [Renuka Singh Virk](https://github.com/renukasinghvirk)



