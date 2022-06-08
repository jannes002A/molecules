# Molecular Dynamics Simulations
This project aims to provide a framework for sampling molecular dynamical systems. The package offers different dynamical systems which can be simulated with different time evolution methods (see also below for more info).

## Environments
In the environments folder different dynmical models can be found as well as different time integration methods. The gradient of the different resulting potential is computed by using atomatic differentiation.

### Methods
This folder contains different time integration methods which can be combined with a dynamical system which can be found in the model folder.
At the moment one can find

- Verlet
- Euler-Maruyama

### Models
In this folder one can find different initializations for differen dynamical systems. At the moment there are

- aniosc[^1]
- butane[^2]
- fpu[^1]
- sde
- timer[^1]
- LJ7[^1]

## Installation

1. clone the repo 
```
$ git clone git@github.com:jannes002A/molecules.git
```

2. set python version (>=3.7)
```
$ pyenv local 3.9.7
```

3. create virtual environment and install required packages
```
$ make venv
```

4. activate venv
```
$ source venv/bin/activate
```

## Usage
You can run demos for the different models by running a specific script in the "/src/molecules/demos/" folder for e.g. in the SDE case
```
$ python src/molecules/demos/sde/run_sde_eng.py
```

#### SDE Double Well
The SDE environment samples a double well potential in a choosen dimension. Here is a 10d examples
![SDE in 10 dimension](/src/molecules/demos/pics/sde10.png) 


#### AniOsc
This model runs an anisotropic oscillator model. 
![10000 Time Steps of AniOsc](/src/molecules/demos/pics/aniosc_demo_10000.png) 

#### Butane
With this env you can run butane trajectories.

![Small butane trajectory](/src/molecules/demos/pics/butane.gif ) 

[^1]: Examples from: 
	MD.M: Matlab MD code  
 	Molecular Dynamics: with Deterministic and Stochastic Numerical Methods,
 	B. Leimkuhler and C. Matthews,
 	Springer-Verlag, 2015, ISBN 978-3-319-16374-1
[^2]: Example from 
	https://sourceforge.net/projects/trajlab/
