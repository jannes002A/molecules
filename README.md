# Reinforcement Learning for Molecular Dynamics
This project aim to use a reinforcement learing approach to solve sampling problems for metastable dynamical systems. The project catins two main packages. The algorithm package contains some state of the art reinforcement learning algorihtms (see below for more info). The environment package contains differnt dynmical system which can be simulated with different time evolution methods (see also below for more info)

## Algorihtms

This folder contains differen state of the art reinforcement learning algorithms. At the moment there are
	
- REINFORCE
- Actor Critic
- Cross Entropy
- DDPG

A demo for an low dimensional SDE environment can be found under 
```
"algorihtm*"/src/run_"alg"_sde.py
```

## Environments
In the environments folder different dynmical models can be found as well as different time integration methods

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

You can run demos for the different models by going into the /environments/demos/ and run the specific run file for e.g. in the SDE case
```
python run_sde_eng.py
```

#### SDE Double Well
The SDE environment samples a double well potential in a choosen dimension. Here is a 10d examples
![SDE in 10 dimension](/environments/demos/pics/sde10.png) 


#### AniOsc
This model runs an anisotropic oscillator model. 
![10000 Time Steps of AniOsc](/environments/demos/pics/aniosc_demo_10000.png) 

### Butane
With this env you can run butane trajectories.

![Small butane trajectory](/environments/demos/pics/butane.gif ) 

[^1]: Examples from: 
	MD.M: Matlab MD code  
 	Molecular Dynamics: with Deterministic and Stochastic Numerical Methods,
 	B. Leimkuhler and C. Matthews,
 	Springer-Verlag, 2015, ISBN 978-3-319-16374-1
[^2]: Example from 
	https://sourceforge.net/projects/trajlab/
