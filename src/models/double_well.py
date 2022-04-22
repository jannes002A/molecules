import jax.numpy as jnp
from jax import random, jit, vmap
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags


class DoubleWell(gym.Env):
    """environment for a sde in a double well potential
    the goal is to reach the set defined by lb and rb
    the sde follows
    dx_t = -(\nabla V(x_t)+u_t)dt + \sqrt(1\beta^-1)dB_t

    Attributes
    ----------
    stop : list
        hitting set
    dim : int
        dimension of the sde
    beta : float (default 4.0)
                inverse temperature
    alpha : np.array (default [1.0])
        heights of the barrier
    sigma : float
        simulation parameter
    min_action : float
        minimal possible action
    max_action : float
        maximal possible action
    min_position : float
        minimal possible position
    max_position : float
        maximal possible position
    low_state : np.array
        lowest position and action
    high_state : np.array
        highest position and action
    observation_space_shape : int
        check where this is used
    action_space : spaces.Box
        action space
    observation_space : spaces.Box
        observation space

    Methods
    -------
    potential(state : jax array)
        calculates the energy of the system for a given state
    potential_batch(states_batch : jax array)
        calculates the energy of the system for a batch of states
    grad(state : jax array)
        calculates the force (derivative) of the energy for a given state
    grad_batch(states_batch : jax array)
        calculates the force (derivative) of the energy for a batch of states
    criterion(state : jax array):
        check if process is in hitting set
    criterion_batch(states_batch : jax array):
        check if states are in hitting set
    get_solution(n_points : int, plots : bool)
        plot solution for 1d case

    """
    def __init__(self, stop, dim, beta=4.0, alpha=[1.0]):
        """
        Parameters
        ----------
        stop : list(floats)
            hitting set defined by the hypercube [stop, max_position)^d
        dim : tuple
            dimension
        beta : float
            inverse temperature
        alpha : list(floats)
            heights of the barrier
        """
        self.min_action = -10.0
        self.max_action = 10.0
        self.min_position = -2.5
        self.max_position = 2.5
        self.dim = (dim,)
        self.dim_flat = dim
        if len(alpha) == dim:
            self.alpha = jnp.array(alpha)
        else:
            self.alpha = jnp.array(alpha * dim)

        if len(stop) == dim:
            self.stop = jnp.array(stop)
        else:
            self.stop = jnp.array(stop * dim)

        self.beta = beta
        self.sigma = jnp.sqrt(2.0 / self.beta)

        self.low_state = np.array([self.min_position,
                                   self.min_action], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_action],
                                   dtype=np.float32)

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(2,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.min_position,
            high=self.max_position,
            shape=(2,),
            dtype=np.float32
        )
        self.network_input = self.dim[0]
        self.network_output = self.dim[0]
        self.name = 'SDE_Double_Well_' + str(self.beta) + '_'+str(self.dim[0])

    def potential(self, state):
        """Calculates potential for state

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        pot: float
            potential at state
        """
        return jnp.sum(self.alpha * (state**2 - 1) ** 2)

    #@jit
    def potential_batch(self, states_batch):
        """Calculates potential for batch of states

        Parameters
        ----------
        states_batch : jax array
            batch of K states of the system

        Returns
        -------
        pot: jax array (K,)
            potential at each state
        """
        return vmap(self.potential, in_axes=0, out_axes=0)(states_batch)

    def grad(self, state):
        """Calculates the gradient of the potential evaluated at the given state

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        grad: jnp array (d,)
            grad at state
        """
        return 4 * self.alpha * state * (state ** 2 - 1)

    def grad_batch(self, states_batch):
        """Calculates the gradient of the potential evaluated at the given batch of states

        Parameters
        ----------
        states_batch : jax array
            batch of K states of the system

        Returns
        -------
        grad: jnp array (K, d)
            grad at state
        """
        return vmap(self.grad, in_axes=0, out_axes=0)(states_batch)

    def criterion(self, state):
        """Calculates if state is in the hitting set

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        bool
            if set is hit
        """
        return (state > jnp.array(self.stop)).all()

    def criterion_batch(self, states_batch):
        """Calculates if states in batch are in the hitting set

        Parameters
        ----------
        states_batch : jax array
            batch of K states of the system

        Returns
        -------
        is_in_hitting_set: jax array of bools
            true if it is in the hitting set
        """
        return vmap(self.criterion, in_axes=0, out_axes=0)(states_batch)

    def get_solution(self, n_points=1000, plots=True):
        """Solves HJB and visualizes the results

        Parameter
        ----------
        n_points : int
            grid size

        plots bools
            True plots are shown

        Returns
        -------
            plots
        """
        bl = self.min_position  # bound left
        br = self.max_position  # bound right
        # dx = 0.01 # discretization
        # n_points = int(np.ceil((br-bl)/dx)) # number of points
        x_space = np.linspace(bl, br, n_points)  # domain
        dx = x_space[1] - x_space[0]
        beta = self.beta
        grad_V = lambda x: 2 * x * (x ** 2 - 1)
        Lj = np.zeros([n_points, n_points])
        f = np.ones(n_points)
        b = np.zeros([n_points, 1])
        bounds = [1.0, 1.1]

        # build the generator of the BVP
        # Hauptdiagonale \beta^-1 \nabla^2 \psi -f \psi
        # Nebendigonale \nabla V \nabla \spi
        Lj = 1 / (beta * dx ** 2) * diags([1, -2, 1], [-1, 0, 1], shape=(n_points, n_points)) - diags(f) + np.dot(
            -1 * diags(grad_V(x_space)), 1 / (2 * dx) * diags([-1, 0, 1], [-1, 0, 1], shape=(n_points, n_points)))
        # define the hitting set
        hit_set_index = np.argwhere((x_space > bounds[0]) & (x_space < bounds[1]))
        for item in hit_set_index:
            b[item] = 1  # exp(g(x)) mit g = 0
            Lj[item, :] = 0
            Lj[item, item] = 1
        # numerical stability
        # L[0, :] = 0
        Lj[0, 0] = -Lj[0, 1]
        # L[0, 1] = -1
        # b[0] = - 1*grad_V(x_space[0])*1/(2*dx)

        # L[n_points-1, :] = 0
        Lj[n_points - 1, n_points - 1] = -Lj[n_points - 1, n_points - 2]
        # L[n_points-1, n_points-2] = -1
        # b[n_points-1] = 1*grad_V(x_space[-1])*1/(2*dx)

        psi = spsolve(Lj, b)
        u_pde = -2 / (beta * dx) * (np.log(psi[1:]) - np.log(psi[:-1]))

        if plots:
            fig = plt.figure(figsize=(12, 10))
            ax1 = fig.add_subplot(311)
            ax1.plot(x_space, psi)
            ax1.set_title('Value function', fontsize=16)

            ax2 = fig.add_subplot(312)
            ax2.plot(x_space[1:], u_pde)
            ax2.set_title('Biasing potential', fontsize=16)

            ax3 = fig.add_subplot(313)
            ax3.plot(x_space[1:], -grad_V(x_space[1:]) - u_pde)
            ax3.set_title('-Grad V + u', fontsize=16)

        return u_pde, psi
