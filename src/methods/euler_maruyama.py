from jax import random
import jax.numpy as jnp
from gym import spaces


class Euler_maru:
    """Euler Maruyama descritization for sde simulation

    Attributes
    ----------
    env : environment object
        environment of the dynamic system
    dim : tuple
        dimension of the dynamic system
    name : str
        name of the simulation
    start : jnp.array
        start value of the position
    dt : float
        time step
    sigma : float
        inverse temperature
    dbt : jnp.array
        init Brownian motion vector
    state : jnp.array
        init start position
    key : int
        random key
    action_space : spaces.Box
        action space
    observation_space : spaces.Box
        position space

    Methods
    -------
    reset()
        reset state to start value
    reset_dbt()
        set dbt to 0
    step(action=jax array)
        calculates position, reward, done
    step_eng()
        calculates position and energy
    """

    def __init__(self, env, start, dt, key=0):
        """
        Parameters
        ----------
        env : environment object
            environment of the dynamic system
        start : jnp.array
            starting position of the system
        dt: float
            time step
        key: int
            random state

        """
        # environment
        self.env = env

        self.min_action = self.env.min_action
        self.max_action = self.env.max_action
        self.min_position = self.env.min_position
        self.max_position = self.env.max_position
        self.dim = self.env.dim
        self.name = self.env.name
        #assert len(start) == self.dim[0], "Dimension missmatch"

        self.start = start
        self.dt = dt
        self.beta = self.env.beta
        self.sigma = self.env.sigma
        self.dbt = jnp.zeros(self.dim)
        self.state = jnp.zeros(self.dim)
        self.seed = key
        self.key = random.PRNGKey(key)

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=self.dim,
            dtype=jnp.float32
        )
        self.observation_space = spaces.Box(
            low=self.min_position,
            high=self.max_position,
            shape=self.dim,
            dtype=jnp.float32
        )
        self.observation_space_dim = self.env.network_input
        self.action_space_dim = self.env.network_output
        self.reset()

    def reset(self):
        """Set the start value of the sde

        set start value of position and brownian motion
        """
        self.reset_dbt()
        self.state = jnp.array(self.start)
        return self.state

    def reset_dbt(self):
        """Set start value of brownian motion"""
        self.dbt = jnp.zeros(self.dim)
        return

    def step(self, action):
        """Performs one euler mayurama step

        Parameters
        ----------
        action : jax array
            action which should be applied to the system

        Returns
        -------
        state : jnp.array
            current state
        reward : float
            reward of the current position
        done : bool
            if target set is reached
        dbt : jnp.array
            current Brownian motion
        """
        # get key
        self.key, subkey = random.split(self.key)

        # reshape action
        action = jnp.reshape(action, self.dim)

        # compute drift and diffusion terms
        pot_grad = self.env.grad(self.state)
        self.dbt = jnp.sqrt(self.dt) * random.normal(subkey, shape=self.dim)

        # stochastic equation of motion
        self.state -= (pot_grad + action) * self.dt + self.sigma * self.dbt

        # is trajectory in hitting set ?
        done = self.env.criterion(self.state)

        # reward
        reward = -1/2 * action.flatten()@action.flatten()*self.dt - self.dt

        return self.state, reward, done, [self.dbt]

    def step_eng(self):
        """Evaluates position and energy of the system
        Function performs a time step of the EM algorithm
        and evaluates the potential at the new position

        Returns
        -------
        state : jnp.array
            position of the system
        momentum : float
            momentum of the system
        pot : flaat
            energy at current position
        pot : float
            computed energy of the current position
        """
        action = jnp.zeros(self.dim)
        state, _, _, _ = self.step(action)
        pot = self.env.potential(self.state)
        momentum = 0

        return state, momentum, pot, pot
