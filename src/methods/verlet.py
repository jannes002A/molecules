from jax import random
import jax.numpy as jnp
from gym import spaces


class verlet:
    """ Verlet algorithm for simulation of dynamical systems

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
    step_eng()
        calculates position and energy
    """

    def __init__(self, env, q0, p0,  dt, key=0):
        """
        Parameters
        ----------
        env : environment object
            environment of the dynamic system
        q0 : jnp.array
            starting position of the system
        p0: jnp.array
            starting momentum of the system
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
        self.momentum = p0
        #assert len(start) == self.dim[0], "Dimension missmatch"

        self.start = q0
        self.dt = dt
        self.state = jnp.zeros(self.dim)
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

        self.reset()

    def reset(self):
        """Set the start value of the sde

        set start value of position and brownian motion
        """
        self.state = jnp.array(self.start)
        return self.state

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
        # half momentum step
        mh = self.momentum + 0.5*self.dt * self.env.grad(self.state)

        # position step
        self.state += self.dt*mh*self.env.mass

        # evaluate energy
        pot = self.env.potential(self.state)

        # half momentum step
        self.momentum = mh + 0.5*self.dt*self.env.grad(self.state)

        # compute energy
        eng = pot + 0.5 * jnp.sum(self.momentum*self.momentum/self.env.mass)

        return self.state, self.momentum, eng, pot
