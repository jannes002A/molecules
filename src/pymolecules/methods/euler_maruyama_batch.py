from jax import random, vmap
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
    K: int
        number of trajectories
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
    is_in_hitting_set : jnp bool array
        True if the trajectory is (in the actial time step) in the hitting set
    been_in_hitting_set : jnp bool array
        True if the trajectory has already arrived in the hitting set
    new_in_hitting_set : jnp bool array
        True if the trajectory has now arrived in the hitting set
    idx_new_in_hitting_set : jnp int array
        array of indices of trajectories which now arrived in the hitting set
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
    """

    def __init__(self, env, start, dt, K, key=0):
        """
        Parameters
        ----------
        env: environment object
            environment of the underlying dynamical system
        start : jnp.array
            starting position of the system
        dt: float
            time step
        K: int
            number of trajectories
        key: int
            random state

        """
        self.env = env
        self.min_action = self.env.min_action
        self.max_action = self.env.max_action
        self.min_position = self.env.min_position
        self.max_position = self.env.max_position
        self.dim = self.env.dim
        self.name = self.env.name
        self.beta = self.env.beta
        self.sigma = self.env.sigma
        #assert len(start) == self.dim[0], "Dimension missmatch"

        self.K = K
        self.start = start
        self.dt = dt
        self.dbt = jnp.zeros((K,) + self.dim)
        self.state = jnp.zeros((K,) + self.dim)
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
        self.action_space_dim = self.env.network_output
        self.observation_space_dim = self.env.network_input
        self.reset()

    def reset(self):
        """Set the start value of the sde

        set start value of position and brownian motion
        """
        self.reset_dbt()
        self.state = jnp.array(self.start)
        self.been_in_hitting_set = jnp.zeros(self.K, dtype=bool)
        self.idx_new_in_hitting_set = jnp.array([], dtype=jnp.int32)
        return self.state

    def reset_dbt(self):
        """Set start value of brownian motion"""
        self.dbt = jnp.zeros((self.K,) + self.dim)
        return

    def update_been_in_hitting_set(self):
        """

        """
        # indices of trajectories new in the target set
        idx = jnp.where(
            (self.is_in_hitting_set == True) &
            (self.been_in_hitting_set == False)
        )[0]

        # update been in hitting set array
        if idx.shape[0] != 0:
            self.idx_new_in_hitting_set = idx
            self.been_in_hitting_set = self.been_in_hitting_set.at[idx].set(True)

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

        # reshape state and action
        state = jnp.reshape(self.state, (self.K,) + self.dim)
        action = jnp.reshape(action, (self.K,) + self.dim)

        # compute drift and diffusion terms
        pot_grad = self.env.grad_batch(state)
        self.dbt = jnp.sqrt(self.dt) * random.normal(subkey, shape=(self.K,)+self.dim)

        # stochastic equation of motion
        self.state -= (pot_grad + action) * self.dt + self.sigma * self.dbt

        # are trajectories in hitting set ?
        self.is_in_hitting_set = self.env.criterion_batch(self.state)
        self.update_been_in_hitting_set()
        done = self.been_in_hitting_set.all()

        # reward
        action_flat = action.reshape(self.K, self.env.dim_flat)
        reward = - 1 / 2 \
               * vmap(jnp.matmul, (0, 0), 0)(action_flat, action_flat) * self.dt \
               - self.dt

        return self.state, reward, done, [self.dbt, self.idx_new_in_hitting_set]
