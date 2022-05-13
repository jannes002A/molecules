from jax import random, vmap
import jax.numpy as jnp
from gym import spaces


class EulerMaru(object):
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
    states : jnp.array
        batch of states
    actions : jnp.array
        batch of actions
    is_in_hitting_set : jnp bool array
        True if the trajectory is (in the actial time step) in the hitting set
    been_in_hitting_set : jnp bool array
        True if the trajectory has already arrived in the hitting set
    new_in_hitting_set : jnp bool array
        True if the trajectory has now arrived in the hitting set
    idx_new_in_hitting_set : jnp int array
        array of indices of trajectories which now arrived in the hitting set
    seed : int
        seed to generate the random key
    action_space : spaces.Box
        action space
    observation_space : spaces.Box
        position space

    Methods
    -------
    reset()
        reset state to start value
    reset_states()
        set states to start
    reset_dbt()
        set dbt to 0
    update_been_in_hitting_set()
        update hitting time arrays
    step(actions=jax array)
        calculates position, reward, done
    """

    def __init__(self, env, start, dt, K, seed=0):
        """ init method

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
        seed: int
            seed to generate the random key

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
        self.states = jnp.zeros((K,) + self.dim)
        self.seed = seed
        self.key = random.PRNGKey(seed)

        # observation space and action space as gym boxes
        self.observation_space = spaces.Box(
            low=self.min_position,
            high=self.max_position,
            shape=self.dim,
            dtype=jnp.float32
        )
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=self.dim,
            dtype=jnp.float32
        )
        self.action_space_dim = self.env.network_output
        self.observation_space_dim = self.env.network_input

        # reset sampler
        self.reset()

    def reset(self):
        """ Set the start value of the sde and the brownian motion increments.
            Also preallocates the hitting set arrays and the control integral arrays.
        """

        # reset state and brownian increments 
        self.reset_states()
        self.reset_dbt()

        # preallocate hitting set arrays
        self.been_in_hitting_set = jnp.zeros(self.K, dtype=bool)
        self.idx_new_in_hitting_set = jnp.array([], dtype=jnp.int32)

        # preallocate control integral
        self.det_int_t = jnp.zeros(self.K)
        self.det_int_fht = jnp.empty(self.K)
        return self.states

    def reset_states(self):
        """ Set start value of the sde
        """
        self.states = jnp.array(self.start)

    def reset_dbt(self):
        """ Set start value of the brownian motion increments
        """
        self.dbt = jnp.zeros((self.K,) + self.dim)

    def update_been_in_hitting_set(self):
        """ update hitting time arrays
        """

        # indices of trajectories new in the target set
        self.idx_new = jnp.where(
            (self.is_in_hitting_set == True) &
            (self.been_in_hitting_set == False)
        )[0]

        # update been in hitting set array
        if self.idx_new.shape[0] != 0:
            self.been_in_hitting_set = self.been_in_hitting_set.at[self.idx_new].set(True)

    def step(self, actions):
        """ Performs one euler mayurama step for a batch of states and actions

        Parameters
        ----------
        actions : jax array
            batch of actions which should be applied to the system

        Returns
        -------
        states : jnp.array
            current batch of states
        rewards : float
            batch of reward of the current positions
        done : bool
            if target set is reached by all trajectories
        dbt : jnp.array
            current Brownian motion
        """
        # get key
        self.key, subkey = random.split(self.key)

        # reshape state and action
        states = jnp.reshape(self.states, (self.K,) + self.dim)
        actions = jnp.reshape(actions, (self.K,) + self.dim)

        # compute drift and diffusion terms
        pot_grad = self.env.grad_batch(states)
        self.dbt = jnp.sqrt(self.dt) * random.normal(subkey, shape=(self.K,)+self.dim)

        # stochastic equation of motion
        states += (- pot_grad + self.sigma * actions) * self.dt + self.sigma * self.dbt
        self.states = states

        # update control integral
        normed_actions = vmap(jnp.linalg.norm, (0,), 0)(actions)
        self.det_int_t += (normed_actions ** 2) * self.dt

        # are trajectories in hitting set ?
        self.is_in_hitting_set = self.env.criterion_batch(states)

        # update target set arrays
        self.update_been_in_hitting_set()

        # done flag
        done = self.been_in_hitting_set.all()

        # reward
        rewards = jnp.zeros(self.K)
        idx_never = jnp.where(self.been_in_hitting_set == False)[0]
        rewards = rewards.at[idx_never].set(
            - 0.5 * self.det_int_t[idx_never] * self.dt - self.dt
        )

        return states, rewards, done, [self.dbt, self.idx_new]
