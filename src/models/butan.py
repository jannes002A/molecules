from jax import grad, random, jit, vmap
import jax.numpy as jnp
import jax.numpy.linalg as jnpl
from gym import spaces
import numpy as np
import torch as T

key = random.PRNGKey(0)

class Butan:
    """
    - environment for a backbone simulation of butane
    - the code is taken from a trajlab example

    Attributes
    ----------
    stop : list
        hitting set

    beta : float (default 4.0)
        inverse temperature

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

    get_angle(state : jax array):
        calculates angle between the two CCC planes for a given state

    get_angle_batch(states_batch : jax array):
        calculates angle between the two CCC planes for a batch of states

    criterion(state : jax array):
        check if process is in hitting set

    criterion_batch(states_batch : jax array):
        check if states are in hitting set


    """
    def __init__(self, stop, beta):
        """
        Parameters
        ----------
        min_action : float min possible action
        max_action : float max possible action
        min_position : float min possible position
        max_position : float max possible position
        alpha : list(floats) heights of the barrier
        stop : list(float) hitting set
        beta : float inverse temperature
        sigma : float simulation parameter
        dim : tuple dimension

        low_state : np.array lowest position and action
        high_state : np.array highest position and action

        observation_space_shape : int check where this is used

        action_space : spaces.Box  action space
        observation_space : spaces.Box observation space
        name : str name

        """
        self.simulation_ndim     = 3
        self.simulation_periodic = False
        self.simulation_verbose  = False

        self.intra_handle = 'pot_CHARMM'

        self.param_kr   = 222.5 * 418.4
        self.param_ka   = 58.0 * 4.184
        self.param_r0   = 0.153
        self.param_a0   = 115 * np.pi/180
        self.param_kd   = 8.3129
        self.param_sig  = 0.38 / 2**(1/6)
        self.param_eps = 1.01 * 4.184

        self.molecule_type      = 'Butan'
        self.molecule_nmol      = 1
        self.molecule_natom     = 4
        self.molecule_atomtype  = ['C', 'C', 'C', 'C']
        self.molecule_mass      = [15, 14, 14, 15],
        self.molecule_geometry  =  0.153 * jnp.array([
            [-jnp.sin(115 * jnp.pi / 180), - jnp.cos(115 * jnp.pi / 180), 0],
            [0, 0, 0],
            [1, 0, 0],
            [(1 + jnp.sin(115 *jnp.pi / 180)), jnp.cos(115 * jnp.pi/180), 0]
        ])

        self.dim = (self.simulation_ndim, self.molecule_natom)
        self.dim_flat = self.simulation_ndim * self.molecule_natom
        self.min_action = -10.0
        self.max_action = 10.0
        self.min_position = -10.0
        self.max_position = 10.0

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(self.molecule_natom*self.simulation_ndim,),
            dtype=jnp.float32
        )

        self.observation_space = spaces.Box(
            low=self.min_position,
            high=self.max_position,
            shape=(self.molecule_natom*self.simulation_ndim,),
            dtype=jnp.float32
        )
        self.network_output = self.molecule_natom * self.simulation_ndim
        self.network_input = self.molecule_natom * self.simulation_ndim
        self.beta = beta
        self.sigma = jnp.sqrt(2.0 / self.beta)
        self.stop = stop

        self.name = 'Butan_' + str(stop) + '_' + str(beta)

    def potential(self, state):
        """Calculates potential for state

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        u float
            potential at state
        """

        # initialize potential
        u = 0

        # get C atoms
        r1 = state[:, 0] #CH3
        r2 = state[:, 1] #CH2
        r3 = state[:, 2] #CH2
        r4 = state[:, 3] #CH3

        # Bond vectors
        r12 = r1 - r2
        r23 = r2 - r3
        r34 = r3 - r4

        # Harmonic model for all bond length
        l12 = jnpl.norm(r12)
        l23 = jnpl.norm(r23)
        l34 = jnpl.norm(r34)
        u += self.param_kr * (l12 - self.param_r0)**2
        u += self.param_kr * (l23 - self.param_r0)**2
        u += self.param_kr * (l34 - self.param_r0)**2

        # unit vectors
        e12 = r12 / l12
        e23 = r23 / l23
        e34 = r34 / l34

        # harmoic model for angle bending
        c123 = jnp.round(-e12@e23, 5)
        c234 = jnp.round(-e23@e34, 5)
        a123 = jnp.arccos(c123)
        a234 = jnp.arccos(c234)

        u += self.param_ka * (a123 - self.param_a0)**2
        u += self.param_ka * (a234 - self.param_a0)**2

        # construct vector perpendicular to 123 and 234
        v123 = jnp.cross(e12, e23)
        v234 = jnp.cross(e23, e34)

        # Get dihedral angle, see http://en.wikipedia.org/wiki/Dihedral_angle
        # Result should be inside (-pi,pi]
        a1234 = jnp.arctan2( e12@v234, v123@v234)

        # Trigonometric model for dihedral angle bending
        # u = u + params.k_d * (1+cos(params.n_d * a1234 - params.d_0)) ;
        u += self.param_kd * self.pot_ryck_bell(a1234)

        # Lennard-Jones interaction between termini
        r = jnpl.norm(r1 - r4)
        sr6 = (self.param_sig/r)**6
        u += 4 * self.param_eps * sr6 * (sr6 - 1)

        return u

    def potential_batch(self, states_batch):
        """Calculates potential for a batch of states

        Parameters
        ----------
        states_batch : jax array
            batch of states of the system

        Returns
        -------
        pot: jax array (K,)
            potential at each state
        """
        return vmap(self.potential, in_axes=0, out_axes=0)(states_batch)

    def pot_ryck_bell(self, tau):
        """Calculates Potential Ryck Bell

        Parameters
        ----------
        tau: float
            potential parameter

        Returns
        -------
        float
            potential
        """
        tau = tau - jnp.pi
        ct = jnp.cos(tau)
        return jnp.polyval(jnp.array([-3.778, 3.156, -0.368, -1.578, 1.462, 1.116]), ct)

    def grad(self, state):
        """Calculates gradient of potential with autodiff from jax

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        jnp array
            gradient of potential
        """
        grad_pot = grad(self.potential)

        return grad_pot(state)

    def grad_batch(self, states_batch):
        """Calculates gradient of potential with autodiff from jax for a batch of states

        Parameters
        ----------
        states_batch : jax array
            batch of states of the system

        Returns
        -------
        pot: jax array (K,)
            gradient of potential
        """
        return vmap(self.grad, in_axes=0, out_axes=0)(states_batch)

    def get_angle(self, state) -> float:
        """Calculates angle between the two CCC planes

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        float
            angle
        """
        state = jnp.reshape(state, [3,4] )
        r1 = state[:, 0]
        r2 = state[:, 1]
        r3 = state[:, 2]
        r4 = state[:, 3]

        # plane 1
        r12 = r1 - r2
        r13 = r1 - r3

        # plane 2
        r23 = r2 - r3
        r24 = r2 - r4

        # compute arc cos of the angle between the two planes
        n1 = jnp.cross(r12, r13)
        nn1 = jnpl.norm(n1)
        n1 /= nn1

        n2 = jnp.cross(r23, r24)
        nn2 = jnpl.norm(n2)
        n2 /= nn2

        acos = abs(n1@n2)/(jnpl.norm(n1)*jnpl.norm(n2))

        # impose max value for acos
        acos = jnp.where(acos > 1.0, 1.0, acos)

        # compute angle
        angle = jnp.arccos(acos)/jnp.pi * 180

        return angle

    def get_angle_batch(self, states_batch) -> float:
        """Calculates angle between the two CCC planes for a batch of states

        Parameters
        ----------
        states_batch : jax array
            batch of states of the system

        Returns
        -------
        angles: jax array (K,)
            angle for each state
        """
        return vmap(self.get_angle, in_axes=0, out_axes=0)(states_batch)

    def criterion(self, state):
        """Checks if stopping criterion is reached

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        bool
            criterion is satisfied
        """

        angle = self.get_angle(state)

        return jnp.where(angle > self.stop, True, False)

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

    def shaping(self):
        #TODO:vecorize the dynamical system for simulation and use a function for reshaping it
        pass
