from jax import random
import jax.numpy as jnp


class LJ7:
    """Computes forces and energy for a Anisotropic Oscillator model.
       Taken from MD.M AniOsc_force and AniOsc_init
       more details can be found here:
       Molecular Dynamics: with Deterministic and Stochastic Numerical Methods,
       B. Leimkuhler and C. Matthews,
       Springer-Verlag, 2015, ISBN 978-3-319-16374-1


    Attributes
    ----------
    mass : jax array
        mass vector for the particles
    N : int
        Number of particle
    dim : int -> tuple
        dimension of the underlying space
    sigma : float
        Lennard Jones length parameter
    ensc : float
        Lennard Jones energy scale
    gamma : float
        friction coefficient
    kT : float
        temperature
    name : str
        name

    Methods
    -------
    potential(q=jax.array)
        calculates the energy of the system for a given state

    grad(q=jax.array)
        calculates the force (derivative) of the energy for a given state
    """

    def __init__(self, mass=0, ):
        """
        Parameter
        ----------
        epsilon : float
            anisotropy parameter
        mass : jax array
            mass
        """
        self.min_action = 0
        self.max_action = 0
        self.min_position = 0
        self.max_position = 0
        self.beta = 0
        self.N = 7
        self.udim = 2
        self.dim = (self.N * self.udim,)
        self.sigma = 1.0
        self.ensc = 1.0
        self.gamma = 1.0
        self.kT = 0.2
        self.name = 'LJ7'

        if mass == 0:
            self.mass = jnp.ones(self.dim[0])

    def potential(self, state):
        """Calculates potential for state

        Parameter
        ----------
        state : jax array
            state of the system

        Returns
        -------
        U (float):
            potential at state
        """
        U=0
        for i in range(0,self.N-1):
            for j in range(i+1,self.N):
                rij = jnp.linalg.norm(state[2*i:2*i+2]-state[2*j:2*j+2])
                U += self.phi(rij)
        return U

    def grad(self, state):
        """Calculates gradient for 'state'

        Parameter
        ----------
        state: jax array
            state of the system

        Returns
        -------
        jax array N*dim x 1
            gradient at state
        """
        force = jnp.array([0.0]*self.dim[0])
        for i in range(0,self.N-1):
            for j in range(i+1,self.N):
                rij = jnp.linalg.norm(state[2*i:2*i+2]-state[2*j:2*j+2])
                fmul = -self.phiprime(rij)/rij
                fd = fmul * state[2*i:2*i+2]-state[2*j:2*j+2]
                force = force.at[2*i].add(fd[0])
                force = force.at[2*i+1].add(fd[1])
                force = force.at[2*j].add(-fd[0])
                force = force.at[2*j+1].add(-fd[1])

        return force

    def phi(self, r):
        return 4*self.ensc * ((self.sigma/r)**12 - (self.sigma/r)**6)

    def phiprime(self, r):
        return -24*self.ensc*(2*(self.sigma/r)**12/r - (self.sigma/r)**6/r)

    def criterion(self, state):
        """Checks if stopping criterion is reached.
        The criterion is not fixed at the moment and 'state' is not used.

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        bool
            criterion is satisfied
        """

        return False
