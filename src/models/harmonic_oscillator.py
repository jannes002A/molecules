import jax.numpy as jnp


class harm_osci:
    """Computes forces and energy for a Harmonic Oscillator model.
       Taken from MD.M HO_force a HO_init
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
    gamma : float
        friction coefficient
    kT : float
        temperature
    h : float
        step size
    name : str
        name

    Methods
    -------
    potential(state=jax.array)
        calculates the energy of the system for a given state

    grad(state=jax.array)
        calculates the force (derivative) of the energy for a given state
    """

    def __init__(self, mass, k=1.0):
        """
        Parameter
        ----------
        k : float
            harmonic oscillator coefficient (defautl is 1.0)
        mass : jax array
            mass
        """
        self.min_action = 0
        self.max_action = 0
        self.min_position = 0
        self.max_position = 0
        self.N = 1
        self.mu = 1.0
        self.gamma = 1.0
        self.kT = 1
        self.h = 0.01
        self.dim = (1,)
        self.beta = 1.0
        self.sigma = 1.0
        self.k = k
        self.mass = mass
        self.network_input = self.dim[0]
        self.network_output = self.dim[0]
        self.name = 'aniosc_' + str(self.k)

    def potential(self, state):
        """Calculates potential for state q

        Parameter
        ----------
        q : jax array
            state of the system

        Returns
        -------
        float
            potential at state
        """

        return self.phi(state)

    def grad(self, state):
        """Calculates derivative for 'state'

        Parameter
        ----------
        q: jax array
            state of the system

        Returns
        -------
        jax array n x 1
            gradient at state
        """
        return -self.phiprime(state)

    def phi(self, r):
        """Calculate the value of phi for input r

        Parameter
        ----------
        r : jax array
            state of the system

        Returns
        -------
        float
            phi
        """
        return 0.5*self.k*r*r

    def phiprime(self, r):
        """Calculate the derivative of phi for input r

        Parameter
        ----------
        r : jax array
            state of the system

        Returns
        -------
        float
            phiprime
        """
        return self.k*r

    def criterion(self, state):
        "Placeholder NOTIMPLEMENTED"
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
        return False

