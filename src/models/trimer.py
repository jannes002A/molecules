import jax.numpy as jnp


class Trimer:
    """Computes forces and energy for a Trimer model.
    Taken from MD.M Trimer_force and Trimer_init
    more details can be found here:
    Molecular Dynamics: with Deterministic and Stochastic Numerical Methods,
    B. Leimkuhler and C. Matthews,
    Springer-Verlag, 2015, ISBN 978-3-319-16374-1

    Attributes
    ----------
    mass : jax array
        mass vector for the particles
    n : int
        number of particles
    dim : tuple (2,)
        dimension of the underlying space
    sigma : float
        Lennard Jones length parameter
    eps : float
        Lennard Jones energy scale
    gamma : float
        friction coefficient (Langevin dynamics)
    kT : float
        temperature
    name : str
        name

    Methods
    -------
    potential(q=jax array)
        calculates the energy of the system for a given state
    grad(q=jax array)
        calculates the force (derivative) of the energy for a given state
    """

    def __init__(self, mass):
        """
        mass : jax array
            masses
        """
        self.min_action = 0
        self.max_action = 0
        self.min_position = 0
        self.max_position = 0
        self.n = 1
        self.dim = (2,)
        self.sigma = 1.0
        self.eps = 1.0
        self.gamma = 1.0
        self.kT = 3.0
        self.mass = mass
        self.name = 'Timer_' + str(self.kT)  # name

    def potential(self, q):
        """Calculates potential for state q

        Parameters
        ----------
        q : jax array
            state of the system

        Returns
        -------
        float
            potential at q
        """
        r = jnp.linalg.norm(q)
        return 2*self.phi(r)+self.phi(2*q[0])

    def grad(self, q):
        """Calculates derivative of potential for state q

        Parameters
        ----------
        q : jax array
            state of the system

        Returns
        -------
        jax array 2 x 1
            gradient at q
        """
        r = jnp.linalg.norm(q)
        f1 = -2.0 * (self.phiprime(r)/r)*q[0] - 2.0 * self.phiprime(2*q[0])
        f2 = -2.0 * (self.phiprime(r)/r) * q[1]
        return jnp.array([f1, f2])

    def phi(self, r):
        """Calculates pair potential

        Parameters
        ----------
        r : float
            state of the system

        Returns
        -------
        float
            pair potential
        """
        return 4.0*self.eps*((self.sigma/r)**12 - (self.sigma/r)**6)

    def phiprime(self, r):
        """Calculates derivative of pair potential

        Parameters
        ----------
        r : float
            state of the system

        Returns
        -------
        float
            derivative pair potential
        """
        return -24.0 * self.eps*(2.0 * ((self.sigma/r)**12)/r - ((self.sigma/r)**6)/r)

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
