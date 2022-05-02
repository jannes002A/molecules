import jax.numpy as jnp


class Aniosc:
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
    epsilon : float
        anisotropy parameter
    N : int
        Number of particle
    dim : int -> tuple
        dimension of the underlying space
    kappa0 : float
        spring coefficient
    l0 : float
        extension scalling
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
    potential(q=jax.array)
        calculates the energy of the system for a given state

    grad(q=jax.array)
        calculates the force (derivative) of the energy for a given state
    """

    def __init__(self, mass, epsilon):
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
        self.N = 1
        self.kappa0 = 1.0
        self.l0 = 1.0
        self.gamma = 1.0
        self.kT = 0.06
        self.h = 0.05
        self.dim = (2,)

        self.epsilon = epsilon
        self.mass = mass
        self.network_input = self.dim[0]
        self.network_output = self.dim[0]
        self.name = 'aniosc_' + str(self.epsilon)

    def potential(self, q):
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
        r = jnp.linalg.norm(q)
        phi = 0.5*self.kappa(self.c3(q, r))*(r-self.l(self.c3(q, r)))*(r-self.l(self.c3(q, r)))
        return phi

    def grad(self, q):
        """Calculates potential for state q

        Parameter
        ----------
        q: jax array
            state of the system

        Returns
        -------
        jax array n x 1
            gradient at state
        """
        r = jnp.linalg.norm(q)
        return jnp.array([-self.phix(q, r), -self.phiy(q, r)])

    def kappa(self, c3):
        """Calculate kappa

        Parameter
        ----------
        c3 : jax array
            state of the system

        Returns
        -------
        float
            kappa
        """
        return self.kappa0*(1.0 - 0.5*self.epsilon*c3)

    def l(self, c3):
        """Calculates l

        Parameter
        ----------
        c3 : jax array
            state of the system

        Returns
        -------
        float
            l
        """
        return self.l0*(1.0 + 0.5*self.epsilon*c3)

    def kappaprime(self):
        """Calculates derivative of kappa

        Parameter
        ----------
        c3 : jax array
            state of the system

        Returns
        -------
        float
            kappaprime
        """
        return -0.5*self.kappa0*self.epsilon

    def lprime(self):
        """Calculates derivative l

        Returns
        -------
        float
            lprime
        """
        return 0.5*self.l0*self.epsilon

    def c3(self, q, r):
        """Calculates c3

        Parameters
        ----------
        q : jax array
            state of the system

        r : jax array
            state of the system

        Returns
        -------
        float
            c3
        """
        c = q[0]/r
        return 4*c*c*c-3*c

    def c3x(self, q, r):
        """Calculates derivative in x of c3

        Parameters
        ----------
        q : jax array
            state of the system

        r : jax array
            state of the system

        Returns
        -------
        float
            c3x
        """
        c = q[0]/r
        cx = (r-q[0]*q[0]/r) / (r*r)
        return (4*3*c*c)*cx

    def c3y(self, q, r):
        """Calculates derivative in y of c3

        Parameters
        ----------
        q : jax array
            state of the system

        r : jax array
            state of the system

        Returns
        -------
        float
            c3y
        """
        c = q[0]/r
        cy = -q[0]*q[1] / (r*r*r)
        return (4*3*c*c-3)*cy

    def phix(self, q, r):
        """Calculates derivative in x of potential

        Parameters
        ----------
        q : jax array
            state of the system

        r : float
            norm of q

        Returns
        -------
        float
            d/dx phi
        """
        y = 0.5 * self.kappaprime() * self.c3x(q, r) * (r - self.l(self.c3(q, r))) * (r - self.l(self.c3(q, r)))
        y += self.kappa(self.c3(q, r)) * (r - self.l(self.c3(q, r))) * ((q[0] / r) - self.lprime() * self.c3x(q, r))
        return y

    def phiy(self, q, r):
        """Calculates derivative in y of potential

        Parameters
        ----------
        q : jax array
            state of the system

        r : float
            norm of q

        Returns
        -------
        float
            d/dy phi
        """
        y = 0.5 * self.kappaprime() * self.c3y(q, r) * (r - self.l(self.c3(q, r),)) * (r - self.l(self.c3(q, r)))
        y += self.kappa(self.c3(q, r)) * (r - self.l(self.c3(q, r))) * ((q[1] / r) - self.lprime() * self.c3y(q, r))
        return y

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
