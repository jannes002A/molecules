import jax.numpy as jnp


class Fpu:
    """Computes forces and energy for a Fermi-Pasta-Ulam model.
    Taken from MD.M FPU_force and FPU_init
    more details can be found here:
    Molecular Dynamics: with Deterministic and Stochastic Numerical Methods,
    B. Leimkuhler and C. Matthews,
    Springer-Verlag, 2015, ISBN 978-3-319-16374-1

    Attributes
    ----------
    n : int
        number of particles
    dim : tuple
        dimension of the underlying space
    kappa : float
        harmonic coefficient
    theta : float
        cubic coefficient
    mu : float
        quartic coefficient
    L : int
        length of the periodic chain
    gamma : float
    friction coefficient (Langevin dynamics)
    kT : float
        temperature
    name : str
        'Timer_' + str(self.kT)  # name

    Methods
    -------
    potential(q=jax array)
        calculates the energy of the system for a given state
    grad(q=jax array)
        calculates the force (derivative) of the energy for a given state
    """

    def __init__(self, mass=[1.0]):
        """
        Parameters
        ----------
        mass: jax array
            n x 1  mass

        """
        self.n = 16
        self.dim = (1,)
        self.kappa = 10.0
        self.theta = 0.0
        self.mu = 0.4
        self.gamma = 1.0
        self.kT = 1.0
        self.mass = jnp.array(mass*self.n)
        self.L = self.n
        self.name = 'fpu'

    def potential(self, state):
        """Calculates potential for state q

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        energy float
            potential at state
        """
        energy = 0
        for i in range(self.n-1):
            energy += self.phi(abs(state[i]-state[i+1]))
        energy += self.phi(self.L+state[0]-state[self.n-1])
        return energy

    def grad(self, state):
        """Calculates derivative of potential for state

        Parameters
        ----------
        state : jax array
            state of the system

        Returns
        -------
        jax array n x 1
            gradient at state
        """
        force = [0]*self.n
        for i in range(self.n-1):
            fmul = -self.phiprime(abs(state[i]-state[i+1]))/abs(state[i]-state[i+1])
            fd = fmul*(state[i]-state[i+1])
            force[i] += fd
            force[i+1] += fd

        fmul = -self.phiprime(abs(self.L+state[0]-state[self.n-1]))/abs(self.L+state[0]-state[self.n-1])
        fd = fmul*(self.L+state[0]-state[self.n-1])
        force[self.n-1] -= fd
        force[0] += fd

        return jnp.array(force)

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
        return self.kappa * r * r/2.0 + self.theta*r*r*r/3.0 + self.mu * r * r * r * r/4.0

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
        return self.kappa*r + self.theta*r*r + self.mu*r*r*r

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


