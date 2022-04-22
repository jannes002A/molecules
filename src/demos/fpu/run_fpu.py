"""script for running a simulation of Fermi-Pasta-Ulam
"""
import jax.numpy as jnp
import sys
sys.path.append('../../')
# define user defined classes
import models.femi_pasta_ulam as fpu
import methods.verlet as ver


def main():
    """Main method for sampling """
    # define position and momentum
    q0 = jnp.array(list(range(16)))
    p0 = [jnp.cos(10*2*jnp.pi*i/16) for i in range(16)]
    p0[15] = 2.0
    p0[0] = -2.0
    p0 = jnp.array(p0)

    # define environment
    env = fpu.Fpu()
    print(env.potential(q0))
    print(env.grad(q0))

    # define sampling method
    sampler = ver.verlet(env, q0, p0, 0.01)
    action = jnp.zeros(env.dim)

    # run simulation
    for i in range(100):
        print(sampler.step_eng())


if __name__ == '__main__':
    main()
