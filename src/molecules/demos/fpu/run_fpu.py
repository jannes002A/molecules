#!/bin/python

import jax.numpy as jnp

import molecules.models.femi_pasta_ulam as fpu
import molecules.methods.verlet as ver

def main():
    """ Main method for running a trajectory following the Fermi-Pasta-Ulam environment
    """

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
    sampler = ver.Verlet(env, q0, p0, 0.01)
    action = jnp.zeros(env.dim)

    # run simulation
    for i in range(100):
        print(sampler.step_eng())


if __name__ == '__main__':
    main()
