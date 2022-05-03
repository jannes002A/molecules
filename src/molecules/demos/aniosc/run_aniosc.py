#!/bin/python

import jax.numpy as jnp

import molecules.models.aniosc as aniosc
import molecules.methods.verlet as ver
import molecules.visual.visual_aniosc as visual


def main():
    """ Main method for running a trajectory following the Ani Oscilator environment
    """

    # define parameters
    epsilon = 0.5  # anisotropy parameter
    q0 = jnp.array([2.0, 0.0])
    p0 = jnp.array([0.0, 0.01])
    mass = jnp.array([1.0, 1.0])

    # get environment check if it is workring
    env = aniosc.Aniosc(mass, epsilon)
    print(env.potential(q0))
    print(env.grad(q0))

    # get sampling method
    sampler = ver.verlet(env, q0, p0, 0.1)
    action = jnp.zeros(env.dim)

    # run sampling and print states
    position = []
    energy = []
    for i in range(100):
        state, _, eng, _ = sampler.step_eng()
        position.append(state)
        energy.append(eng)

    # visualize trajectory
    visual.visualize_trajectory(visual.convert2df(position))


if __name__ == '__main__':
    main()
