#!/bin/python

from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

import molecules.models.Lennard_Jones_7 as lj7
import molecules.methods.verlet as ver
import molecules.visual.visual_butan as visual


def main():
    """ Main method for running a trajectory following the LennardJones environment
    """

    # define parameters
    pos = [0] * 14
    for i in range(1, 7):
        pos[2 * i] = 2 ** (1 / 6) * jnp.cos(i * jnp.pi / 3).item()
        pos[2 * i + 1] = 2 ** (1 / 6) * jnp.sin(i * jnp.pi / 3).item()

    state = jnp.array(pos)

    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    momentum = 0.8 * random.normal(subkey, shape=(14,))

    pbarx = 0
    pbary = 0

    for i in range(0, 7):
        pbarx += momentum[2 * i].item()
        pbary += momentum[2 * i + 1].item()

    for i in range(0, 7):
        momentum = momentum.at[2 * i].add(-pbarx / 7.0)
        momentum = momentum.at[2 * i + 1].add(-pbary / 7.0)

    # define environment
    env = lj7.LJ7()
    print(env.potential(state))
    print(env.grad(state))

    # define sampling method
    sampler = ver.verlet(env, state, momentum, 0.01)

    # run simulation and print states
    position = []
    #angle = []
    for i in range(100):
        state,_,_,_ = sampler.step_eng()
        position.append(state)
    #    angle.append(env.get_angle(state))
    #    print(state)

    #visual.visualize_trajectory(position)


if __name__ == '__main__':
    main()
