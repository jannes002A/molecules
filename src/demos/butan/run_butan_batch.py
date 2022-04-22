#!/bin/python

import sys
import time
sys.path.append('../../')

import jax.numpy as jnp
import matplotlib.pyplot as plt

import methods.euler_maruyama_batch as em
import models.butan as butan
import visual.visual_butan as visual

def main():
    """ Main method for running a simulation of Butane
    """

    # define environment
    env = butan.Butan(stop=160, beta=4.0)

    # define initial state. Initial 3d position of the 4 Carbon atoms
    q0 = jnp.array([
        [0.1, 0.2, 0.1, 0.2],
        [0.1, 0.2, 0.3, 0.4],
        [0.0, 0.0, 0.0, 0.0],
    ])
    print('initial state: {}'.format(q0))
    print('initial angle: {}'.format(env.get_angle(q0)))
    print('potential at initial state: {}'.format(env.potential(q0)))
    print('gradient at initial state: {}'.format(env.grad(q0)))

    # batch size
    K = 10
    q0 = jnp.repeat(q0[jnp.newaxis, :, :], K, axis=0)

    # define sampling method
    dt = 0.000005
    sampler = em.Euler_maru(env, q0, dt, K, key=10)
    action = jnp.zeros((K,) + env.dim)

    # preallocate position, angle and potential along the trajectories
    n_max_steps = 100
    position = jnp.empty((n_max_steps+1, K)+env.dim, jnp.float32)
    angle = jnp.empty((n_max_steps+1, K), jnp.float32)
    potential = jnp.empty((n_max_steps+1, K), jnp.float32)

    # get initial position, angle and potential
    position = position.at[0].set(q0)
    angle = angle.at[0].set(env.get_angle_batch(q0))
    potential = potential.at[0].set(env.potential_batch(q0))

    # run simulation
    for n in range(n_max_steps):

        # update position
        state, _, _, _ = sampler.step(action)

        # get position, angle and potential
        position = position.at[n+1].set(state)
        angle = angle.at[n+1].set(env.get_angle_batch(state))
        potential = potential.at[n+1].set(env.potential_batch(state))

    #visual.visualize_trajectory_batch(position)
    visual.visualize_angle_batch(angle, dt)
    visual.visualize_potential_batch(potential, dt)


if __name__ == '__main__':
    main()
