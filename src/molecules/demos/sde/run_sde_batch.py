#!/bin/python

import jax.numpy as jnp

import molecules.methods.euler_maruyama_batch as em
import molecules.models.double_well as dw
import molecules.visual.visual_sde as visual


def main():
    """ Main method for running a batch of trajectories following the double well environment
    """

    # example d = 2, beta = 1.0, alpha_i = 0.5 
    d = 1
    beta = 1.0
    alpha_i = 1.0

    # number of trajectories
    K = 5

    # initial position
    x0 = -1.0 * jnp.ones((K, d))
    #print('initial state: {}'.format(x0))

    # define environment
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])
    #print('potential at initial state: {}'.format(env.potential_batch(x0)))
    #print('gradient at initial state: {}'.format(env.grad_batch(x0)))

    # define sampling method
    dt = 0.01
    sampler = em.EulerMaru(env, x0, dt, K, seed=1)

    # set the control to zero
    actions = jnp.zeros((K, d))

    # preallocate position (ith coordinate) and potential along the trajectories
    n_max_steps = 10000
    position = jnp.empty((n_max_steps+1, K), jnp.float32)
    potential = jnp.empty((n_max_steps+1, K), jnp.float32)

    # get initial position and potential
    i = 0
    position = position.at[0, :].set(x0[:, i])
    potential = potential.at[0, :].set(env.potential_batch(x0))

    # run simulation
    for n in range(n_max_steps):

        # update position
        states, rewards, done, obs = sampler.step(actions)
        dbt, idx_new = obs

        print('time step: {}, rewards: {}'.format(n, rewards))

        # get position and potential
        position = position.at[n+1, :].set(states[:, i])
        potential = potential.at[n+1, :].set(env.potential_batch(states))

        if done:
            print('time step: {:d}, state: {}'.format(n, states))
            break

    # visualization of the ith coordinate of the trajectories
    #visual.visualize_trajectory_batch(position, i, dt)

    # visualization of the potential along the trajectories
    #visual.visualize_potential_batch(potential, dt)

if __name__ == '__main__':
    main()
