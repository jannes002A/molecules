#!/bin/python

import jax.numpy as jnp
import matplotlib.pyplot as plt

import molecules.methods.euler_maruyama_batch as em
import molecules.models.butan as butan
import molecules.visual.visual_butan as visual

def main():
    """ Main method for running a batch of Butane simulations
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
    K = 5
    q0 = jnp.repeat(q0[jnp.newaxis, :, :], K, axis=0)

    # define sampling method
    dt = 0.000005
    sampler = em.EulerMaru(env, q0, dt, K, seed=1)
    actions = jnp.zeros((K,) + env.dim)

    # preallocate position, angle and potential along the trajectories
    n_max_steps = 100
    positions = jnp.empty((n_max_steps+1, K)+env.dim, jnp.float32)
    angles = jnp.empty((n_max_steps+1, K), jnp.float32)
    potentials = jnp.empty((n_max_steps+1, K), jnp.float32)

    # get initial position, angle and potential
    positions = positions.at[0].set(q0)
    angles = angles.at[0].set(env.get_angle_batch(q0))
    potentials = potentials.at[0].set(env.potential_batch(q0))

    # run simulation
    for n in range(n_max_steps):

        # update position
        states, rewards, done, obs = sampler.step(actions)

        print('time step: {}, rewards: {}'.format(n, rewards))

        # get position, angle and potential
        positions = positions.at[n+1].set(states)
        angles = angles.at[n+1].set(env.get_angle_batch(states))
        potentials = potentials.at[n+1].set(env.potential_batch(states))

    visual.visualize_trajectory(positions[:, 0])
    visual.visualize_angle_batch(angles, dt)
    visual.visualize_potential_batch(potentials, dt)


if __name__ == '__main__':
    main()
