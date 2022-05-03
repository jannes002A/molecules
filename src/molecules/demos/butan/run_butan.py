#!/bin/python

import jax.numpy as jnp
import matplotlib.pyplot as plt

import molecules.methods.euler_maruyama as em
import molecules.models.butan as butan
import molecules.visual.visual_butan as visual

def main():
    """ Main method for running a simulation of Butane
    """

    # define parameters. Initial 3d position of the 4 Carbon atoms
    q0 = jnp.array([
        [0.1, 0.2, 0.1, 0.2],
        [0.1, 0.2, 0.3, 0.4],
        [0.0, 0.0, 0.0, 0.0],
    ])
    print('initial state: {}'.format(q0))

    # define environment
    env = butan.Butan(stop=160, beta=4.0)
    print('potential at initial state: {}'.format(env.potential(q0)))
    print('gradient at initial state: {}'.format(env.grad(q0)))

    # define sampling method
    dt = 0.000005
    sampler = em.Euler_maru(env, q0, dt, key=10)
    action = jnp.zeros(env.dim)

    # position, angle and potential
    position = [q0]
    angle = [env.get_angle(q0)]
    potential = [env.potential(q0)]

    # run simulation
    n_max_steps = 100
    for i in range(n_max_steps):

        # update position
        state, _, _, _ = sampler.step(action)

        # get position, angle and potential
        position.append(state)
        angle.append(env.get_angle(state))
        potential.append(env.potential(state))

    # visualize trajectory
    visual.visualize_trajectory(position)

    #TODO! visualize also the potential and the angle along the trajectory
    #visual.visualize_trajectory(position, dt)
    #visual.visualize_angle_batch(angle, dt)
    #visual.visualize_potential(potential, dt)


if __name__ == '__main__':
    main()
