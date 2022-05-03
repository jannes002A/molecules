#!/bin/python

import time

import jax.numpy as jnp
import numpy as np

import molecules.methods.euler_maruyama as em
import molecules.models.double_well as dw
import molecules.visual.visual_sde as visual

def main():
    """ Main method for running a trajectory following the double well environment
    """

    # define parameters
    d = 4
    dim = (d,)
    beta = 1.0
    alpha_i = 1.0

    # initial position
    x0 = jnp.array([-1.0] * d)
    print('initial state: {}'.format(x0))

    # define environment
    env = dw.DoubleWell(stop=[1.0], dim=d, beta=beta, alpha=[alpha_i])
    print('potential at initial state: {}'.format(env.potential(x0)))
    print('gradient at initial state: {}'.format(env.grad(x0)))

    # define sampling method
    dt = 0.01
    sampler = em.Euler_maru(env, x0, dt=dt, key=1)

    # set the control to zero
    action = jnp.zeros(env.dim)

    # position and potential
    position = [np.asarray(x0)]
    potential = [env.potential(jnp.asarray(x0))]

    # run simulation
    n_max_steps = 10000

    # start timer
    start = time.time()

    for n in range(n_max_steps):

        # update position
        state, reward, done, _ = sampler.step(action)
        #breakpoint()
        print(state)
        # get position and potential
        position.append(np.asarray(state))
        potential.append(env.potential(state))

        # stop simulation if hitting set is reached
        if done:
            msg = 'time step: {:d}, state: {}'.format(n, state)
            print(msg)
            break

    # stop timer
    end = time.time()
    print('CT: {:.1e}'.format(end-start))

    # visualization of trajectory
    visual.visualize_trajectory(visual.convert2df(position))
    #TODO! bug by visualizing the trajectory

if __name__ == '__main__':
    main()
