#!/bin/python

from collections import namedtuple
import time

import jax.numpy as jnp
from jax import random

from molecules.models.doublewell_env import DoubleWell, potential, gradient
from molecules.methods.euler_marujama import EulerMaru, step

def main():
    """ Main method for running a trajectory following the double well environment
    """

    # define parameters
    d = 4
    dim = (d,)
    beta = 1.0
    stop = jnp.array([1.0] * d)
    alpha = jnp.array([1.0] * d)

    # initial position
    x0 = jnp.array([-1.0] * d)
    print('initial state: {}'.format(x0))

    # start named tuple
    env = DoubleWell(dim, beta, stop, alpha)
    print('potential at initial state: {}'.format(potential(env, x0)))
    print('gradient at initial state: {}'.format(gradient(env, x0)))

    # get key
    key = random.PRNGKey(seed=1)

    # define sampling method
    dt = 0.01
    em = EulerMaru(env, dt=dt, seed=1)

    # set the control to zero
    action = jnp.zeros(env.dim)

    # initialize state
    state = x0

    # position and potential along the trajectory
    positions = [x0]
    potentials = [potential(env, x0)]

    # run simulation
    n_max_steps = 10000

    # start timer
    start = time.time()

    for n in range(n_max_steps):

        # update key
        key, subkey = random.split(key)

        # compute brownian increments
        dbt = jnp.sqrt(em.dt) * random.normal(subkey, shape=em.env.dim)

        # update position
        state, reward, done, obs = step(em, state, action, dbt)

        #print(state)

        # get position and potential
        positions.append(state)
        potentials.append(potential(env, state))

        # stop simulation if hitting set is reached
        if done:
            msg = 'time step: {:d}, state: {}'.format(n, state)
            print(msg)
            break

    # stop timer
    end = time.time()
    print('CT: {:.1e}'.format(end-start))

if __name__ == '__main__':
    main()
