#!/bin/python

from collections import namedtuple
import time
import sys

import jax.numpy as jnp
from jax import random

from doublewell_env import DoubleWell, potential_batch, gradient_batch
from euler_marujama import EulerMaru, step_batch

def main():
    """ Main method for running a trajectory following the SDE environment
    """

    # define parameters
    d = 2
    dim = (d,)
    beta = 1.0
    stop = jnp.array([1.0] * d)
    alpha = jnp.array([1.0] * d)

    # number of trajectories
    K = 10

    # initial position
    x0 = -1.0 * jnp.ones((K, d))
    print('initial state: {}'.format(x0))

    # start named tuple
    env = DoubleWell(dim, beta, stop, alpha)
    print('potential at initial state: {}'.format(potential_batch(env, x0)))
    print('gradient at initial state: {}'.format(gradient_batch(env, x0)))

    # get key
    key = random.PRNGKey(seed=1)

    # define sampling method
    dt = 0.01
    em = EulerMaru(env, dt=dt, key=key)

    # set the control to zero
    actions = jnp.zeros((K, d))

    # initialize state
    states = x0

    # position and potential along the trajectory
    positions = [x0]
    potentials = [potential_batch(env, x0)]

    # run simulation
    n_max_steps = 100000

    # start timer
    start = time.time()

    for n in range(n_max_steps):

        # update key
        key, subkey = random.split(key)

        # compute brownian increments
        dbt = jnp.sqrt(em.dt) * random.normal(subkey, shape=(K,) + em.env.dim)

        # update position
        state, reward, done, obs = step_batch(em, states, actions, dbt)

        # get position and potential
        positions.append(states)
        potentials.append(potential_batch(env, states))

        # stop simulation if hitting set is reached for all trajectories
        if done.all():
            msg = 'time step: {:d}, state: {}'.format(n, state)
            print(msg)
            break

    end = time.time()
    print('CT: {:.1e}'.format(end-start))

if __name__ == '__main__':
    main()
