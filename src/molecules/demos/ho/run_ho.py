#!/bin/python

import jax.numpy as jnp
import matplotlib.pyplot as plt

import molecules.models.harmonic_oscillator as ho
import molecules.methods.euler_maruyama as em
import molecules.visual.visual_sde as vsde

def main():
    """ Main method for running a trajectory following the harmonic oscillator environment
    """

    # define parameters
    state0 = jnp.array([-1.0])

    # define environment
    env = ho.harm_osci(mass=1.0)
    print(env.potential(state0))
    print(env.grad(state0))

    # define sampling method
    sampler = em.EulerMaru(env, state0, 0.001, seed=10)

    # run simulation
    state = []
    energy = []
    for i in range(100):
        new_state, _, p, _ = sampler.step_eng()
        state.append(new_state)
        energy.append(p)
        #print(new_state)

    # plot trajectory
    #TODO: check
    #vsde.visualize_trajectory(vsde.convert2df(state))
    #vsde.visualize_trajectory(vsde.convert2df(energy))


if __name__ == '__main__':
    main()
