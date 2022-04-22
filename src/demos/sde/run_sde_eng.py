"""script for running a simulation of SDE
position and potential only
"""
import jax.numpy as jnp
import sys
import matplotlib.pyplot as plt
sys.path.append('../../')
# import user defined classes
import models.double_well as dw
import methods.euler_maruyama as em
import visual.visual_sde as vsde

def main():
    """Main method for sampling """
    # define parameters
    d = 10
    x0 = jnp.array([-1.0]*d)

    # define environment
    env = dw.DoubleWell(stop=[1.0], dim=d)
    print(env.potential(x0))
    print(env.grad(x0))

    # define sampling method
    sampler = em.Euler_maru(env, x0, 0.001, key=10)

    # run simulation
    state = []
    energy = []
    for i in range(100):
        new_state, _, p, _ = sampler.step_eng()
        state.append(new_state)
        energy.append(p)
        #print(new_state)

    # plot trajectory
    vsde.visualize_trajectory(vsde.convert2df(state))
    vsde.visualize_trajectory(vsde.convert2df(energy))


if __name__ == '__main__':
    main()
