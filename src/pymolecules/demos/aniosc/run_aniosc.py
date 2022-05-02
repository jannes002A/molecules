"""script for running a simulation of Ani Oscilator
"""
import jax.numpy as jnp
import sys
sys.path.append('../../')
# import user defined classes
import models.aniosc as aniosc
import methods.verlet as ver
import visual.visual_aniosc as visual


def main():
    """Main method for sampling """
    # define parameters
    epsilon = 0.5  # anisotropy parameter
    q0 = jnp.array([2.0, 0.0])
    p0 = jnp.array([0.0, 0.01])
    mass = jnp.array([1.0, 1.0])

    # get environment check if it is workring
    env = aniosc.Aniosc(mass, epsilon)
    print(env.potential(q0))
    print(env.grad(q0))

    # get sampling method
    sampler = ver.verlet(env, q0, p0, 0.1)
    action = jnp.zeros(env.dim)

    # run sampling and print states
    position = []
    energy = []
    for i in range(100):
        state, _, eng, _ = sampler.step_eng()
        position.append(state)
        energy.append(eng)

    # visualize trajectory
    visual.visualize_trajectory(visual.convert2df(position))


if __name__ == '__main__':
    main()
