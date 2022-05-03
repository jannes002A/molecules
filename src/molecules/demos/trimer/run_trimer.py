"""script for running a simulation of Timer """
import jax.numpy as jnp
import sys
sys.path.append('../../')
# import user defined classes
import models.trimer as trimer
import methods.verlet as ver


def main():
    """Main method for sampling """
    # Example 1D
    q0 = jnp.array([1.0, 0.0])
    p0 = jnp.array([0.02, -0.01])
    mass = jnp.array([2, 2.0/3.0])

    # define environment
    env = trimer.Trimer(mass)
    print(env.potential(q0))
    print(env.grad(q0))

    # define sampling method
    sampler = ver.verlet(env, q0, p0, 0.01)

    # run simulation
    for i in range(100):
        print(sampler.step_eng())


if __name__ == '__main__':
    main()
