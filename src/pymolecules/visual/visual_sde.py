"""helper function for visualization of sdes"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def convert2df(states: list) -> pd.DataFrame:
    """Function to convert trajectories into a dataframe for quick visualization

    Parameters
    ----------
    states : list#
        data of a trajectory

    Returns
    -------
        pd.DataFrame
            trajectory data in a data frame
    """
    return pd.DataFrame(states)


def visualize_trajectory(df: pd.DataFrame) -> None:
    """Function for quick visualization

    Parameters
    ----------
    df : pd.DataFrame
        data of a trajectory

    Returns
    -------
        pd.DataFrame
            trajectory data in a data frame
    """
    df.plot()
    plt.xlabel('Time Steps')
    plt.ylabel('Particle Position')
    plt.draw()
    plt.show()

def visualize_trajectory_batch(position: list, i: int, dt: float) -> None:
    """Function for visualizing the ith coordinate of the trajectories as a function of time

    Parameters
    ----------
    position : jnp array
        batch of i-th coordinates of the position
    i : int
        i-th coordinate of the position which we visualize
    dt : float
        time discretization step
    """
    # number of states and final time
    n_states = position.shape[0]
    t_init = 0
    t_final = (n_states - 1) * dt

    # time steps and position of states
    x = np.arange(t_init, t_final + dt, dt)
    y = position

    plt.plot(x, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Particle position ({:d}-th coordinate)'.format(i))
    plt.draw()
    plt.show()

def visualize_potential_batch(potential: list, dt: float) -> None:
    """Function for visualizing the potential of the trajectories as a function of time

    Parameters
    ----------
    potential : jnp array
       potential along the trajectories
    dt : float
        time discretization step
    """

    # number of states and final time
    n_states = potential.shape[0]
    t_init = 0
    t_final = (n_states - 1) * dt

    # time steps and position of states
    x = np.arange(t_init, t_final + dt, dt)
    y = potential

    plt.plot(x, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Potential')
    plt.draw()
    plt.show()
