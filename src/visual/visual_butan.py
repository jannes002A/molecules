"""helper function for visualization of butan"""
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
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


def visualize_trajectory(states: list) -> None:
    """Function for quick visualization

    Parameters
    ----------
    states : pd.DataFrame
        data of a trajectory

    Returns
    -------
        pd.DataFrame
            trajectory data in a data frame
    """
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(len(states)):
        cstate = states[i]
        ax.plot(cstate[0, :], cstate[1, :], cstate[2, :], 'bo-')
        ax.set_xlim(0.0, 0.5)
        ax.set_ylim(0.0, 0.5)
        ax.set_zlim(0.0, 0.5)
        ax.view_init(elev=50,azim=10)
        plt.draw()
        #plt.savefig(f'./butane/Butane_{i}.jpg')
        plt.pause(0.04)
        ax.cla()

def visualize_angle_batch(angle: list, dt: float) -> None:
    """Function for visualizing the angle of the trajectories as a function of time

    Parameters
    ----------
    angle : jnp array
       angle along the trajectories
    dt : float
        time discretization step
    """

    # number of states and final time
    n_states = angle.shape[0]
    t_init = 0
    t_final = (n_states - 1) * dt

    # time steps and position of states
    x = np.arange(t_init, t_final + dt, dt)
    y = angle

    plt.plot(x, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle')
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
