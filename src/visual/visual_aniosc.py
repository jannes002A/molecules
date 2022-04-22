"""helper function for visualization of aniosc"""
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
    plt.plot(df.loc[:, 0], df.loc[:, 1])
    plt.xlabel('x direction')
    plt.ylabel('y direction')
    plt.draw()
    plt.show()
