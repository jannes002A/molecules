from collections import namedtuple

import jax.numpy as jnp
from jax import jit, vmap

DoubleWell = namedtuple(
    "DoubleWell",
    [
        "dim",
        "beta",
        "stop",
        "alpha",
    ],
)

@jit
def potential(env: namedtuple, state: jnp.ndarray):
    """ Calculates the potential at the given state

    Parameters
    ----------
    env : namedtuple
        tuple containing the parameters of the double well environment
    state : jax array
        state of the system

    Returns
    -------
    pot: float
        potential at state
    """
    return jnp.sum(env.alpha * (state**2 - 1) ** 2)

@jit
def potential_batch(env: namedtuple, states_batch: jnp.ndarray):
    """Calculates potential for batch of states

    Parameters
    ----------
    env : namedtuple
        tuple containing the parameters of the double well environment
    states_batch : jax array
        batch of K states of the system

    Returns
    -------
    pot: jax array (K,)
        potential at each state
    """
    return vmap(potential, in_axes=(None, 0), out_axes=0)(env, states_batch)

@jit
def gradient(env: namedtuple, state: jnp.ndarray):
    """ Calculates the gradient at the given state

    Parameters
    ----------
    env : namedtuple
        tuple containing the parameters of the double well environment
    state : jax array
        state of the system

    Returns
    -------
    grad: jax array (d,)
        gradient at state
    """
    return 4 * env.alpha * state * (state ** 2 - 1)

@jit
def gradient_batch(env: namedtuple, states_batch: jnp.ndarray):
    """ Calculates the gradient of the potential evaluated at the given batch of states

    Parameters
    ----------
    env : namedtuple
        tuple containing the parameters of the double well environment
    states_batch : jax array
        batch of K states of the system

    Returns
    -------
    grad: jnp array (K, d)
        grad at state
    """
    return vmap(gradient, in_axes=(None, 0), out_axes=0)(env, states_batch)

@jit
def criterion(env: namedtuple, state: jnp.ndarray):
    """ Calculates if the state is in the hitting set

    Parameters
    ----------
    env : namedtuple
        tuple containing the parameters of the double well environment
    state : jax array
        state of the system

    Returns
    -------
    bool
        if set is hit
    """
    return (state > jnp.array(env.stop)).all()

@jit
def criterion_batch(env: namedtuple, states_batch: jnp.ndarray):
    """Calculates if states in batch are in the hitting set

    Parameters
    ----------
    states_batch : jax array
        batch of K states of the system

    Returns
    -------
    is_in_hitting_set: jax array of bools
        true if it is in the hitting set
    """
    return vmap(criterion, in_axes=(None, 0), out_axes=0)(env, states_batch)
