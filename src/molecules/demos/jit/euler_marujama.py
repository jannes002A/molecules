from collections import namedtuple
import time

from jax import jit, random
import jax.numpy as jnp

from molecules.models.doublewell_env import gradient, gradient_batch, criterion, criterion_batch


EulerMaru = namedtuple(
    "EulerMaru",
    [
        "env",
        "dt",
        "key",
    ],
    defaults=[0]
)


@jit
def step(em, state, action, dbt):
    """ Performs one euler mayurama step

    Parameters
    ----------
    state : jnp.array
        current state
    action : jax array
        action which should be applied to the system

    Returns
    -------
    new state : jnp.array
        updated state
    reward : float
        reward of the current position
    done : bool
        if target set is reached
    dbt : jnp.array
        current Brownian motion
    """

    # compute drift and diffusion terms
    pot_grad = gradient(em.env, state)

    # stochastic equation of motion
    new_state = state -(pot_grad + action) * em.dt + jnp.sqrt(2 / em.env.beta) * dbt

    # is trajectory in hitting set ?
    done = criterion(em.env, new_state)

    # reward
    reward = -1/2 * action @ action * em.dt - em.dt

    return new_state, reward, done, None

@jit
def step_batch(em, states, actions, dbt):
    """ Performs one euler mayurama step for a batch of states

    Parameters
    ----------
    states : jnp.array
        batch of states
    actions : jax array
        batch of actions which should be applied to the system

    Returns
    -------
    new_states : jnp.array
        updated states
    rewards : float
        reward of the current position
    done : bool
        if target set is reached
    dbt : jnp.array
        current Brownian motion
    """

    # compute drift and diffusion terms
    pot_grad = gradient_batch(em.env, states)

    # stochastic equation of motion
    new_states = states -(pot_grad + actions) * em.dt + jnp.sqrt(2 / em.env.beta) * dbt

    # is trajectory in hitting set ?
    done = criterion_batch(em.env, new_states)

    # reward
    #reward = -1/2 * actions @ actions * em.dt - em.dt
    reward = 0

    return new_states, reward, done, None
