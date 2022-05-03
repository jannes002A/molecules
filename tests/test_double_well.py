import jax.numpy as jnp
from jax import random, grad, jit, vmap
import numpy as np
import pytest

import molecules.models.double_well as dw

class TestDoubleWell:

    @pytest.fixture
    def key(self):
        key = random.PRNGKey(1)
        return key

    @pytest.fixture
    def doublewell_env(self):
        return dw.DoubleWell(stop=[1.0], dim=10, beta=1.0, alpha=[0.5])

    @pytest.fixture
    def state(self, doublewell_env, key):

        # get dimension of the state space
        dim = doublewell_env.dim

        return random.uniform(key, dim, jnp.float32, -3, 3)

    @pytest.fixture
    def states_batch(self, doublewell_env, key):

        # batch size
        K = 10

        # get dimension of the state space
        dim = doublewell_env.dim

        return random.uniform(key, (K,)+dim, jnp.float32, -3, 3)

    def test_state_space(self, doublewell_env):

        # check sde state space. d coordinates for the particle following the doublewell
        assert isinstance(doublewell_env.dim, tuple)
        assert len(doublewell_env.dim) == 1

    def test_potential(self, doublewell_env, state):

        # evaluate potential at the state 
        potential = doublewell_env.potential(state)

        # check object type and array dimension
        assert isinstance(potential, jnp.ndarray)
        assert potential.ndim == 0

    def test_potential_computation(self, doublewell_env, state):

        # get dimension of the potential domain
        dim = doublewell_env.dim
        d = dim[0]

        # get parameter alpha
        alpha = doublewell_env.alpha

        # evaluate potential at x 
        potential = doublewell_env.potential(state)

        # compute potential by looping through the dimension
        potential_test = 0
        for i in range(d):
            potential_test += alpha[i] * (state[i]**2 - 1)**2

        assert np.isclose(potential, potential_test)

    def test_potential_vectorization(self, doublewell_env, states_batch):

        # get dimension of the state space
        dim = doublewell_env.dim

        # batch size
        K = states_batch.shape[0]

        # evaluate potential at x 
        potential = doublewell_env.potential_batch(states_batch)

        # check object type, array dimension and shape
        assert isinstance(potential, jnp.ndarray)
        assert potential.ndim == 1
        assert potential.shape == (K,)

    def test_gradient(self, doublewell_env, state):

        # evaluate potential at the state 
        gradient = doublewell_env.grad(state)

        # check object type, array dimension and shape
        assert isinstance(gradient, jnp.ndarray)
        assert gradient.ndim == 1
        assert gradient.shape == doublewell_env.dim

    def test_gradient_computation(self, doublewell_env, state):

        # get dimension of the potential domain
        dim = doublewell_env.dim
        d = dim[0]

        # get parameter alpha
        alpha = doublewell_env.alpha

        # evaluate potential at x 
        gradient = doublewell_env.grad(state)

        # compute gradient by looping through the dimension
        gradient_test = jnp.array([
            4 * alpha[i] * state[i] * (state[i] ** 2 - 1) for i in range(d)
        ])

        assert np.isclose(gradient, gradient_test).all()

    def test_gradient_vectorization(self, doublewell_env, states_batch):

        # get dimension of the potential domain
        dim = doublewell_env.dim
        d = dim[0]

        # batch size
        K = states_batch.shape[0]

        # evaluate potential at x 
        gradient = doublewell_env.grad_batch(states_batch)

        # check object type and array dimension
        assert isinstance(gradient, jnp.ndarray)
        assert gradient.ndim == 2
        assert gradient.shape == (K, d)

    def test_criterion(self, doublewell_env):

        # get dimension of the potential domain
        dim = doublewell_env.dim
        d = dim[0]

        # state in the hitting set
        state = 1.5 * jnp.ones(d, jnp.float32)

        # apply criterion
        is_in_hitting_set = doublewell_env.criterion(state)

        assert is_in_hitting_set

        # state not in the hitting set
        state = 1.1 * jnp.ones(d, jnp.float32)
        state = state.at[0].set(0.9)

        # apply criterion
        is_in_hitting_set = doublewell_env.criterion(state)

        assert not is_in_hitting_set
