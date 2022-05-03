import jax.numpy as jnp
from jax import random, grad, jit, vmap
import numpy as np
import pytest

import molecules.models.butan as butan

class TestButan:

    @pytest.fixture
    def key(self):
        key = random.PRNGKey(1)
        return key

    @pytest.fixture
    def butan_env(self):
        return butan.Butan(stop=160, beta=4.0)

    @pytest.fixture
    def state(self, butan_env, key):

        # get dimension of the state space
        dim = butan_env.dim

        return random.uniform(key, dim, jnp.float32, -1, 1)

    @pytest.fixture
    def states_batch(self, butan_env, key):

        # batch size
        K = 10

        # get dimension of the state space
        dim = butan_env.dim

        return random.uniform(key, (K,)+dim, jnp.float32, -1, 1)

    def test_state_space(self, butan_env):

        # check butan state space. 3 spacial coordinates for 4 carbon atoms
        assert isinstance(butan_env.dim, tuple)
        assert butan_env.dim == (3, 4)

    def test_potential(self, butan_env, state):

        # evaluate potential at the state 
        potential = butan_env.potential(state)

        # check object type and array dimension
        assert isinstance(potential, jnp.ndarray)
        assert potential.ndim == 0

    def test_potential_vectorization(self, butan_env, states_batch):

        # batch size
        K = states_batch.shape[0]

        # evaluate potential at x 
        potential = butan_env.potential_batch(states_batch)

        # check object type, array dimension and shape
        assert isinstance(potential, jnp.ndarray)
        assert potential.ndim == 1
        assert potential.shape == (K,)

    def test_gradient(self, butan_env, state):

        # evaluate gradient at the state
        gradient = butan_env.grad(state)

        # check object type, array dimension and shape
        assert isinstance(gradient, jnp.ndarray)
        assert gradient.shape == butan_env.dim

    def test_gradient_vectorization(self, butan_env, states_batch):

        # get dimension of the state space
        dim = butan_env.dim

        # batch size
        K = states_batch.shape[0]

        # evaluate potential at x 
        gradient = butan_env.grad_batch(states_batch)

        # check object type and array dimension
        assert isinstance(gradient, jnp.ndarray)
        assert gradient.ndim == 1+2
        assert gradient.shape == (K,)+dim

    def test_criterion(self, butan_env, key):

        # get dimension of the state space
        dim = butan_env.dim

        # state not in the hitting set. The two planes are the same: x_3 = 0.
        state = random.uniform(key, dim, jnp.float32, -1, 1)
        state = state.at[2, :].set(0)

        # apply criterion
        is_in_hitting_set = butan_env.criterion(state)

        assert not is_in_hitting_set

        # state in the hitting set. 
        #TODO: find a state which is in the hitting set
        #state = random.uniform(key, dim, jnp.float32, -1, 1)

        # apply criterion
        #is_in_hitting_set = butan_env.criterion(state)

        #assert is_in_hitting_set

    def test_criterion_vectorized(self, butan_env, key):

        # get dimension of the state space
        dim = butan_env.dim

        # batch size
        K = 10

        # state not in the hitting set. The two planes are the same: x_3 = 0.
        state = random.uniform(key, dim, jnp.float32, -1, 1)
        state = state.at[2, :].set(0)

        # batch of states
        states_batch = jnp.repeat(state[jnp.newaxis, :, :], K, axis=0)

        # apply criterion
        are_in_hitting_set = butan_env.criterion_batch(states_batch)

        assert not are_in_hitting_set.all()
