import jax.numpy as jnp
import pytest

import models.double_well as dw
import methods.euler_maruyama_batch as em


class TestEulerMayurama:

    @pytest.fixture
    def doublewell_env(self):
        return dw.DoubleWell(stop=[1.0], dim=2, beta=1.0, alpha=[0.5])

    @pytest.fixture
    def doublewell_sampler(self, doublewell_env):

        # get dimension of the double well problem
        dim = doublewell_env.dim
        d = dim[0]

        # batch size
        K = 10

        # initial position i-th coordinate
        xinit = -1.0 * jnp.ones((K, d))

        # time step
        dt = 0.01

        return em.Euler_maru(doublewell_env, start=xinit, dt=dt, K=K)

    #@pytest.mark.skip()
    def test_em_step(self, doublewell_env, doublewell_sampler):

        # environment and sampler
        env = doublewell_env
        sampler = doublewell_sampler
        K = sampler.K

        # potential
        potential = env.potential

        # initial action
        action = jnp.zeros((K,) + env.dim)

        # step dynamics forward
        output = sampler.step(action)

        # check type output
        assert type(output) == tuple
        assert len(output) == 4

        # unwrapp tuple
        state, reward, done, obs = output

        # state
        assert isinstance(state, jnp.ndarray)
        assert state.ndim == 2
        assert state.shape == (K,)+env.dim

        # reward
        assert isinstance(reward, jnp.ndarray)
        assert reward.ndim == 1
        assert reward.shape[0] == K

        # done
        assert isinstance(done, jnp.ndarray)
        assert done.dtype == bool
