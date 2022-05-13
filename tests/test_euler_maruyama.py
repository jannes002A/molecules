import jax.numpy as jnp
import pytest

import molecules.models.double_well as dw
import molecules.methods.euler_maruyama as em

class TestEulerMayurama:

    @pytest.fixture
    def doublewell_env(self):
        return dw.DoubleWell(stop=[1.0], dim=10, beta=1.0, alpha=[0.5])

    @pytest.fixture
    def doublewell_sampler(self, doublewell_env):

        # get dimension of the double well problem
        dim = doublewell_env.dim
        d = dim[0]

        # initial position i-th coordinate
        xinit_i = - 1.0

        # time step
        dt = 0.01

        return em.EulerMaru(doublewell_env, start=[xinit_i] * d, dt=dt)

    #@pytest.mark.skip()
    def test_em_step(self, doublewell_env, doublewell_sampler):

        # environment and sampler
        env = doublewell_env
        sampler = doublewell_sampler

        # potential
        potential = env.potential

        # initial action
        action = jnp.zeros(env.dim)

        # step dynamics forward
        output = sampler.step(action)

        # check type output
        assert type(output) == tuple
        assert len(output) == 4

        # unwrapp tuple
        state, reward, done, dbt = output

        # state
        assert isinstance(state, jnp.ndarray)
        assert state.ndim == 1
        assert state.shape == env.dim

        # reward
        assert isinstance(reward, jnp.ndarray)
        assert reward.ndim == 0

        # done
        assert isinstance(done, jnp.ndarray)
        assert done.dtype == bool
