import numpy as np
from functools import partial
from nbodysim import n_body_ode_reference, make_n_body_ode_eff

def test_optimized_versus_reference_elementwise():
    """
    Test whether the optimized function behaves identically.
    """
    rng = np.random.default_rng(seed=42)
    n_states = 100
    n_bodies = 10
    d = 2
    G_test = 1.0
    
    masses = np.abs(rng.standard_normal(size=(n_bodies,)) + 0.1)
    states = rng.standard_normal((n_states, n_bodies * d * 2))
    
    fn_a = partial(n_body_ode_reference, masses=masses, G=G_test)
    fn_b = make_n_body_ode_eff(masses, G=G_test)

    for state_idx in range(n_states):
        print(f"Checking state {state_idx}")
        state = states[state_idx, :]
        state_dx_fn_a = fn_a(0.0, state)
        state_dx_fn_b = fn_b(0.0, state).ravel()
        assert np.allclose(state_dx_fn_a, state_dx_fn_b)


def test_optimized_versus_reference_vectorized():
    """
    Test whether the optimized function behaves identically when used in a vectorized manner.
    """
    rng = np.random.default_rng(seed=42)
    n_states = 100
    n_bodies = 10
    d = 2
    G_test = 1.0
    
    masses = np.abs(rng.standard_normal(size=(n_bodies,)) + 0.1)
    states = rng.standard_normal((n_states, n_bodies * d * 2))
    
    fn_a = partial(n_body_ode_reference, masses=masses, G=G_test)
    fn_b = make_n_body_ode_eff(masses, G=G_test)
    states_dx_fn_b = fn_b(0.0, states.T).T

    for state_idx in range(n_states):
        state = states[state_idx, :]
        state_dx_fn_a = fn_a(0.0, state)
        state_dx_fn_b = states_dx_fn_b[state_idx, :]
        assert np.allclose(state_dx_fn_a, state_dx_fn_b)



def test_small_difference():
    # Used only for plotting
    # from matplotlib.colors import SymLogNorm
    # import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed=42)
    n_bodies = 10
    d = 2
    G_test = 1.0
    
    masses = np.abs(rng.standard_normal(size=(n_bodies,)) + 0.1)
    state = rng.standard_normal((n_bodies * d * 2)) * 20
    
    fn_n, fn_jac  = make_n_body_ode_eff(masses, G=G_test, jacobian=True)

    # res = scipy.differentiate.jacobian(partial(fn_n, 0.0), state)

    dx = 1e-5
    diffv = np.eye(n_bodies * d * 2) * dx
    
    states_dt_l = fn_n(0.0, state[:, np.newaxis] - diffv)
    states_dt_r = fn_n(0.0, state[:, np.newaxis] + diffv)

    states_dt_dr = (states_dt_r - states_dt_l) / (2 * dx)
    # states_dt_dr = states_dt_dr[:20, ...]

    state_jac = fn_jac(0.0, state)

    states_dt_dr = states_dt_dr.reshape(diffv.shape[0], -1)
    state_jac = state_jac.reshape(diffv.shape[0], -1)
    # state_diff = states_dt_dr - state_jac

    # Visualisation for user
    # fig, ax = plt.subplots(1, 3)
    # ax[0].imshow(states_dt_dr, norm=SymLogNorm(1e-5))
    # ax[0].set_title("Estimated Ground Truth")
    # ax[1].imshow(state_jac, norm=SymLogNorm(1e-5))
    # ax[1].set_title("Implementation")
    # ax[2].imshow(state_diff, norm=SymLogNorm(1e-5))
    # ax[2].set_title("Difference")
    # plt.show()

    assert np.allclose(states_dt_dr, state_jac)
