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

def get_simple_system():
    masses = np.array([1e26, 1e26, 1e26])

    # Start in a simple lunar eclipse configuration, keeps things simple.
    initial_positions = np.array([
        [ 0.0e8,  1.0e8],
        [ 0.0e8, -1.0e8],
        [ 1.0e8,  0.0e8],
    ])
    # For ease, going with the average velocity as the starting velocity.
    # perpendicular to the offset direction
    initial_velocities = np.array([
        [ 1.0e3, -1.0e3],
        [-1.0e3,  1.0e3],
        [-1.0e3,  1.0e3]
    ])

    initial_state = np.concatenate([initial_positions.ravel(), initial_velocities.ravel()])

    return masses, initial_state

def test_simple_sim():
    from scipy.integrate import solve_ivp

    rtol = 1e-5 # defaults to 1e-3
    atol = 1e-10 # defaults to 1e-6
    masses, initial_state = get_simple_system()
    t_span = [0.0, float(60.0 * 60.0 * 24.0)] # a day
    ode = make_n_body_ode_eff(masses)
    _res = solve_ivp(ode, t_span, y0=initial_state, dense_output=True, vectorized=True, method="DOP853", rtol=rtol, atol=atol)

def test_simple_sim_with_jacobian():
    from scipy.integrate import solve_ivp
    
    rtol = 1e-5 # defaults to 1e-3
    atol = 1e-10 # defaults to 1e-6
    masses, initial_state = get_simple_system()
    t_span = [0.0, float(60.0 * 60.0 * 24.0)] # a day
    ode, ode_jac = make_n_body_ode_eff(masses, jacobian=True)
    _res = solve_ivp(ode, t_span, y0=initial_state, jac=ode_jac, dense_output=True, vectorized=True,  method="BDF", rtol=rtol, atol=atol)
