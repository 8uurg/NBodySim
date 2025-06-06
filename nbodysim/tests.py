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