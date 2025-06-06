import numpy as np
from functools import partial
from nbodysim import n_body_ode_reference, make_n_body_ode_eff
import pytest

@pytest.mark.benchmark(group='ode_step')
def test_bench_reference(benchmark):
    """
    Benchmark the optimized and unoptimized version.
    """

    rng = np.random.default_rng(seed=42)
    n_states = 100
    n_bodies = 10
    d = 2
    G_test = 1.0
    
    masses = np.abs(rng.standard_normal(size=(n_bodies,)) + 0.1)
    states = rng.standard_normal((n_states, n_bodies * d * 2))
    
    fn_a = partial(n_body_ode_reference, masses=masses, G=G_test)

    def reference():
        for state_idx in range(n_states):
            state = states[state_idx, :]
            _state_dx_fn_a = fn_a(0.0, state)

    benchmark(reference)

@pytest.mark.benchmark(group='ode_step')
def test_bench_optimized_seq(benchmark):
    """
    Benchmark the optimized and unoptimized version.
    """

    rng = np.random.default_rng(seed=42)
    n_states = 100
    n_bodies = 10
    d = 2
    G_test = 1.0
    
    masses = np.abs(rng.standard_normal(size=(n_bodies,)) + 0.1)
    states = rng.standard_normal((n_states, n_bodies * d * 2))
    
    fn_b = make_n_body_ode_eff(masses, G=G_test)
    
    def optimized_seq():
        for state_idx in range(n_states):
            state = states[state_idx, :]
            _state_dx_fn_b = fn_b(0.0, state)
    

    benchmark(optimized_seq)

@pytest.mark.benchmark(group='ode_step')
def test_bench_optimized_batch(benchmark):
    """
    Benchmark the optimized and unoptimized version.
    """

    rng = np.random.default_rng(seed=42)
    n_states = 100
    n_bodies = 10
    d = 2
    G_test = 1.0
    
    masses = np.abs(rng.standard_normal(size=(n_bodies,)) + 0.1)
    states = rng.standard_normal((n_states, n_bodies * d * 2))
    
    fn_b = make_n_body_ode_eff(masses, G=G_test)

    
    def optimized_batch():
        fn_b(0.0, states)

    benchmark(optimized_batch)
