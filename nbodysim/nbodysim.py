import numpy as np
from scipy.spatial.distance import pdist, squareform
# from scipy.integrate import solve_ivp

# gravitational constant - which value we use here will depend on the units that we use.
# https://en.wikipedia.org/wiki/Gravitational_constant - recommends 6.67430(15) 10^{−11} m^{3} kg^{−1} s^{−2}
G = 6.67430e-11 # m^{3} kg^{−1} s^{−2}
# Or, if using km instead of meters km = 10^3m, km^2 = 10^6m, km^3 = 10^9m
# G = 6.67430e-20 # km^{3} kg^{−1} s^{−2}

def calculate_interactions_reference(positions: np.ndarray, masses: np.ndarray, G: float=G):
    """
    Calculate the interactions between n different bodies.

    Note that this implementation is according to the textbook formulas, and not optimized,
    many optimizations (referred to within the function body below) would allow for 
    the computations to be implemented more efficiently with little effect on the precision.
     
    Parameters:
        positions - a #bodies x dimensions matrix. (in matching distance unit to the gravitional constant)
        masses    - a #bodies matrix. (in matching weight units to the gravitional constant)
    
    """
    masses = masses.reshape(-1, 1)
    
    # Calculate all pairwise distances - in practice, the estimated force below can be very small
    # and not all of these may be strictly necessary.
    # Furthermore, symmetry could be exploited here.
    r = squareform(pdist(positions))
    # fill in the diagonal - to avoid division by zero for bodies themselves.
    np.fill_diagonal(r, np.inf)

    # With this, we can calculate the magnitude of the force between any of the two bodies using the
    # F = G * (m_1 m_2) / (r**2)
    force = G * (masses * masses.T) / np.square(r)
    # The direction of these forces is then the difference vector, normalized.
    directions = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    directions /= r[:, :, np.newaxis]
    # the net force vector is hence
    force_vec = np.sum(force[:, :, np.newaxis] * directions, axis=1)

    # force = mass * acceleration - we need acceleration for the actual simulation.
    acceleration = force_vec / masses
    return acceleration

def n_body_ode_reference(time: float, state: np.ndarray, masses: np.ndarray, G=G):
    """
    The function that provides the derivative provided a state at a given time.

    Parameters:
        time   - the current time
        state  - the state of the system, detailed further below.
    
        The following parameters should be set (e.g. using partial) prior to calling `solve_ivp`
        as this argument is not provided by the solver.
        n_bodies
        masses - the masses of each body in the system, 

    State:
        The state of the system consists of two parts, the first half describes the position of each body
        while the second part describes its velocity. Initial values should be provided for these entities.
    """
    # Other constants
    n_bodies = masses.shape[0]

    # Split the state in two parts - positions and velocities
    positions = state[:len(state)//2].reshape(n_bodies, -1)
    velocities = state[len(state)//2:].reshape(n_bodies, -1)

    # For reference - the dimensionality of the space is as follows,
    # we don't actually need this number, as this is implicit in how the matrices
    # are being used!
    # d = positions.shape[1]

    # velocities is the derivative of the first half (positions)

    # compute the (# masses x d) acceleration matrix, which forms the derivative of the second half
    acceleration = calculate_interactions_reference(positions, masses, G=G)

    d_state = np.concatenate([velocities.ravel(), acceleration.ravel()])
    return d_state

def make_n_body_ode_eff(masses: np.ndarray, G=G):
    """
    Construct the ode equation given a particular set of masses.

    Parameters:
        masses - the masses of each body in the system.

    Returns:
        A function that takes arguments (time, state)

        time   - the current time
        state  - the state of the system, detailed further below.

        State:
            The state of the system consists of two parts, the first half describes the position of each body
            while the second part describes its velocity. Initial values should be provided for these entities.
    """
    n_bodies = masses.shape[0]

    # Precompute - this is constant across all steps
    masses = masses.reshape(-1, 1)
    # Usually we use the following precomputed value...
    # Gm1m2 = (G * (masses * masses.T))[:, :, np.newaxis]
    # But when calculating acceleration, we divide by masses again
    Gm2 = (G * (masses.T))[np.newaxis, :, :, np.newaxis]
    

    def n_body_ode_eff(_times: float, state: np.ndarray):
        # Precomputed (cached) values
        nonlocal masses, n_bodies, Gm2
        # For vectorization, we need to be able to accept both a time vector, and state can be a [state]
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        else:
            state = state.T
        n_states = state.shape[0]
        n_state_elem = state.shape[1]


        # Split the state in two parts - positions and velocities
        positions = state[:, :n_state_elem//2].reshape(n_states, n_bodies, -1)
        velocities = state[:, n_state_elem//2:].reshape(n_states, n_bodies, -1)
        
        # velocities is the derivative of the first half (positions)

        # The force (including direction) are as follows
        m_acceleration = positions[:, np.newaxis, :, :] - positions[:, :, np.newaxis, :]
        m_acceleration_norm = (np.linalg.norm(m_acceleration, axis=-1, keepdims=True) ** 3)
        m_acceleration /= np.where(m_acceleration_norm <= 0.0, np.inf, m_acceleration_norm)
        m_acceleration *= Gm2
        
        # the net force vector is hence
        net_acceleration = np.sum(m_acceleration, axis=2)

        d_state = np.concatenate([velocities.reshape(n_states, -1),
                                  net_acceleration.reshape(n_states, -1)], axis=1)
        return d_state.T
    
    return n_body_ode_eff
