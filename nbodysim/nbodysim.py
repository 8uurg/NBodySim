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

def make_n_body_ode_eff(masses: np.ndarray, G=G, jacobian=False):
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
    Gm2p = (G * (masses.T))[:, :, np.newaxis]


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

    def n_body_ode_eff_jac(_times: float, state: np.ndarray):
        # Precomputed (cached) values
        nonlocal masses, n_bodies, Gm2p
        # The Jacobian matrix has shape (n, n) and its element (i, j) is equal to `d f_i / d y_j`
        # Not sure if the library will ever call this in a vectorized format
        
        assert len(state.shape) == 1, "jac does not expect a vectorized call right now"
        n_state_elem = state.shape[0]

        # We are calculating the jacobian
        # Notes for future vectorization: states are independent:
        # np.zeros((n_states, n_state_elem, n_state_elem)) should suffice!
        jac = np.zeros((n_state_elem, n_state_elem))

        # There are 4 blocks in the jacobian.
        # A B
        # C D
        #
        # A is zero: f_i for this block does not depend on the position.
        # B is identity: f_i is exactly the same as y_(i + offset)
        # D is zero: changes in velocity are not affected by the speed itself
        #            only by the position
        # C is more involved - see below.

        # Split the state in two parts - positions and velocities
        half_state_elem = n_state_elem//2
        positions = state[:half_state_elem].reshape(n_bodies, -1)
        # velocities = state[half_state_elem:].reshape(n_bodies, -1)

        # For the first i positions, the jacobian is simple.
        pos_ids = list(range(half_state_elem))
        vel_ids = list(range(half_state_elem, n_state_elem))
        # velocities is the derivative of the first half (positions) - as this is literally a copy
        # of the state - set B.
        jac[pos_ids, vel_ids] = 1

        # for the remaining elements, the velocities are not used in the jacobian (another bit of sparsity)
        # only these positions are used - so we are going to be a bit lazy, and ignore the remaining half :)
        # Shape here is (s, b, d, bd')
        positions_dy = np.eye((half_state_elem)).reshape((n_bodies, positions.shape[-1],  half_state_elem))

        # The force (including direction) are as follows
        m_acceleration = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        # jacobian, shape here is (s, b_a, b_b, d, s', bd')
        m_acceleration_dy = positions_dy[np.newaxis, :, :, :] - positions_dy[:, np.newaxis, :, :]

        # m_acceleration_norm = (np.linalg.norm(m_acceleration, axis=-1, keepdims=True) ** 3) - split up.
        m_acceleration_sq = np.square(m_acceleration)

        # (g(x)) ** 2 dx = f'(g(x)) g'(x)
        # f(a) = (a) ** 2, f'(a) = 2a
        # hence
        # = g(x) * g'(x)
        m_acceleration_sq_dy = 2 * m_acceleration_dy * m_acceleration[:, :, :, np.newaxis]

        # sum - simple
        m_acceleration_sqs = np.sum(m_acceleration_sq, axis=-1, keepdims=True)
        m_acceleration_sqs_dy = np.sum(m_acceleration_sq_dy, axis=-2, keepdims=True)
        
        # sqrt(x)**3 dx = x^1.5 dx = 1.5 x^0.5 = 1.5 sqrt(x)
        # 2/3 sqrt(m_acceleration_sqs) * m_acceleration_sqs_dy
        m_acceleration_norm = m_acceleration_sqs ** 1.5
        m_acceleration_norm_dy = 1.5 * np.sqrt(m_acceleration_sqs)[:, :, :, np.newaxis] * m_acceleration_sqs_dy
        # m_acceleration /= np.where(m_acceleration_norm <= 0.0, np.inf, m_acceleration_norm)
        m_acceleration_norm = np.where(m_acceleration_norm <= 0.0, np.inf, m_acceleration_norm)
        # f(x) / g(x) dx = (f'(x)g(x) - f(x)g'(x)) / (g(x)**2)
        m_acceleration_dy = np.nan_to_num(m_acceleration_dy * m_acceleration_norm[:, :, :, np.newaxis] - m_acceleration[:, :, :, np.newaxis] * m_acceleration_norm_dy, nan=0.0) / np.square(m_acceleration_norm[:, :, :, np.newaxis])
        m_acceleration /= m_acceleration_norm
        # Multiplication by a constant :)
        m_acceleration *= Gm2p
        m_acceleration_dy *= Gm2p[:, :, :, np.newaxis]
        
        # the net force vector is hence
        # net_acceleration = np.sum(m_acceleration, axis=1)
        net_acceleration_dy = np.sum(m_acceleration_dy, axis=1)

        # d_state = np.concatenate([velocities.reshape(n_states, -1),
        #                           net_acceleration.reshape(n_states, -1)], axis=1)
        # velocities.reshape(n_states, -1) - already done
        jac[half_state_elem:, :half_state_elem] = net_acceleration_dy.reshape(half_state_elem, half_state_elem)
        return jac

    if jacobian:
        return n_body_ode_eff, n_body_ode_eff_jac

    return n_body_ode_eff
