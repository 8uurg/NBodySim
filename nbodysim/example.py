import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# import the implementation
from nbodysim import make_n_body_ode_eff

# Some reference values for the example systems below!
mass_sun = 1.988475e30 # kg
mass_earth = 5.9722e24 # kg
mass_moon = 7.348e22 # kg

# earth_velocity_rel_sun = 29.78e3 #m/s - or 29.78 km / s
# sun_earth_distance = 149.60e9 #m - or 149.60e6 km
moon_velocity_rel_earth = 1.022e3 #m/s - or 1.022 km / s
earth_moon_distance = 385e6 #m - or 385000 km 

# Tuned to get roughly equivalent properties.
earth_velocity_rel_sun = 30.31e3 #m/s
sun_earth_distance = 147.160e9 #m
moon_velocity_rel_earth = 1.1086e3 #m/s - or 1.022 km / s
earth_moon_distance = 385e6 #m - or 385000 km 


def get_system_sun_earth_moon():
    entity_name = ["Sun", "Earth", "Moon"]
    entity_color = ["yellow", "blue", "gray"]
    
    masses = np.array([mass_sun, mass_earth, mass_moon])

    # Start in a simple lunar eclipse configuration, keeps things simple.
    initial_positions = np.array([
        [0.0, 0.0], # sun
        [0.0, sun_earth_distance], # earth
        [0.0, sun_earth_distance + earth_moon_distance], # moon
    ])
    # For ease, going with the average velocity as the starting velocity.
    # perpendicular to the offset direction
    initial_velocities = np.array([
        [0.0, 0.0],
        [earth_velocity_rel_sun, 0.0],
        [earth_velocity_rel_sun + moon_velocity_rel_earth, 0.0],
    ])
    return entity_name, entity_color, masses, initial_positions, initial_velocities

def get_system_sun_earth():
    entity_name = ["Sun", "Earth"]
    entity_color = ["yellow", "blue"]
    
    masses = np.array([mass_sun, mass_earth])

    # Start in a simple lunar eclipse configuration, keeps things simple.
    initial_positions = np.array([
        [0.0, 0.0], # sun
        [0.0, sun_earth_distance], # earth
    ])
    # For ease, going with the average velocity as the starting velocity.
    # perpendicular to the offset direction
    initial_velocities = np.array([
        [0.0, 0.0],
        [earth_velocity_rel_sun, 0.0],
    ])
    return entity_name, entity_color, masses, initial_positions, initial_velocities
    

def get_system_earth_moon():
    entity_name = ["Earth", "Moon"]
    entity_color = ["blue", "gray"]
    
    masses = np.array([mass_earth, mass_moon])

    # Start in a simple lunar eclipse configuration, keeps things simple.
    initial_positions = np.array([
        [0.0, 0.0], # earth
        [0.0, earth_moon_distance], # moon
    ])
    # For ease, going with the average velocity as the starting velocity.
    # perpendicular to the offset direction
    initial_velocities = np.array([
        [0.0, 0.0],
        [moon_velocity_rel_earth, 0.0],
    ])
    return entity_name, entity_color, masses, initial_positions, initial_velocities

def main():
    # Default tolerances are insufficient - increase tolerances to avoid degradation.
    rtol = 1e-5 # defaults to 1e-3
    atol = 1e-10 # defaults to 1e-6
    # As an example, here is the earth-moon-sun system (approximately)
    entity_name, entity_color, masses, initial_positions, initial_velocities = get_system_sun_earth_moon()
    # entity_name, entity_color, masses, initial_positions, initial_velocities = get_system_sun_earth()
    # entity_name, entity_color, masses, initial_positions, initial_velocities = get_system_earth_moon()
    
    center_on = None
    # center_on = "Earth"

    # Determine if we wish to center the plot
    if isinstance(center_on, str):
        try:
            center_on = entity_name.index(center_on)
        except ValueError as _e:
            pass

    # For my own use!
    n_bodies = initial_velocities.shape[0]
    d = initial_velocities.shape[1]

    initial_state = np.concatenate([initial_positions.ravel(), initial_velocities.ravel()])

    # provide the masses as a constant to the ODE.
    # ode = partial(n_body_ode_reference, masses=masses)
    ode = make_n_body_ode_eff(masses)

    # units are seconds
    # t_span = [0.0, float(60.0 * 60.0 * 24.0)] # a day
    t_span = [0.0, float(60 * 60 * 24 * 366)] # a year
    # t_span = [0.0, float(60 * 60 * 24 * 366 * 4)] # 4 years
    # t_span = [0.0, float(60 * 60 * 24 * 366 * 30)] # 30 years


    # use a ODE solver to do the difficult job of simulating the system
    # reference implementation does not allow for vectorization.
    res = solve_ivp(ode, t_span, y0=initial_state, dense_output=True, vectorized=True, method="DOP853", rtol=rtol, atol=atol)
    # res is a Bunch object with the following fields defined:
    # - t : ndarray, shape (n_points,) Time points.
    # - y : ndarray, shape (n, n_points) Values of the solution at t.
    # - sol : OdeSolution or None Found solution as OdeSolution instance; None if dense_output was set to False.
    # - t_events : list of ndarray or None Contains for each event type a list of arrays at which an event of that type event was detected. None if events was None.
    # - y_events : list of ndarray or None For each value of t_events, the corresponding value of the solution. None if events was None.
    # - nfev : int Number of evaluations of the right-hand side.
    # - njev : int Number of evaluations of the Jacobian.
    # - nlu : int Number of LU decompositions
    # - status : int Solver status

    # print(f"{res.y.shape}") # should be (12, ...), (3 x 2 positions + 3 x 2 velocities)
    positions = res.y[:n_bodies * d, :].reshape(n_bodies, d, -1)
    
    
    # Plot positions & history
    fig, axs = plt.subplots(1, 1, squeeze=False)
    axs[0, 0].set_aspect('equal')


    t = res.t
    # Start markers
    initial_positions_plt = initial_positions
    if center_on is not None:
        initial_positions_plt = initial_positions_plt - initial_positions[[center_on], :]

    for body_idx in range(n_bodies):
        body_color = entity_color[body_idx]
        axs[0, 0].scatter(initial_positions_plt[[body_idx], 0], initial_positions_plt[[body_idx], 1], color=body_color, marker="*")

    # State points (at which the ODE solver paused to adjust for errors), not necessary monotonically spaced.
    # Used for debugging the interpolation portion in the next part.
    # for body_idx in range(n_bodies):
    #     # body_name = entity_name[body_idx]
    #     body_color = entity_color[body_idx]
    #     plt.scatter(positions[body_idx, 0, :], positions[body_idx, 1, :], color=body_color, alpha=0.4)

    # Plot trajectory with linear interpolation
    # for body_idx in range(n_bodies):
    #     body_color = entity_color[body_idx]
    #     plt.plot(positions[body_idx, 0, :], positions[body_idx, 1, :], color=body_color, alpha=0.9)
    
    # So let's do this using interpolation instead
    tx = np.linspace(0, t.max(), 100000)
    state_interp = res.sol(tx)
    positions_interp = state_interp[:n_bodies * d, :].reshape(n_bodies, d, -1)

    if center_on is not None:
        positions_interp = positions_interp - positions_interp[[center_on], :, :]

    # plot distance between two points - disabled for now
    # r = np.linalg.norm(positions_interp[0, :, :] - positions_interp[1, :, :], axis=0)
    # axs[0, 1].plot(r)

    for body_idx in range(n_bodies):
        body_color = entity_color[body_idx]
        axs[0, 0].plot(positions_interp[body_idx, 0, :], positions_interp[body_idx, 1, :], color=body_color, alpha=0.9)

    positions_plt = positions
    if center_on is not None:
        positions_plt = positions_plt - positions_plt[[center_on], :, :]

    # End markers
    for body_idx in range(n_bodies):
        body_name = entity_name[body_idx]
        body_color = entity_color[body_idx]
        axs[0, 0].scatter(positions_plt[[body_idx], 0, -1], positions_plt[[body_idx], 1, -1], label=body_name, color=body_color)
    
    axs[0, 0].legend()
    axs[0, 0].set_facecolor("black")
    plt.show()


if __name__ == "__main__":
    main()
