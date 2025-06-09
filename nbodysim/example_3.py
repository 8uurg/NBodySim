import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import chain
# import the implementation
from nbodysim import make_n_body_ode_eff

# Three similar masses

def get_simple_system():
    entity_name = ["A", "B", "C"]
    entity_color = ["red", "green", "blue"]
    
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
    return entity_name, entity_color, masses, initial_positions, initial_velocities

def main():
    # Default tolerances are insufficient - increase tolerances to avoid degradation.
    rtol = 1e-5 # defaults to 1e-3
    atol = 1e-10 # defaults to 1e-6
    entity_name, entity_color, masses, initial_positions, initial_velocities = get_simple_system()
    
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
    ode, ode_jac = make_n_body_ode_eff(masses, jacobian=True)

    # units are seconds
    # t_span = [0.0, float(60.0 * 60.0 * 24.0)] # a day
    t_span = [0.0, float(60.0 * 60.0 * 24.0 * 3)] # 3 days
    # t_span = [0.0, float(60.0 * 60.0 * 24.0 * 4)] # 4 days
    # t_span = [0.0, float(60.0 * 60.0 * 24.0 * 7)] # 1 week
    # t_span = [0.0, float(60.0 * 60.0 * 24.0 * 9)] # 9 days
    # t_span = [0.0, float(60.0 * 60.0 * 24.0 * 10)] # 10 days


    # use a ODE solver to do the difficult job of simulating the system
    # reference implementation does not allow for vectorization.
    res = solve_ivp(ode, t_span, y0=initial_state, jac=ode_jac, dense_output=True, vectorized=True, method="LSODA", rtol=rtol, atol=atol)
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

    # So let's do this using interpolation instead
    tx = np.linspace(0, t.max(), 100000)
    state_interp = res.sol(tx)
    positions_interp = state_interp[:n_bodies * d, :].reshape(n_bodies, d, -1)

    if center_on is not None:
        positions_interp = positions_interp - positions_interp[[center_on], :, :]

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


    # Create animation
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_facecolor("black")
    

    # 'Tails'
    line_artists: list[plt.Line2D] = []
    for body_idx in range(n_bodies):
        body_color = entity_color[body_idx]
        line_artist = ax.plot(positions_interp[body_idx, 0, :], positions_interp[body_idx, 1, :], color=body_color, alpha=0.9)
        
        line_artists.append(line_artist[0])
    
    point_artists: list[plt.PathCollection] = []
    # Position markers
    for body_idx in range(n_bodies):
        body_name = entity_name[body_idx]
        body_color = entity_color[body_idx]
        point_artist = ax.scatter(positions_plt[[body_idx], 0, -1], positions_plt[[body_idx], 1, -1], label=body_name, color=body_color)
        point_artists.append(point_artist)
    
    plt.legend()

    tail_length = 500
    def animation_function(frame):
        print(f"Rendering frame {frame}")
        start_frame = max(frame - tail_length, 0)
        for body_idx in range(n_bodies):
            line_artists[body_idx].set_data(positions_interp[body_idx, 0, start_frame:frame], positions_interp[body_idx, 1, start_frame:frame])
            point_artists[body_idx].set_offsets(positions_interp[[body_idx], :, frame])
                                                
        return chain(line_artists, point_artists)

    ani = animation.FuncAnimation(fig, animation_function, range(0, len(tx), 50), blit=True, interval=10)
    ani.save("nbodysimulation-lsoda.mp4")
        


if __name__ == "__main__":
    main()
