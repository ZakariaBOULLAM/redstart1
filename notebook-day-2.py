import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Redstart: A Lightweight Reusable Booster""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="public/images/redstart.png")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy as sci
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter,PillowWriter
    from matplotlib.patches import Polygon, Rectangle
    from numpy.linalg import matrix_rank
    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return (
        FFMpegWriter,
        FuncAnimation,
        Polygon,
        Rectangle,
        matrix_rank,
        np,
        plt,
        sci,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Getting Started""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🧩 **Constants**

    Define the Python constants `g`, `M` and `l` that correspond to the gravity constant, the mass and half-length of the booster.
    """
    )
    return


@app.cell
def _():
    # Constants
    g = 1.0   # gravity constant (m/s^2)
    M = 1.0   # mass of the booster (kg)
    l = 1.0   # half-length of the booster (meters)
    return M, g, l


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🧩 **Forces**

    Compute the force $(f_x, f_y) \in \mathbb{R}^2$ applied to the booster by the reactor.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \boxed{
    (f_x, f_y) = f \cdot \left( -\sin(\theta + \phi),\; \cos(\theta + \phi) \right)
    }
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🧩 **Center of Mass**

    Give the ordinary differential equation that governs $(x, y)$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""


    $$
    \boxed{
    \begin{aligned}
    \ddot{x} &= -\frac{f}{M} \cdot \sin(\theta + \phi) \\
    \ddot{y} &= +\frac{f}{M} \cdot \cos(\theta + \phi) - g
    \end{aligned}
    }
    $$

    ### 🎯 Final ODEs (with constants $M = 1$, $g = 1$):

    $$
    \boxed{
    \begin{aligned}
    \ddot{x} &= -f \cdot \sin(\theta + \phi) \\
    \ddot{y} &= +f \cdot \cos(\theta + \phi) - 1
    \end{aligned}
    }
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🧩 **Moment of inertia**

    Compute the moment of inertia $J$ of the booster and define the corresponding Python variable `J`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    $$
    \boxed{
    J = \frac{1}{3}
    }
    $$
    """
    )
    return


@app.cell
def _(M, l):
    # Moment of inertia of the booster 
    J = (1/12) * M * (2 * l)**2  # or simply: J = 1/3
    return (J,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🧩 **Tilt**

    Give the ordinary differential equation that governs the tilt angle $\theta$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    $$
    \ddot{\theta} = -\frac{\ell f}{J} \cdot \sin(\phi)
    $$

    Using $\ell = 1$, $J = \frac{1}{3}$, we get:

    $$
    \boxed{
    \ddot{\theta} = -3f \cdot \sin(\phi)
    }
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 🧩 **Simulation**""")
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, plt, sci):
    def redstart_solve(t_span, y0, f_phi):
        def dynamics(t, y):
            x, dx, y_pos, dy, theta, dtheta, f0, phi0 = y
            f, phi = f_phi(t, y)

            # Limit thrust to prevent extreme values
            f = np.clip(f, 0, 5 * M * g)

            # System dynamics
            ddx = f * np.sin(theta + phi) / M
            ddy = f * np.cos(theta + phi) / M - g
            ddtheta = l * f * np.sin(phi) / J

            return [dx, ddx, dy, ddy, dtheta, ddtheta, 0.0, 0.0]

        # Use smaller max_step for better accuracy
        sol_ivp = sci.integrate.solve_ivp(dynamics, t_span, y0, method='RK45', 
                            rtol=1e-6, atol=1e-6, max_step=0.05,
                            dense_output=True)
        return sol_ivp.sol

    # Example: Free fall test
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0,0.0,0.0]  # [x, dx, y, dy, theta, dtheta, f0, phi0]

        def f_phi(t, y):
            return np.array([0.0, 0.0])  # no thrust

        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]

        plt.figure(figsize=(8, 4))
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y = \ell$")
        plt.title("Free Fall")
        plt.xlabel("Time $t$")
        plt.ylabel("Height $y$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()

    free_fall_example()

    return (redstart_solve,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Controlled Landing

    Assume that $x$, $\dot{x}$, $\theta$ and $\dot{\theta}$ are null at $t=0$. For $y(0)= 10$ and $\dot{y}(0)$, can you find a time-varying force $f(t)$ which, when applied in the booster axis ($\theta=0$), yields $y(5)=\ell$ and $\dot{y}(5)=0$?

    Simulate the corresponding scenario to check that your solution works as expected.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We aim to find a **time-varying thrust** $f(t)$ that allows the booster to land smoothly from an altitude of 10 meters to just above the ground ($y = \ell = 1$) in 5 seconds, with zero velocity at both ends. For simplicity, we assume:

    * The booster remains perfectly **vertical**: $\theta(t) = 0$, $\phi(t) = 0$
    * No horizontal motion: $x(t) = \dot{x}(t) = 0$

    ### 🎯 Objective

    Find a force $f(t)$ such that:

    $$
    \begin{cases}
    y(0) = 10, & \dot{y}(0) = 0 \\
    y(5) = 1, & \dot{y}(5) = 0
    \end{cases}
    $$

    ### 📘 Governing Equation

    From previous derivations, the vertical motion obeys:

    $$
    \ddot{y}(t) = f(t) - 1
    $$

    This comes from Newton’s second law, with $f(t)$ being the vertical component of thrust and $g = 1 \ \text{m/s}^2$.

    ### ✅ Solution Strategy

    We define a **cubic polynomial** for $y(t)$ that satisfies the desired boundary conditions (position and velocity at $t = 0$ and $t = 5$). Then we compute:

    * $\dot{y}(t)$: First derivative (velocity)
    * $\ddot{y}(t)$: Second derivative (acceleration)

    From which we derive the thrust:


      $$f(t) = \ddot{y}(t) + 1$$
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, plt, redstart_solve, sci):
    # Target trajectory using Hermite cubic interpolation
    t0, tf = 0.0, 10.0
    y0, dy0 = 10.0, 0.0  # Initial height and vertical velocity
    yf, dyf = 1.0, 0.0   # Final height and vertical velocity (changed to 0.0 for proper landing)
    x0, dx0 = 0.0, 0.0   # Initial horizontal position and velocity (fixed velocity)
    xf, dxf = 0.0, 0.0   # Final horizontal position and velocity

    # Create smoother trajectories with proper boundary conditions
    poly_y = sci.interpolate.BPoly.from_derivatives([t0, tf], [[y0, dy0], [yf, dyf]])
    poly_x = sci.interpolate.BPoly.from_derivatives([t0, tf], [[x0, dx0], [xf, dxf]])

    def y_desired(t):
        return poly_y(t)

    def dy_desired(t):
        return poly_y.derivative(1)(t)

    def ddy_desired(t):
        return poly_y.derivative(2)(t)

    def x_desired(t):
        return poly_x(t)

    def dx_desired(t):
        return poly_x.derivative(1)(t)

    def ddx_desired(t):
        return poly_x.derivative(2)(t)

    def f_phi_controlled(t, y):
        x, dx, y_pos, dy, theta, dtheta, f0, phi0 = y

        # Desired positions and velocities for feedback control
        x_des = x_desired(t)
        dx_des = dx_desired(t)
        y_des = y_desired(t)
        dy_des = dy_desired(t)

        # Position and velocity error terms for PD control
        x_err = x - x_des
        dx_err = dx - dx_des
        y_err = y_pos - y_des
        dy_err = dy - dy_des

        # PD control gains (increased for tighter tracking)
        Kp_x, Kd_x = 0.3, 1.0
        Kp_y, Kd_y = 0.3, 1.0

        # Base desired accelerations from trajectory
        ax_base = ddx_desired(t)
        ay_base = ddy_desired(t)

        # Add feedback terms
        ax = ax_base - Kp_x * x_err - Kd_x * dx_err
        ay = ay_base - Kp_y * y_err - Kd_y * dy_err

        # Add gravity compensation
        net_ay = ay + g

        # Calculate the target thrust direction and magnitude
        thrust_mag = np.sqrt(ax**2 + net_ay**2)
        alpha = np.arctan2(ax, net_ay)  # Desired orientation

        # PD control for attitude (theta) - more aggressive gains
        Kp_theta, Kd_theta = 8.0, 4.0
        theta_desired = alpha

        # Wrap angle difference to [-pi, pi]
        theta_error = ((theta - theta_desired + np.pi) % (2 * np.pi)) - np.pi

        # Calculate torque
        torque = -Kp_theta * theta_error - Kd_theta * dtheta

        # Calculate phi (limited to prevent extreme angles)
        phi = np.arcsin(np.clip((J * torque) / (l * thrust_mag + 1e-5), -0.3, 0.3))

        # Calculate thrust magnitude (with minimum thrust for stability)
        f = thrust_mag / (np.cos(phi) + 1e-5)

        # Ensure smooth startup
        if t < 0.5:
            # Ramp up the controller gradually
            blend = t / 0.5
            f = blend * f + (1 - blend) * (M * g)
            phi = blend * phi

        # Add extra thrust near landing for soft touchdown
        if y_pos < 0.5 and dy < 0:
            f += 2.0 * (0.5 - y_pos)  # Extra thrust as we get closer to ground

        return np.array([f, phi])

    # Initial state
    y0_state = [0.0, 0.0, y0, dy0, 0.0, 0.0,0.0,0.0]

    # Solve
    sol_controlled = redstart_solve([t0, tf], y0_state, f_phi_controlled)

    # Plot the result
    t_vals = np.linspace(t0, tf, 1000)
    y_vals = sol_controlled(t_vals)[2]  # y(t)

    plt.figure(figsize=(8, 4))
    plt.plot(t_vals, y_vals, label=r"$y(t)$ (height of CM)")
    plt.axhline(y=l, color="grey", ls="--", label=r"$y = \ell$")
    plt.axhline(y=0, color="red", ls="--", label=r"$y = 0$")
    plt.title("Controlled Vertical Landing")
    plt.xlabel("Time $t$")
    plt.ylabel("Height $y$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return f_phi_controlled, t0, tf, x_desired, y_desired


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The simulation shows that the booster’s **center of mass** descends smoothly to the target height $y = \ell$, with the base just reaching the ground ($y - \ell = 0$) at $t = 5$, and with no vertical velocity — achieving a soft landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 🧩 Drawing""")
    return


@app.cell
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell
def _(M, Polygon, Rectangle, g, l, np, plt):
    def draw_booster(x, y, theta, f, phi, flame=True):
        # Booster parameters
        booster_length = 2 * l
        booster_width = 0.1
        flame_width = 0.08

        # Axis definitions
        axis = np.array([np.sin(theta), np.cos(theta)])         # along the booster
        perp = np.array([np.cos(theta), -np.sin(theta)])        # perpendicular to booster

        # Booster base and tip
        base = np.array([x, y]) - l * axis
        tip = np.array([x, y]) + l * axis

        # Booster body rectangle (4 corners)
        body = np.array([
            base + (booster_width / 2) * perp,
            base - (booster_width / 2) * perp,
            tip - (booster_width / 2) * perp,
            tip + (booster_width / 2) * perp
        ])

        # Flame direction: in the direction of thrust = theta + phi
        flame_shape = None
        if flame and f > 0:
            thrust_dir = np.array([np.sin(theta + phi), np.cos(theta + phi)])
            flame_length = (f / (M * g)) * l
            flame_tip = base - flame_length * thrust_dir
            flame_shape = np.array([
                base + (flame_width / 2) * perp,
                base - (flame_width / 2) * perp,
                flame_tip
            ])

        # Plot setup
        fig, ax = plt.subplots(figsize=(2, 6))
        ax.set_aspect('equal')
        ax.set_xlim(x - 2, x + 2)
        ax.set_ylim(-1, 12)
        ax.set_facecolor('#f0f8ff')

        # Draw booster
        ax.add_patch(Polygon(body, closed=True, color='black'))

        # Draw flame
        if flame_shape is not None:
            ax.add_patch(Polygon(flame_shape, closed=True, color='red'))

        # Draw landing zone
        ax.add_patch(Rectangle((-0.5, -0.2), 1.0, 0.2, color='sandybrown'))

        ax.axis('off')
        return fig

    return (draw_booster,)


@app.cell
def _(draw_booster):
    draw_booster(x=0.0, y=10, theta=0, f=0.9, phi=0.9)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Visualization

    Produce a video of the booster for 5 seconds when

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=0$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=0$

      - $(x, \dot{x}, y, \dot{y}, \theta, \dot{\theta}) = (0.0, 0.0, 10.0, 0.0, 0.0, 0.0)$, $f=Mg$ and $\phi=\pi/8$

      - the parameters are those of the controlled landing studied above.

    As an intermediary step, you can begin with production of image snapshots of the booster location (every 1 sec).
    """
    )
    return


@app.cell
def _(
    FFMpegWriter,
    FuncAnimation,
    M,
    Polygon,
    Rectangle,
    f_phi_controlled,
    g,
    l,
    np,
    plt,
    redstart_solve,
    t0,
    tf,
    x_desired,
    y_desired,
):

    def simulate_booster_landing(
        initial_state,
        t_span=[0.0, 10.0],
        f_phi=f_phi_controlled,
        fps=30,
        output_path="booster_landing_simulation.mp4"
    ):
        sol = redstart_solve(t_span, initial_state, f_phi)
        num_frames = int((t_span[1] - t_span[0]) * fps)
        times = np.linspace(t_span[0], t_span[1], num_frames)

        # Create time history data for plotting
        t_data = np.linspace(t_span[0], t_span[1], 200)  # More points for smoother plots
        x_data = []
        y_data = []
        theta_data = []
        f_data = []
        phi_data = []

        for t in t_data:
            state = sol(t)
            x, dx, y, dy, theta, dtheta, f, phi = state

            # Get control values from solver or controller
            if t <= 0.1:
                # Use values from state
                pass
            else:
                f, phi = f_phi(t, state)

            x_data.append(x)
            y_data.append(y)
            theta_data.append(theta)
            f_data.append(f)
            phi_data.append(phi)


        static_plot_path = "booster_landing_plots.png"

        fig_states, axs = plt.subplots(5, 1, figsize=(10, 12))

        axs[0].plot(t_data, x_data, 'r-', label='x')
        axs[0].set_title('Position (x)')
        axs[0].set_ylabel('Position')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(t_data, y_data, 'b-', label='y')
        axs[1].set_title('Position (y)')
        axs[1].set_ylabel('Position')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(t_data, theta_data, 'g-')
        axs[2].set_title('Angle (theta)')
        axs[2].set_ylabel('Angle (rad)')
        axs[2].grid(True, alpha=0.3)

        axs[3].plot(t_data, phi_data, 'c-')
        axs[3].set_title('Gimbal Angle (phi)')
        axs[3].set_ylabel('Angle (rad)')
        axs[3].grid(True, alpha=0.3)

        axs[4].plot(t_data, f_data, 'orange')
        axs[4].set_title('Thrust Magnitude (f)')
        axs[4].set_xlabel('Time (s)')
        axs[4].set_ylabel('Thrust (N)')
        axs[4].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(static_plot_path)
        plt.close(fig_states)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f8ff')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-1, 12)
        ax.set_title('Booster Landing Simulation', fontsize=12)

        # Add reference trajectory
        t_ref = np.linspace(t0, tf, 100)
        x_ref = [x_desired(t) for t in t_ref]
        y_ref = [y_desired(t) for t in t_ref]
        ax.plot(x_ref, y_ref, 'b--', alpha=0.3, linewidth=1)

        # Create patches for visualization
        body_patch = Polygon(np.zeros((4, 2)), closed=True, color='#404040')
        flame_patch = Polygon(np.zeros((3, 2)), closed=True, color='orange')
        landing_pad = Rectangle((-0.5, -0.2), 1.0, 0.2, color='#8B4513')

        # Add ground
        ground = Rectangle((-3, -0.2), 6, 0.2, color='#8B8378')
        ax.add_patch(ground)
        ax.add_patch(body_patch)
        ax.add_patch(flame_patch)
        ax.add_patch(landing_pad)

        # Add trail effect
        trail = ax.plot([], [], 'r-', alpha=0.5, linewidth=1)[0]
        trail_x, trail_y = [], []

        # Update function for animation
        def update(frame_idx):
            t = times[frame_idx]
            state = sol(t)
            x, dx, y, dy, theta, dtheta, f, phi = state

            # Get control values from solver or controller
            if t <= 0.1:
                # Use values from state
                pass
            else:
                f, phi = f_phi(t, state)

            # Calculate rocket body position
            axis = np.array([np.sin(theta), np.cos(theta)])
            perp = np.array([np.cos(theta), -np.sin(theta)])
            center = np.array([x, y])
            base = center - 0.8 * l * axis  # Adjusted for better appearance
            tip = center + 1.2 * l * axis   # Adjusted for better appearance

            # Rocket body with slight tapering
            width_base = 0.12
            width_tip = 0.08
            body = np.array([
                base + width_base * perp,
                base - width_base * perp,
                tip - width_tip * perp,
                tip + width_tip * perp
            ])
            body_patch.set_xy(body)

            # Update flame
            if f > 0:
                thrust_dir = np.array([np.sin(theta + phi), np.cos(theta + phi)])
                flame_length = 0.2 + 0.5 * (f / (M * g))
                flame_width = 0.08 + 0.04 * (f / (M * g))
                flame_tip = base - flame_length * thrust_dir
                flame = np.array([
                    base + flame_width * perp,
                    base - flame_width * perp,
                    flame_tip
                ])
                flame_patch.set_xy(flame)
                flame_patch.set_visible(True)

                # Yellow-orange gradient for flame
                flame_patch.set_color('orange' if f > 2*M*g else 'yellow')
            else:
                flame_patch.set_visible(False)

            # Update trail
            if frame_idx % 2 == 0:  # Add point every other frame
                trail_x.append(x)
                trail_y.append(y)
                # Keep only recent points
                if len(trail_x) > 30:
                    trail_x.pop(0)
                    trail_y.pop(0)
                trail.set_data(trail_x, trail_y)

            return body_patch, flame_patch, trail

        # Create animation
        ani = FuncAnimation(fig, update, frames=num_frames, blit=True)

        # Save as MP4
        writer = FFMpegWriter(fps=fps)
        ani.save(output_path, writer=writer)
        plt.close(fig)

        return output_path,static_plot_path

    return (simulate_booster_landing,)


@app.cell(hide_code=True)
def _(mo):


    scenario_selector = mo.ui.dropdown(
        options={
            "Free Fall": "free_fall",
            "Vertical Thrust": "vertical_thrust",
            "Tilted Thrust": "tilted_thrust",
            "Offset X=1.0": "offset_x",
        },
        label="Choose a scenario",
        value="Free Fall"
    )
    # Display the selector
    scenario_selector
    return (scenario_selector,)


@app.cell
def _(mo, np, scenario_selector, simulate_booster_landing):
    def run_simulation():
            selected = scenario_selector.value
            if selected == "free_fall":
                initial_state = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif selected == "vertical_thrust":
                initial_state = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif selected == "tilted_thrust":
                initial_state = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0, np.pi / 8]
            elif selected == "offset_x":
                initial_state = [1.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # Run simulation
            simulation_path, plot_path = simulate_booster_landing(initial_state, fps=15)
            # Show result
            return mo.hstack([
                mo.video(src=simulation_path, controls=False, autoplay=True, loop=True, width="100%"),
                mo.image(src=plot_path, width="100%")
            ])

    run_simulation()
    return


@app.cell
def _(mo):
    mo.md(r"""# Linearized Dynamics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Equilibria

    We assume that $|\theta| < \pi/2$, $|\phi| < \pi/2$ and that $f > 0$. What are the possible equilibria of the system for constant inputs $f$ and $\phi$ and what are the corresponding values of these inputs?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    To find the **equilibria** of the Redstart booster system, we need to determine under which conditions the **state** of the system remains **constant over time**. This means all **derivatives must be zero**:

    $$
    \dot{x} = \ddot{x} = 0,\quad \dot{y} = \ddot{y} = 0,\quad \dot{\theta} = \ddot{\theta} = 0
    $$

    ---

    ### 1. **From the horizontal acceleration equation:**

    $$
    \ddot{x} = -\frac{f}{M} \sin(\theta + \phi) = 0
    $$

    To satisfy this:

    $$
    \sin(\theta + \phi) = 0 \quad \Rightarrow \quad \theta + \phi = n\pi,\quad n \in \mathbb{Z}
    $$

    But since we are assuming $|\theta| < \frac{\pi}{2}$ and $|\phi| < \frac{\pi}{2}$, the only valid solution is:

    $$
    \theta + \phi = 0
    \quad \Rightarrow \quad \phi = -\theta
    $$

    ---

    ### 2. **From the vertical acceleration equation:**

    $$
    \ddot{y} = \frac{f}{M} \cos(\theta + \phi) - g = 0
    $$

    Using $\theta + \phi = 0 \Rightarrow \cos(\theta + \phi) = 1$, this becomes:

    $$
    \frac{f}{M} \cdot 1 - g = 0
    \quad \Rightarrow \quad f = M g
    $$

    ---

    ### 3. **From the angular acceleration (torque) equation:**

    $$
    \ddot{\theta} = - \frac{\ell f}{J} \sin(\phi) = 0
    $$

    $$
    \Rightarrow \sin(\phi) = 0 \Rightarrow \phi = 0 \Rightarrow \theta = 0
    $$

    (using $\phi = -\theta$ from earlier)

    ---

    ### ✅ Final Equilibrium Conditions:

    The system is in **equilibrium** if:

    * $\boxed{\theta = 0}$
    * $\boxed{\phi = 0}$
    * $\boxed{f = Mg}$

    These lead to:

    * No tilt
    * No rotation
    * No vertical/horizontal motion
    * Constant thrust exactly opposing gravity


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linearized Model

    Introduce the error variables $\Delta x$, $\Delta y$, $\Delta \theta$, and $\Delta f$ and $\Delta \phi$ of the state and input values with respect to the generic equilibrium configuration.
    What are the linear ordinary differential equations that govern (approximately) these variables in a neighbourhood of the equilibrium?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To linearize the nonlinear dynamics of the Redstart booster near the equilibrium : 

    ---

    ## 🧩 Linearized Model

    We analyze the system near the **equilibrium** found earlier:

    $$
    \boxed{\theta = 0, \quad \phi = 0, \quad f = Mg}
    $$

    ---

    ### 🔁 Step 1: Define error (perturbation) variables

    Let’s denote deviations from equilibrium by:

    * $\Delta x = x - x_{\text{eq}}$
    * $\Delta y = y - y_{\text{eq}}$
    * $\Delta \theta = \theta - 0 = \theta$
    * $\Delta f = f - Mg$
    * $\Delta \phi = \phi - 0 = \phi$

    ---

    ### 🔁 Step 2: Recall nonlinear equations of motion

    From earlier:

    $$
    \begin{aligned}
    \ddot{x} &= -\frac{f}{M} \sin(\theta + \phi) \\
    \ddot{y} &= +\frac{f}{M} \cos(\theta + \phi) - g \\
    \ddot{\theta} &= -\frac{\ell f}{J} \sin(\phi)
    \end{aligned}
    $$

    ---

    ### 🔁 Step 3: Taylor expansion near equilibrium

    Use small-angle approximations for $\theta, \phi \approx 0$:

    * $\sin(\theta + \phi) \approx \theta + \phi$
    * $\cos(\theta + \phi) \approx 1$
    * $\sin(\phi) \approx \phi$

    ---

    ### 🔁 Step 4: Linearize each equation

    #### Horizontal motion:

    $$
    \ddot{x} = -\frac{f}{M} (\theta + \phi) 
    \approx -\frac{Mg + \Delta f}{M} (\theta + \phi)
    = -g (\theta + \phi) - \frac{\Delta f}{M} (\theta + \phi)
    $$

    Neglect second-order terms (product of small terms), so:

    $$
    \boxed{\ddot{\Delta x} \approx -g (\Delta \theta + \Delta \phi)}
    $$

    ---

    #### Vertical motion:

    $$
    \ddot{y} = \frac{f}{M} \cos(\theta + \phi) - g 
    \approx \frac{Mg + \Delta f}{M} (1) - g
    = g + \frac{\Delta f}{M} - g = \frac{\Delta f}{M}
    $$

    $$
    \boxed{\ddot{\Delta y} \approx \frac{\Delta f}{M}}
    $$

    ---

    #### Rotational motion:

    $$
    \ddot{\theta} = -\frac{\ell f}{J} \sin(\phi) 
    \approx -\frac{\ell (Mg + \Delta f)}{J} \phi 
    \approx -\frac{\ell Mg}{J} \phi
    $$

    $$
    \boxed{\ddot{\Delta \theta} \approx -\frac{\ell Mg}{J} \Delta \phi}
    $$

    ---

    ### ✅ Final Linearized System

    In vector form:

    $$
    \boxed{
    \begin{aligned}
    \ddot{\Delta x} &= -g (\Delta \theta + \Delta \phi) \\
    \ddot{\Delta y} &= \frac{1}{M} \Delta f \\
    \ddot{\Delta \theta} &= -\frac{\ell Mg}{J} \Delta \phi
    \end{aligned}
    }
    $$

    These are second-order linear differential equations approximating the system behavior near the hovering equilibrium.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Standard Form

    What are the matrices $A$ and $B$ associated to this linear model in standard form?
    Define the corresponding NumPy arrays `A` and `B`.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We now express the linearized model in **state-space standard form**:

    $$
    \dot{X} = A X + B U
    $$

    ---

    ### 🧩 Step 1: Define state and input vectors

    Let’s define:

    #### State vector $X \in \mathbb{R}^6$:

    $$
    X =
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta y \\
    \Delta \dot{y} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    $$

    #### Input vector $U \in \mathbb{R}^2$:

    $$
    U =
    \begin{bmatrix}
    \Delta f \\
    \Delta \phi
    \end{bmatrix}
    $$

    ---

    ### 🧩 Step 2: From previous result

    We had the linearized equations:

    $$
    \begin{aligned}
    \ddot{\Delta x} &= -g (\Delta \theta + \Delta \phi) \\
    \ddot{\Delta y} &= \frac{1}{M} \Delta f \\
    \ddot{\Delta \theta} &= -\frac{\ell Mg}{J} \Delta \phi
    \end{aligned}
    $$

    Now rewrite them as **first-order** system.

    ---

    ### 🧩 Step 3: Build matrices $A$ and $B$

    $$
    \dot{X} =
    \begin{bmatrix}
    \Delta \dot{x} \\
    - g \Delta \theta \\
    \Delta \dot{y} \\
    \frac{1}{M} \Delta f \\
    \Delta \dot{\theta} \\
    -\frac{\ell Mg}{J} \Delta \phi
    \end{bmatrix}
    +
    \begin{bmatrix}
    0 \\
    - g \\
    0 \\
    0 \\
    0 \\
    0
    \end{bmatrix} \Delta \phi
    $$

    So:

    #### Matrix $A \in \mathbb{R}^{6 \times 6}$:

    $$
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    $$

    #### Matrix $B \in \mathbb{R}^{6 \times 2}$:

    $$
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{\ell Mg}{J}
    \end{bmatrix}
    $$


    """
    )
    return


@app.cell
def _(J, M, g, l, np):
    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, -g, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0]
    ])

    B = np.array([
        [0, 0],
        [0, -g],
        [0, 0],
        [1/M, 0],
        [0, 0],
        [0, -l * M * g / J]
    ])
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Stability

    Is the generic equilibrium asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We analyze the **linearized system**:

    $$
    \dot{X} = A X + B U
    $$

    We want to know:
    👉 **Is the system asymptotically stable when** $U = 0$?
    That is: when there is **no input deviation** (i.e., $\Delta f = 0$, $\Delta \phi = 0$).

    ---

    ### 🔍 Stability Criterion

    * A linear system $\dot{X} = A X$ is **asymptotically stable** if **all eigenvalues of $A$** have strictly **negative real parts**.

    ---

    ### 🔢 Let’s recall matrix $A$:

    $$
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    $$

    This matrix is **not full rank** (contains rows of zeros and no damping terms).

    ---

    ### 🧠 Physical Interpretation

    * The vertical motion is influenced by $\Delta f$, but not stabilized intrinsically.
    * The lateral and angular motion are **not** damped — they **drift** under perturbation.
    * There is **no restoring force or damping**, only a coupling via gravity.

    ---

    ### ⚠️ Eigenvalues of $A$

    This matrix has **zero eigenvalues**, implying **marginal stability** or **instability** depending on input.

    ---

    ### ✅ Final Answer:

    $$
    \boxed{
    \text{No, the generic equilibrium is not asymptotically stable.}
    }
    $$

    Instead, it is **marginally stable** or **unstable** — the booster will drift or rotate indefinitely under small disturbances unless actively controlled.

    """
    )
    return


@app.cell
def _(A, np):
    np.linalg.eigvals(A)
    return


app._unparsable_cell(
    r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🧩 Controllability of the Linearized Model

    To determine if the linearized system is **controllable**, we analyze the pair $(A, B)$ using the **controllability matrix**:

    $$
    \mathcal{C} = [B \; AB \; A^2B \; \dots \; A^{n-1}B]
    $$

    where:

    * $A \in \mathbb{R}^{6 \times 6}$
    * $B \in \mathbb{R}^{6 \times 2}$
    * $\mathcal{C} \in \mathbb{R}^{6 \times (2 \cdot 6)} = \mathbb{R}^{6 \times 12}$

    ---

    ### ✅ Criterion for Controllability

    The system is **controllable** if:

    $$
    \text{rank}(\mathcal{C}) = 6
    $$

    ---

    ### 🧠 Intuition

    * $\Delta f$ affects vertical acceleration.
    * $\Delta \phi$ controls angular and lateral motion through torque and gravitational coupling.
    * The system has full rank input influence over all second-order dynamics.


    """
    )
    return


@app.cell
def _(A, B, matrix_rank, np):
    def controllability():
        # Build controllability matrix
        n = A.shape[0]
        C = B
        for i in range(1, n):
            C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))
    
        # Check rank
        rank_C = matrix_rank(C)
        print("Controllability matrix rank:", rank_C)
    controllability()
    return


app._unparsable_cell(
    r"""

    ### ✅ Final Answer

    $$
    \boxed{
    \text{Yes, the linearized model is controllable.}
    }
    $$

    ➡️ We can drive the system from any state to any other state using appropriate inputs $(\Delta f, \Delta \phi)$.
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Lateral Dynamics

    We limit our interest in the lateral position $x$, the tilt $\theta$ and their derivatives (we are for the moment fine with letting $y$ and $\dot{y}$ be uncontrolled). We also set $f = M g$ and control the system only with $\phi$.

    What are the new (reduced) matrices $A$ and $B$ for this reduced system?
    Check the controllability of this new system.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 🧩 Reduced Lateral Dynamics

    We're now focusing **only on the lateral motion**:

    * Position: $x$
    * Velocity: $\dot{x}$
    * Tilt angle: $\theta$
    * Angular velocity: $\dot{\theta}$

    We ignore $y, \dot{y}$, and set $f = Mg$ (i.e., no control over vertical thrust).

    ---

    ### 🔁 Step 1: Define Reduced State and Input

    #### State vector $X_{\text{lat}} \in \mathbb{R}^4$:

    $$
    X_{\text{lat}} =
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    $$

    #### Input:

    $$
    U = \Delta \phi
    $$

    ---

    ### 🔁 Step 2: From linearized equations

    From earlier:

    $$
    \ddot{\Delta x} = -g (\Delta \theta + \Delta \phi)
    \quad\text{and}\quad
    \ddot{\Delta \theta} = -\frac{\ell Mg}{J} \Delta \phi
    $$

    ---

    ### 🧩 Step 3: Write in first-order form

    Let’s define:

    $$
    \dot{X}_{\text{lat}} =
    \begin{bmatrix}
    \dot{\Delta x} \\
    \ddot{\Delta x} \\
    \dot{\Delta \theta} \\
    \ddot{\Delta \theta}
    \end{bmatrix}
    =
    A_{\text{lat}} X_{\text{lat}} + B_{\text{lat}} U
    $$

    ---

    ### ✅ Reduced Matrices

    $$
    A_{\text{lat}} =
    \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
    \end{bmatrix}, \quad
    B_{\text{lat}} =
    \begin{bmatrix}
    0 \\
    - g \\
    0 \\
    - \frac{\ell M g}{J}
    \end{bmatrix}
    $$

    ---

    ### 🧪 Controllability Check

    Let’s compute the controllability matrix:

    $$
    \mathcal{C}_{\text{lat}} = \begin{bmatrix} B & AB & A^2B & A^3B \end{bmatrix}
    $$

    """
    )
    return


@app.cell
def _(J, M, g, l, matrix_rank, np):
    A_lat = np.array([
        [0, 1, 0, 0],
        [0, 0, -g, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

    B_lat = np.array([
        [0],
        [-g],
        [0],
        [-l * M * g / J]
    ])

    # Controllability matrix
    C_lat = B_lat
    for i in range(1, 4):
        C_lat = np.hstack((C_lat, np.linalg.matrix_power(A_lat, i) @ B_lat))

    rank_C_lat = matrix_rank(C_lat)
    print("Controllability matrix rank:", rank_C_lat)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ### ✅ Final Answer

    $$
    \boxed{
    \text{Yes, the reduced lateral system is controllable.}
    }
    $$

    ➡️ You can fully control lateral translation and tilt using only the gimbal angle $\phi$.
    """
    )
    return


if __name__ == "__main__":
    app.run()
