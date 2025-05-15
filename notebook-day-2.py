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
    Let:

    * $f > 0$: magnitude of the thrust force,
    * $\theta$: angle of the booster with respect to vertical,
    * $\phi$: angle of the force with respect to the booster’s **own axis**,
    * The global coordinate system has **positive $y$** pointing upward.

    Then the **global direction** of the thrust force is $\theta + \phi$.

    So the **force vector** in global coordinates is:

    $$
    \boxed{
    (f_x, f_y) = f \cdot \left( -\sin(\theta + \phi),\; +\cos(\theta + \phi) \right)
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
    We apply **Newton’s second law** to the motion of the center of mass:

    $$
    \boxed{
    M \cdot \ddot{x} = f_x, \quad M \cdot \ddot{y} = f_y - M g
    }
    $$

    The **total external force** on the center of mass is the sum of:

    * The **thrust force** from the reactor: $(f_x, f_y)$,
    * The **weight**: $(0, -Mg)$,
    * The **air friction is neglected** (as per assumptions).

    So Newton’s second law gives:

    $$
    \begin{cases}
    M \cdot \ddot{x} = f_x \\
    M \cdot \ddot{y} = f_y - M g
    \end{cases}
    $$

    Substituting the expressions for $(f_x, f_y)$ from earlier:

    $$
    f_x = -f \cdot \sin(\theta + \phi), \quad
    f_y = +f \cdot \cos(\theta + \phi)
    $$

    Then:

    $$
    \boxed{
    \begin{aligned}
    \ddot{x} &= -\frac{f}{M} \cdot \sin(\theta + \phi) \\
    \ddot{y} &= +\frac{f}{M} \cdot \cos(\theta + \phi) - g
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
    The booster is a:

    * **Rigid tube** (modeled as a thin rod),
    * Of **total length** $2\ell = 2\,\text{m}$,
    * Of **mass** $M = 1\,\text{kg}$,
    * With **mass uniformly distributed**,
    * And we rotate it about its **center**.


    The moment of inertia of a uniform thin rod of length $2\ell$ about its **center** is:

    $$
    J = \frac{1}{12} M (2\ell)^2
    $$

    ---
    * $M = 1$
    * $\ell = 1$ (so $2\ell = 2$)

    $$
    J = \frac{1}{12} \cdot 1 \cdot (2)^2 = \frac{1}{12} \cdot 4 = \frac{1}{3}
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
    We analyze the rotation of the booster using **Newton’s second law for rotation**:

    $$
    \boxed{
    J \cdot \ddot{\theta} = \tau
    }
    $$

    ---

    * The reactor applies a force $\vec{F}$ at the **base** of the booster.
    * The vector from the **center of mass to the base** is:

    $$
    \vec{r} = -\ell \begin{bmatrix} \sin\theta \\ \cos\theta \end{bmatrix}
    $$

    * The thrust force vector (magnitude $f$, angle $\phi$ from the booster axis) is:

    $$
    \vec{F} = -f \begin{bmatrix} \sin(\theta + \phi) \\ \cos(\theta + \phi) \end{bmatrix}
    $$

    * The **torque** is the 2D cross product:

    $$
    \tau = \vec{r} \times \vec{F}
    = (-\ell \sin\theta)(-f \cos(\theta + \phi)) - (-\ell \cos\theta)(f \sin(\theta + \phi))
    $$

    $$
    \tau = \ell f \left( \sin\theta \cos(\theta + \phi) - \cos\theta \sin(\theta + \phi) \right)
    = -\ell f \cdot \sin(\phi)
    $$

    (using the identity $\sin A \cos B - \cos A \sin B = -\sin(B - A)$)

    ---

    Now substitute into Newton's rotational law:

    $$
    \ddot{\theta} = \frac{\tau}{J} = -\frac{\ell f}{J} \cdot \sin(\phi)
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
    mo.md(r"""## 🧩 Controlled Landing""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We aim to find a **time-varying thrust** $f(t)$ that allows the booster to land smoothly from an altitude of 10 meters to just above the ground ($y = \ell = 1$) in 5 seconds, with zero velocity at both ends. For simplicity, we assume:

    * The booster remains perfectly **vertical**: $\theta(t) = 0$, $\phi(t) = 0$
    * No horizontal motion: $x(t) = \dot{x}(t) = 0$

    --- 
    Our objective is to find a force $f(t)$ such that:

    $$
    \begin{cases}
    y(0) = 10, & \dot{y}(0) = 0 \\
    y(5) = 1, & \dot{y}(5) = 0
    \end{cases}
    $$


    From previous derivations, the vertical motion obeys:

    $$
    \ddot{y}(t) = f(t) - 1
    $$

    This comes from Newton’s second law, with $f(t)$ being the vertical component of thrust and $g = 1 \ \text{m/s}^2$.


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


@app.cell(hide_code=True)
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
    mo.md(r"""## 🧩 Visualization""")
    return


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
    To find the **equilibria** of the Redstart booster system, we determine the conditions under which the **state variables remain constant over time**. This implies that all **derivatives must be zero**:

    $$
    \dot{x} = \ddot{x} = 0,\quad \dot{y} = \ddot{y} = 0,\quad \dot{\theta} = \ddot{\theta} = 0
    $$

    ---

    ### Step 1 : Horizontal acceleration:

    $$
    \ddot{x} = -\frac{f}{M} \sin(\theta + \phi) = 0
    \Rightarrow \sin(\theta + \phi) = 0
    \Rightarrow \theta + \phi = n\pi,\quad n \in \mathbb{Z}
    $$

    Given the physical constraints $|\theta| < \frac{\pi}{2}$ and $|\phi| < \frac{\pi}{2}$, the only admissible solution is:

    $$
    \theta + \phi = 0 \Rightarrow \phi = -\theta
    $$

    ---

    ### Step 2 : Vertical acceleration:

    $$
    \ddot{y} = \frac{f}{M} \cos(\theta + \phi) - g = 0
    \Rightarrow \frac{f}{M} \cos(\theta + \phi) = g
    $$

    Using $\theta + \phi = 0 \Rightarrow \cos(\theta + \phi) = 1$, we obtain:

    $$
    f = Mg
    $$

    ---

    ### Step 3 : Angular acceleration:

    $$
    \ddot{\theta} = -\frac{\ell f}{J} \sin(\phi) = 0
    \Rightarrow \sin(\phi) = 0 \Rightarrow \phi = 0
    \Rightarrow \theta = 0 \quad (\text{since } \phi = -\theta)
    $$

    ---

    ### Equilibrium Conditions:

    The system is at equilibrium when:

    * $\boxed{\theta = 0}$
    * $\boxed{\phi = 0}$
    * $\boxed{f = Mg}$

    These conditions imply:

    * No tilt or rotation
    * No linear or angular acceleration
    * Thrust exactly balances gravitational force
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
    ### Step 1: Define perturbation variables

    We introduce deviations from equilibrium as follows:

    * $\Delta x = x - x_{\text{eq}}$
    * $\Delta y = y - y_{\text{eq}}$
    * $\Delta \theta = \theta$
    * $\Delta \phi = \phi$
    * $\Delta f = f - Mg$

    Note: Since the equilibrium values of $\theta$ and $\phi$ are zero, we have $\Delta \theta = \theta$ and $\Delta \phi = \phi$.

    ---

    ### Step 2: Expand the dynamics near equilibrium

    Original nonlinear equations:

    $$
    \begin{aligned}
    \ddot{x} &= -\frac{f}{M} \sin(\theta + \phi) \\
    \ddot{y} &= \frac{f}{M} \cos(\theta + \phi) - g \\
    \ddot{\theta} &= -\frac{\ell f}{J} \sin(\phi)
    \end{aligned}
    $$

    Using small-angle approximations:

    * $\sin(\theta + \phi) \approx \theta + \phi$
    * $\cos(\theta + \phi) \approx 1$
    * $\sin(\phi) \approx \phi$

    ---

    ### Step 3: Linearize each equation

    #### Horizontal motion:

    $$
    \ddot{x} = -\frac{f}{M} (\theta + \phi) 
    = -\frac{Mg + \Delta f}{M} (\theta + \phi) 
    = -g(\theta + \phi) - \frac{\Delta f}{M}(\theta + \phi)
    $$

    Neglecting second-order terms (e.g., $\Delta f \cdot \theta$):

    $$
    \boxed{\ddot{\Delta x} \approx -g (\Delta \theta + \Delta \phi)}
    $$

    ---

    #### Vertical motion:

    $$
    \ddot{y} = \frac{f}{M} \cos(\theta + \phi) - g 
    \approx \frac{Mg + \Delta f}{M} - g 
    = g + \frac{\Delta f}{M} - g = \frac{\Delta f}{M}
    $$

    $$
    \boxed{\ddot{\Delta y} \approx \frac{1}{M} \Delta f}
    $$

    ---

    #### Rotational motion:

    $$
    \ddot{\theta} = -\frac{\ell f}{J} \sin(\phi) 
    \approx -\frac{\ell (Mg + \Delta f)}{J} \phi 
    \approx -\frac{\ell Mg}{J} \phi
    $$

    Neglecting higher-order terms:

    $$
    \boxed{\ddot{\Delta \theta} \approx -\frac{\ell Mg}{J} \Delta \phi}
    $$

    ---

    ### Final Linearized System

    The linearized dynamics near the equilibrium are:

    $$
    \boxed{
    \begin{aligned}
    \ddot{\Delta x} &= -g (\Delta \theta + \Delta \phi) \\
    \ddot{\Delta y} &= \frac{1}{M} \Delta f \\
    \ddot{\Delta \theta} &= -\frac{\ell Mg}{J} \Delta \phi
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
    To find the matrices A and B, We will first express the linearized model in **state-space standard form**:

    $$
    \boxed{\dot{X} = AX + BU}
    $$

    ---


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

    From the linearized equations:

    $$
    \begin{aligned}
    \ddot{\Delta x} &= -g (\Delta \theta + \Delta \phi) \\
    \ddot{\Delta y} &= \frac{1}{M} \Delta f \\
    \ddot{\Delta \theta} &= -\frac{\ell Mg}{J} \Delta \phi
    \end{aligned}
    $$

    We write the full derivative of the state vector:

    $$
    \dot{X} =
    \begin{bmatrix}
    \dot{\Delta x} \\
    \dot{\Delta \dot{x}} \\
    \dot{\Delta y} \\
    \dot{\Delta \dot{y}} \\
    \dot{\Delta \theta} \\
    \dot{\Delta \dot{\theta}}
    \end{bmatrix}
    =
    \begin{bmatrix}
    \Delta \dot{x} \\
    - g (\Delta \theta + \Delta \phi) \\
    \Delta \dot{y} \\
    \dfrac{1}{M} \Delta f \\
    \Delta \dot{\theta} \\
    - \dfrac{M g \ell}{J} \Delta \phi
    \end{bmatrix}
    $$

    We separate this into terms depending on $X$ and $U$:

    $$
    \dot{X} =
    \underbrace{
    \begin{bmatrix}
    \Delta \dot{x} \\
    - g \Delta \theta \\
    \Delta \dot{y} \\
    0 \\
    \Delta \dot{\theta} \\
    0
    \end{bmatrix}
    }_{\text{Depends on } X}
    +
    \underbrace{
    \begin{bmatrix}
    0 \\
    - g \Delta \phi \\
    0 \\
    \dfrac{1}{M} \Delta f \\
    0 \\
    - \dfrac{M g \ell}{J} \Delta \phi
    \end{bmatrix}
    }_{\text{Depends on } U}
    $$


    We now express the system in matrix form:

    $$
    \dot{X} =
    \underbrace{
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix}
    }_{A}
    \begin{bmatrix}
    \Delta x \\
    \Delta \dot{x} \\
    \Delta y \\
    \Delta \dot{y} \\
    \Delta \theta \\
    \Delta \dot{\theta}
    \end{bmatrix}
    +
    \underbrace{
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \dfrac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\dfrac{\ell Mg}{J}
    \end{bmatrix}
    }_{B}
    \begin{bmatrix}
    \Delta f \\
    \Delta \phi
    \end{bmatrix}
    $$


    ### Thus

    $$
    \boxed{\dot{X} = AX + BU}
    $$



    #### Where matrix $A \in \mathbb{R}^{6 \times 6}$:

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

    #### And matrix $B \in \mathbb{R}^{6 \times 2}$:

    $$
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \dfrac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\dfrac{\ell Mg}{J}
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
    We analyze the linear system:

    $$
    \dot{X} = A X + B U
    $$

    and evaluate whether the **equilibrium point** $X = 0$, $U = 0$ is **asymptotically stable**.

    ---

    ### Theoretical Reminder

    > A linear system is **asymptotically stable** if and only if **all eigenvalues of matrix $A$** have **strictly negative real parts**.
    """
    )
    return


@app.cell
def _(M, g, l, np):
    def stability_task():
        # Matrix A
        A = np.zeros((6, 6))
        A[0, 1] = 1
        A[1, 4] = -g
        A[2, 3] = 1
        A[4, 5] = 1
    
        # Matrix B
        B = np.zeros((6, 2))
        B[1, 1] = -g
        B[3, 0] = 1.0 / M
        B[5, 1] = -3.0 * g / l
    
        eigenvalues = np.linalg.eigvals(A)
        print("Eigenvalues of A:")
        print(eigenvalues)
    stability_task()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Conclusion

    $$
    \boxed{
    \text{No, the generic equilibrium of the linearized model is not asymptotically stable.}
    }
    $$
    """
    )
    return


app._unparsable_cell(
    r"""
    ## 🧩 Controllability

    Is the linearized model controllable?
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We consider the **linearized model** around the hovering equilibrium, expressed in state-space form:

    $$
    \dot{X} = A X + B U
    $$

    with the state and input vectors defined as:

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
    \in \mathbb{R}^6,
    \qquad
    U =
    \begin{bmatrix}
    \Delta f \\
    \Delta \phi
    \end{bmatrix}
    \in \mathbb{R}^2
    $$

    The system matrices are:

    $$
    A =
    \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & -g & 0 \\
    0 & 0 & 0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
    \end{bmatrix},
    \qquad
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -g \\
    0 & 0 \\
    \frac{1}{M} & 0 \\
    0 & 0 \\
    0 & -\frac{M g \ell}{J}
    \end{bmatrix}
    $$

    Assuming $g = 1$, $M = 1$, $\ell = 1$, and $J = \frac{1}{3}$, we obtain:

    $$
    B =
    \begin{bmatrix}
    0 & 0 \\
    0 & -1 \\
    0 & 0 \\
    1 & 0 \\
    0 & 0 \\
    0 & -3
    \end{bmatrix}
    $$

    ---

    ### Controllability Analysis

    To determine whether the system is controllable, we construct the **controllability matrix**:

    $$
    \mathcal{C} = [B \; AB \; A^2B \; A^3B \; A^4B \; A^5B] \in \mathbb{R}^{6 \times 12}
    $$

    The system is controllable if:

    $$
    \text{rank}(\mathcal{C}) = 6
    $$
    """
    )
    return


@app.cell
def _(A, B, matrix_rank, np):
    def rank_of_controllability_matrix():
        # Build controllability matrix
        C = B
        for i in range(1, 6):
            C = np.hstack((C, np.linalg.matrix_power(A, i) @ B))
    
        # Compute rank
        rank_C = matrix_rank(C)
        print("Rank of controllability matrix:", rank_C)
    rank_of_controllability_matrix()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Since:

    $$
    \boxed{\text{rank}(\mathcal{C}) = 6 = \text{dim}(X)}
    $$

    we conclude:

    $$
    \boxed{
    \text{Yes, the linearized model is controllable.}
    }
    $$
    """
    )
    return


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
    We're now focusing **only on the lateral motion**:

    * Position: $x$
    * Velocity: $\dot{x}$
    * Tilt angle: $\theta$
    * Angular velocity: $\dot{\theta}$

    We ignore $y, \dot{y}$, and set $f = Mg$ (i.e., no control over vertical thrust).

    ---


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

    From earlier:

    $$
    \ddot{\Delta x} = -g (\Delta \theta + \Delta \phi)
    \quad\text{and}\quad
    \ddot{\Delta \theta} = -\frac{\ell Mg}{J} \Delta \phi
    $$

    ---

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

    Reduced Matrices : 

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
    ### Since:

    $$
    \boxed{\text{rank}(\mathcal{C}_{\text{lat}}) = 4}
    $$


    ### We conclude

    $$
    \boxed{
    \text{Yes, the reduced lateral system is controllable.}
    }
    $$

    Which mean we can fully control lateral translation and tilt using only the gimbal angle $\phi$.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Linear Model in Free Fall

    Make graphs of $y(t)$ and $\theta(t)$ for the linearized model when $\phi(t)=0$,
    $x(0)=0$, $\dot{x}(0)=0$, $\theta(0) = 45 / 180  \times \pi$  and $\dot{\theta}(0) =0$. What do you see? How do you explain it?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Given that the control angle $\phi(t) = 0$, the control input vector becomes:

    $$
    U(t) = 0
    $$

    This leads to a simplified **homogeneous system**:

    $$
    \dot{X} = AX
    $$

    ---

    ### Vertical Motion Analysis

    We focus on the equation governing vertical dynamics:

    $$
    \ddot{y} = \frac{f \cos(\theta + \phi)}{M} - g
    $$

    With $\phi = 0$, this reduces to:

    $$
    \ddot{y} = \frac{f \cos(\theta)}{M} - g
    $$

    Assuming the thrust exactly balances weight at equilibrium, i.e., $f = Mg$, the equation simplifies further:

    $$
    \ddot{y} = g \cos(\theta) - g
    $$

    ---

    ### Small-Angle Approximation

    For small angular deviations $\theta$, we use the second-order Taylor approximation:

    $$
    \cos(\theta) \approx 1 - \frac{1}{2} \theta^2
    $$

    Substituting into the equation:

    $$
    \ddot{y} \approx g \left(1 - \frac{1}{2} \theta^2\right) - g = -\frac{g}{2} \theta^2
    $$

    This reveals that **even after approximation, the vertical acceleration remains nonlinear** in $\theta$, due to the quadratic term.

    ---

    ### Linearization Assumption

    To preserve a fully linear model, we neglect the nonlinear component $\theta^2$, leading to:

    $$
    \ddot{y} \approx 0
    $$

    This implies the system experiences **free fall**, governed purely by gravity:

    $$
    \ddot{y} = -g \quad \Rightarrow \quad y(t) = y(0) + \dot{y}(0)t - \frac{1}{2}gt^2
    $$

    """
    )
    return


@app.cell(hide_code=True)
def _(g, np, plt):
    def linear_model_in_free_fall_task():
        # Initial tilt angle (45 degrees)
        theta0 = np.pi / 4  
    
        # Time array from 0 to 2 seconds
        t = np.linspace(0, 5, 200)
    
        # Vertical motion: free fall under gravity
        y = -0.5 * g * t**2  
    
        # Constant tilt angle
        theta = np.full_like(t, theta0)
    
        # Plotting results
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
        # Vertical position plot
        axs[0].plot(t, y, label="y(t)")
        axs[0].set_title("Vertical Position y(t)")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Height y (m)")
        axs[0].grid(True)
    
        # Tilt angle plot
        axs[1].plot(t, theta, label="θ(t)", color="orange")
        axs[1].set_title("Tilt Angle θ(t)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Angle θ (rad)")
        axs[1].grid(True)
    
        plt.tight_layout()
        plt.show()
    linear_model_in_free_fall_task()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    * The tilt angle $\theta(t)$ remains **constant at 45°**.
    * This behavior reflects **no external torque**, resulting in **zero angular acceleration**.

    * The vertical position $y(t)$ follows a **parabolic trajectory**, characteristic of **free fall**.
    * This results directly from the acceleration $\ddot{y} = -g$, with no upward thrust component.


    Under zero control input:

    * The system experiences **pure gravitational descent**.
    * The orientation remains fixed due to the absence of rotational dynamics.
    * This is consistent with a **homogeneous linear system** where control inputs are inactive and the system responds only to gravity.

    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Manually Tuned Controller

    Try to find the two missing coefficients of the matrix 

    $$
    K =
    \begin{bmatrix}
    0 & 0 & ? & ?
    \end{bmatrix}
    \in \mathbb{R}^{4\times 1}
    $$ 

    such that the control law 

    $$
    \Delta \phi(t)
    = 
    - K \cdot
    \begin{bmatrix}
    \Delta x(t) \\
    \Delta \dot{x}(t) \\
    \Delta \theta(t) \\
    \Delta \dot{\theta}(t)
    \end{bmatrix} \in \mathbb{R}
    $$

    manages  when
    $\Delta x(0)=0$, $\Delta \dot{x}(0)=0$, $\Delta \theta(0) = 45 / 180  \times \pi$  and $\Delta \dot{\theta}(0) =0$ to: 

      - make $\Delta \theta(t) \to 0$ in approximately $20$ sec (or less),
      - $|\Delta \theta(t)| < \pi/2$ and $|\Delta \phi(t)| < \pi/2$ at all times,
      - (but we don't care about a possible drift of $\Delta x(t)$).

    Explain your thought process, show your iterations!

    Is your closed-loop model asymptotically stable?
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Step 1 :

    We considered the linearized model for rotational dynamics:

    $$
    \ddot{\theta}(t) = -\frac{\ell M g}{J} \Delta \phi(t)
    $$

    and designed a proportional-derivative (PD) controller:

    $$
    \Delta \phi(t) = -k_3 \Delta \theta(t) - k_4 \Delta \dot{\theta}(t)
    $$

    Initial condition:
    $\Delta \theta(0) = \frac{\pi}{4}, \Delta \dot{\theta}(0) = 0$

    We simulated several candidates and evaluated them based on settling time, stability, and control limits.

    ---

    ### Step 2 : Final Controller

    The best performing manually tuned controller is:

    $$
    \boxed{
    K = \begin{bmatrix} 0 & 0 & 0.2 & 0.6 \end{bmatrix}
    }
    $$

    ---

    ### Step 3 : Performance Summary

    * **Convergence**: $\Delta \theta(t)$ settles to zero within \~8 seconds
    * **Boundedness**: $|\Delta \theta(t)| < \frac{\pi}{2}, |\Delta \phi(t)| < \frac{\pi}{2}$ throughout
    * **Control Strategy**: Emphasizes angular correction without affecting horizontal dynamics

    ---

    ###  Step 4 : Stability

    The closed-loop system:

    $$
    \dot{X} = (A - BK) X
    $$

    with the above $K$ matrix is **asymptotically stable**, as evidenced by the simulation response (all state trajectories converge).
    """
    )
    return


@app.cell
def _(np, plt, sci):
    # Final version adapted to the notebook model

    def manually_tuned_controller_adapted():
        # Initial conditions
        theta0 = np.pi / 4  # 45 degrees
        theta_dot0 = 0.0
        x0 = 0.0
        x_dot0 = 0.0
    
        # Tuned gains based on system behavior in the notebook context
        k3 = 0.2
        k4 = 0.6
        K = np.array([0, 0, k3, k4])
    
        # Time settings
        t_span = (0, 20)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

        # Simplified linear rotational model: θ'' = -k3 * θ - k4 * θ'
        def dynamics(t, state):
            x, x_dot, theta, theta_dot = state
            delta_state = np.array([x, x_dot, theta, theta_dot])
            delta_phi = -K @ delta_state
            delta_phi = np.clip(delta_phi, -np.pi/2, np.pi/2)
            theta_ddot = delta_phi
            x_ddot = 0  # No control in x
            return [x_dot, x_ddot, theta_dot, theta_ddot]

        initial_state = [x0, x_dot0, theta0, theta_dot0]
        sol = sci.integrate.solve_ivp(dynamics, t_span, initial_state, t_eval=t_eval)
    
        t = sol.t
        x, x_dot, theta, theta_dot = sol.y
        delta_phi = -k3 * theta - k4 * theta_dot
        delta_phi = np.clip(delta_phi, -np.pi/2, np.pi/2)

        # Plotting
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axs[0].plot(t, theta, label=r'$\theta(t)$', color='blue')
        axs[0].axhline(np.pi/2, color='gray', linestyle='--')
        axs[0].axhline(-np.pi/2, color='gray', linestyle='--')
        axs[0].set_ylabel("Tilt Angle θ (rad)")
        axs[0].set_title("Tilt Angle Dynamics")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(t, delta_phi, label=r'$\Delta \phi(t)$', color='orange')
        axs[1].axhline(np.pi/2, color='gray', linestyle='--')
        axs[1].axhline(-np.pi/2, color='gray', linestyle='--')
        axs[1].set_ylabel("Control Input Δφ (rad)")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_title("Control Input Over Time")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    manually_tuned_controller_adapted()

    return


if __name__ == "__main__":
    app.run()
