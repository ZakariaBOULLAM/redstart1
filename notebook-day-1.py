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
    mo.md(r"""Project Redstart is an attempt to design the control systems of a reusable booster during landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In principle, it is similar to SpaceX's Falcon Heavy Booster.

    >The Falcon Heavy booster is the first stage of SpaceX's powerful Falcon Heavy rocket, which consists of three modified Falcon 9 boosters strapped together. These boosters provide the massive thrust needed to lift heavy payloads—like satellites or spacecraft—into orbit. After launch, the two side boosters separate and land back on Earth for reuse, while the center booster either lands on a droneship or is discarded in high-energy missions.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.Html("""
    <iframe width="560" height="315" src="https://www.youtube.com/embed/RYUr-5PYA7s?si=EXPnjNVnqmJSsIjc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>""")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dependencies""")
    return


@app.cell
def _():
    import scipy
    import scipy.integrate as sci
    import scipy.interpolate as sciB
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    from tqdm import tqdm

    # The use of autograd is optional in this project, but it may come in handy!
    import autograd
    import autograd.numpy as np
    import autograd.numpy.linalg as la
    from autograd import isinstance, tuple
    return FFMpegWriter, FuncAnimation, np, plt, sci, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The Model

    The Redstart booster in model as a rigid tube of length $2 \ell$ and negligible diameter whose mass $M$ is uniformly spread along its length. It may be located in 2D space by the coordinates $(x, y)$ of its center of mass and the angle $\theta$ it makes with respect to the vertical (with the convention that $\theta > 0$ for a left tilt, i.e. the angle is measured counterclockwise)

    This booster has an orientable reactor at its base ; the force that it generates is of amplitude $f>0$ and the angle of the force with respect to the booster axis is $\phi$ (with a counterclockwise convention).

    We assume that the booster is subject to gravity, the reactor force and that the friction of the air is negligible.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image(src="public/images/geometry.svg"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Constants

    For the sake of simplicity (this is merely a toy model!) in the sequel we assume that: 

      - the total length $2 \ell$ of the booster is 2 meters,
      - its mass $M$ is 1 kg,
      - the gravity constant $g$ is 1 m/s^2.

    This set of values is not realistic, but will simplify our computations and do not impact the structure of the booster dynamics.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Helpers

    ### Rotation matrix

    $$ 
    \begin{bmatrix}
    \cos \alpha & - \sin \alpha \\
    \sin \alpha &  \cos \alpha  \\
    \end{bmatrix}
    $$
    """
    )
    return


@app.cell
def _(np):
    def R(alpha):
        return np.array([
            [np.cos(alpha), -np.sin(alpha)], 
            [np.sin(alpha),  np.cos(alpha)]
        ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Videos

    It will be very handy to make small videos to visualize the evolution of our booster!
    Here is an example of how such videos can be made with Matplotlib and displayed in marimo.
    """
    )
    return


@app.cell
def _(FFMpegWriter, FuncAnimation, mo, np, plt, tqdm):
    def make_video(output):
        fig = plt.figure(figsize=(10, 6)) # width, height in inches (1 inch = 2.54 cm)
        num_frames = 100
        fps = 30 # Number of frames per second

        def animate(frame_index):    
            # Clear the canvas and redraw everything at each step
            plt.clf()
            plt.xlim(0, 2*np.pi)
            plt.ylim(-1.5, 1.5)
            plt.title(f"Sine Wave Animation - Frame {frame_index+1}/{num_frames}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)

            x = np.linspace(0, 2*np.pi, 100)
            phase = frame_index / 10
            y = np.sin(x + phase)
            plt.plot(x, y, "r-", lw=2, label=f"sin(x + {phase:.1f})")
            plt.legend()

            pbar.update(1)

        pbar = tqdm(total=num_frames, desc="Generating video")
        anim = FuncAnimation(fig, animate, frames=num_frames)
        writer = FFMpegWriter(fps=fps)
        anim.save(output, writer=writer)

        print()
        print(f"Animation saved as {output!r}")

    _filename = "wave_animation.mp4"
    make_video(_filename)
    (mo.video(src=_filename))
    return


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
    * The global coordinate system has **positive $y$**.

    Then the **global direction** of the thrust force is $\theta + \phi$.
    So the **force vector** in global coordinates is:

    $$
    \boxed{
    (f_x, f_y) = f \cdot \left( \sin(\theta + \phi),\; \cos(\theta + \phi) \right)
    }
    $$

    ---
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
    * The **air friction is neglected** (as per assumptions),

    So Newton’s second law gives:

    $$
    \begin{cases}
    M \cdot \ddot{x} = f_x \\
    M \cdot \ddot{y} = f_y - M g
    \end{cases}
    $$

    Substituting the expressions for $(f_x, f_y)$ from earlier:

    $$
    f_x = f \cdot \sin(\theta + \phi), \quad
    f_y = f \cdot \cos(\theta + \phi)
    $$

    Then:

    $$
    \boxed{
    \begin{aligned}
    \ddot{x} &= \frac{f}{M} \cdot \sin(\theta + \phi) \\
    \ddot{y} &= \frac{f}{M} \cdot \cos(\theta + \phi) - g
    \end{aligned}
    }
    $$

    ---

    ### 🎯 Final ODEs:

    With the constants $M = 1$, $g = 1$, we simplify:

    $$
    \boxed{
    \begin{aligned}
    \ddot{x} &= f \cdot \sin(\theta + \phi) \\
    \ddot{y} &= f \cdot \cos(\theta + \phi) - 1
    \end{aligned}
    }
    $$

    These are the **second-order ODEs** governing the horizontal and vertical motion of the center of mass.
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

    ### 📐 Apply the constants:

    * $M = 1$
    * $\ell = 1$ (so $2\ell = 2$)

    $$
    J = \frac{1}{12} \cdot 1 \cdot (2)^2 = \frac{1}{12} \cdot 4 = \frac{1}{3}
    $$

    ---

    ### ✅ Final Answer:

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
    We analyze the rotation of the booster using **Newton’s second law for rotation**:

    $$
    J \cdot \ddot{\theta} = \tau
    $$

    ---

    ### 🔧 Step 1: Torque from the Reactor

    * The reactor applies a force $\vec{F}$ at the **base** of the booster.
    * The vector from the **center of mass to the base** is:

    $$
    \vec{r} = -\ell \begin{bmatrix} \sin\theta \\ \cos\theta \end{bmatrix}
    $$

    * The force vector (amplitude $f$, angle $\phi$ from the booster axis) is:

    $$
    \vec{F} = f \begin{bmatrix} \sin(\theta + \phi) \\ \cos(\theta + \phi) \end{bmatrix}
    $$

    * The **torque** is the 2D cross product:

    $$
    \tau = \vec{r} \times \vec{F}
    = (-\ell \sin\theta)(f \cos(\theta + \phi)) - (-\ell \cos\theta)(f \sin(\theta + \phi))
    $$

    $$
    \tau = \ell f \left[ -\sin\theta \cos(\theta + \phi) + \cos\theta \sin(\theta + \phi) \right]
    = \ell f \cdot \sin(\phi)
    $$

    (using the identity $\cos A \sin B - \sin A \cos B = \sin(B - A)$)

    ---

    ### 🧮 Final ODE for $\theta(t)$

    Now substitute into Newton's rotational law:

    $$
    \ddot{\theta} = \frac{\tau}{J} = \frac{\ell f}{J} \cdot \sin(\phi)
    $$

    Using $\ell = 1$, $J = \frac{1}{3}$, we get:

    $$
    \boxed{
    \ddot{\theta} = 3f \cdot \sin(\phi)
    }
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 🧩 **Simulation**

    Define a function `redstart_solve` that, given as input parameters: 

      - `t_span`: a pair of initial time `t_0` and final time `t_f`,
      - `y0`: the value of the state `[x, dx, y, dy, theta, dtheta]` at `t_0`,
      - `f_phi`: a function that given the current time `t` and current state value `y`
         returns the values of the inputs `f` and `phi` in an array.

    returns:

      - `sol`: a function that given a time `t` returns the value of the state `[x, dx, y, dy, theta, dtheta]` at time `t` (and that also accepts 1d-arrays of times for multiple state evaluations).

    A typical usage would be:

    ```python
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0] # state: [x, dx, y, dy, theta, dtheta]
        def f_phi(t, y):
            return np.array([0.0, 0.0]) # input [f, phi]
        sol = redstart_solve(t_span, y0, f_phi)
        t = np.linspace(t_span[0], t_span[1], 1000)
        y_t = sol(t)[2]
        plt.plot(t, y_t, label=r"$y(t)$ (height in meters)")
        plt.plot(t, l * np.ones_like(t), color="grey", ls="--", label=r"$y=\ell$")
        plt.title("Free Fall")
        plt.xlabel("time $t$")
        plt.grid(True)
        plt.legend()
        return plt.gcf()
    free_fall_example()
    ```

    Test this typical example with your function `redstart_solve` and check that its graphical output makes sense.
    """
    )
    return


@app.cell(hide_code=True)
def _(J, M, g, l, np, plt, sci):
    # Dynamics solver
    def redstart_solve(t_span, y0, f_phi):
        def dynamics(t, y):
            x, dx, y_pos, dy, theta, dtheta = y
            f, phi = f_phi(t, y)

            # Translational accelerations
            ddx = f * np.sin(theta + phi) / M
            ddy = f * np.cos(theta + phi) / M - g

            # Rotational acceleration
            ddtheta = (l * f * np.sin(phi)) / J

            return [dx, ddx, dy, ddy, dtheta, ddtheta]

        sol_ivp = sci.solve_ivp(dynamics, t_span, y0, dense_output=True)

        def sol(t):
            return sol_ivp.sol(t)

        return sol

    # Example: Free fall test
    def free_fall_example():
        t_span = [0.0, 5.0]
        y0 = [0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # [x, dx, y, dy, theta, dtheta]

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
    **Note:**
    The plot shows the vertical position $y(t)$ of the booster’s center of mass during free fall, starting from 10 meters. It smoothly decreases and approaches the reference line $y = \ell = 1$ meter, which corresponds to the minimum safe altitude before the base of the booster touches the ground (since the booster has length $2\ell$).

    This confirms that our function `redstart_solve` correctly simulates the dynamics of the system under gravity.

    **Important:** In this model, the booster appears to “sink” into the ground because we simulate only the center of mass and **do not include ground collision detection**. This is expected in a basic model and can be refined later.

    """
    )
    return


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
def _(l, np, plt, redstart_solve):
    from scipy.interpolate import BPoly

    # Time interval
    t0, tf = 0.0, 5.0

    # Boundary conditions
    y0, dy0 = 10.0, 0.0
    yf, dyf = 1.0, 0.0

    # Fit a cubic polynomial y(t) that satisfies the boundary conditions
    # We'll use Hermite interpolation with positions and velocities
    times = [t0, tf]
    values = [[y0, dy0], [yf, dyf]]  # [ [y0, dy0], [yf, dyf] ]
    poly = BPoly.from_derivatives(times, values)

    # Derivatives
    def y_desired(t):
        return poly(t)

    def dy_desired(t):
        return poly.derivative()(t)

    def ddy_desired(t):
        return poly.derivative(2)(t)

    # Compute required thrust: ddot(y) = f(t) - 1 => f(t) = ddot(y) + 1
    def f_phi_controlled(t, y):
        f = ddy_desired(t) + 1
        return np.array([f, 0.0])  # keep phi = 0

    # Initial state
    y0_state = [0.0, 0.0, y0, dy0, 0.0, 0.0]

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

    return (f_phi_controlled,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The simulation shows that the booster’s **center of mass** descends smoothly to the target height $y = \ell$, with the base just reaching the ground ($y - \ell = 0$) at $t = 5$, and with no vertical velocity — achieving a soft landing.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 🧩 Drawing

    Create a function that draws the body of the booster, the flame of its reactor as well as its target landing zone on the ground (of coordinates $(0, 0)$).

    The drawing can be very simple (a rectangle for the body and another one of a different color for the flame will do perfectly!).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(mo.image("public/images/booster_drawing.png"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Make sure that the orientation of the flame is correct and that its length is proportional to the force $f$ with the length equal to $\ell$ when $f=Mg$.

    The function shall accept the parameters `x`, `y`, `theta`, `f` and `phi`.
    """
    )
    return


@app.cell
def _(M, g, l, np, plt):
    from matplotlib.patches import Polygon, Rectangle

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

    return Polygon, Rectangle, draw_booster


@app.cell
def _(draw_booster, np):
    draw_booster(x=0.0, y=6, theta=0*np.pi/12, f=0.9, phi=0.9)
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
):
    import os 
    def simulate_booster_landing(
        initial_state,
        t_span=[0.0, 5.0],
        f_phi=f_phi_controlled,
        fps=30,
        output_path="C:\\auto\\redstart-master\\booster_landing.mp4"
    ):
        # Solve the system
        sol = redstart_solve(t_span, initial_state, f_phi)

        # Time values for frames
        num_frames = int((t_span[1] - t_span[0]) * fps)
        times = np.linspace(t_span[0], t_span[1], num_frames)

        # Set up the figure for animation
        fig, ax = plt.subplots(figsize=(2, 6))
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f8ff')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 12)
        ax.axis('off')

        body_patch = Polygon(np.zeros((4, 2)), closed=True, color='black')
        flame_patch = Polygon(np.zeros((3, 2)), closed=True, color='orange')
        landing_pad = Rectangle((-0.5, -0.2), 1.0, 0.2, color='sandybrown')

        ax.add_patch(body_patch)
        ax.add_patch(flame_patch)
        ax.add_patch(landing_pad)

        def update(frame_idx):
            t = times[frame_idx]
            state = sol(t)
            x, dx, y, dy, theta, dtheta = state
            f, phi = f_phi(t, state)

            # Compute geometry
            axis = np.array([np.sin(theta), np.cos(theta)])
            perp = np.array([np.cos(theta), -np.sin(theta)])
            base = np.array([x, y]) - l * axis
            tip = np.array([x, y]) + l * axis

            body = np.array([
                base + (0.05) * perp,
                base - (0.05) * perp,
                tip - (0.05) * perp,
                tip + (0.05) * perp
            ])
            body_patch.set_xy(body)

            if f > 0:
                thrust_dir = np.array([np.sin(theta + phi), np.cos(theta + phi)])
                flame_length = (f / (M * g)) * l
                flame_tip = base - flame_length * thrust_dir
                flame = np.array([
                    base + (0.04) * perp,
                    base - (0.04) * perp,
                    flame_tip
                ])
                flame_patch.set_xy(flame)
                flame_patch.set_visible(True)
            else:
                flame_patch.set_visible(False)

            return body_patch, flame_patch

        ani = FuncAnimation(
            fig, update, frames=num_frames, blit=True
        )

        writer = FFMpegWriter(fps=fps)
        ani.save(output_path, writer=writer)
        plt.close(fig)

        return output_path

    # Generate smooth video
    video_path = simulate_booster_landing(
        initial_state=[0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
        fps=30
    )
    video_path

    return


if __name__ == "__main__":
    app.run()
