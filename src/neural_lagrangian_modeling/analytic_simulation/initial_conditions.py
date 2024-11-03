"""Some nice initial conditions for the three body problem."""


import numpy as np
from neural_lagrangian_modeling import datamodels

def figure_eight() -> tuple[datamodels.MassiveBody, datamodels.MassiveBody, datamodels.MassiveBody]:
    """Generate initial conditions for figure-8 solution.
    From http://homepages.math.uic.edu/~jan/mcs320s07/Project_Two_Body.html"""

    # Initial positions
    r1 = np.array([0.97000436, -0.24308753, 0], dtype=np.float128)
    r2 = np.array([-r1[0], -r1[1], 0], dtype=np.float128)  # Mirror of r1
    r3 = np.array([0., 0., 0.], dtype=np.float128)  # Center

    # Initial velocities
    v3 = np.array([-0.93240737, -0.86473146, 0], dtype=np.float128)
    v1 = np.array([0.46620369, 0.43236573, 0], dtype=np.float128)  # -v3/2
    v2 = v1  # Same as v1

    # Equal masses
    m = 1.0

    return (
        datamodels.MassiveBody(mass=m, position=r1, velocity=v1),
        datamodels.MassiveBody(mass=m, position=r2, velocity=v2),
        datamodels.MassiveBody(mass=m, position=r3, velocity=v3)
    )

def lagrange_triangle() -> tuple[datamodels.MassiveBody, datamodels.MassiveBody, datamodels.MassiveBody]:
    """Generate initial conditions for equilateral triangle solution.
    Bodies rotate around their center of mass maintaining triangular formation."""

    # Place bodies in equilateral triangle
    r1 = np.array([1., 0., 0.], dtype=np.float128)
    r2 = np.array([-0.5, 0.866025404, 0.], dtype=np.float128)  # cos(120°), sin(120°)
    r3 = np.array([-0.5, -0.866025404, 0.], dtype=np.float128)  # cos(-120°), sin(-120°)

    # Angular velocity for stable orbit
    omega = 1.0  # Can be adjusted

    # Velocities for circular motion
    v1 = omega * np.array([0., 1., 0.], dtype=np.float128)
    v2 = omega * np.array([-0.866025404, -0.5, 0.], dtype=np.float128)
    v3 = omega * np.array([0.866025404, -0.5, 0.], dtype=np.float128)

    # Equal masses
    m = 1.0

    return (
        datamodels.MassiveBody(mass=m, position=r1, velocity=v1),
        datamodels.MassiveBody(mass=m, position=r2, velocity=v2),
        datamodels.MassiveBody(mass=m, position=r3, velocity=v3)
    )

def sun_earth_moon() -> tuple[datamodels.MassiveBody, datamodels.MassiveBody, datamodels.MassiveBody]:
    """Generate initial conditions for a hierarchical three-body system.
    Roughly similar to Sun-Earth-Moon scale ratios, but not exact."""

    # Mass ratios (not to scale, but maintaining hierarchy)
    m_sun = 1000.0
    m_earth = 1.0
    m_moon = 0.01

    # Positions
    r_sun = np.array([0., 0., 0.], dtype=np.float128)
    r_earth = np.array([1., 0., 0.], dtype=np.float128)
    r_moon = np.array([1.1, 0., 0.], dtype=np.float128)

    # Velocities (approximately circular orbits)
    v_sun = np.array([0., 0., 0.], dtype=np.float128)
    v_earth = np.array([0., np.sqrt(m_sun), 0.], dtype=np.float128)
    v_moon = np.array([0., np.sqrt(m_sun/1.1) + 0.1, 0.], dtype=np.float128)

    return (
        datamodels.MassiveBody(mass=m_sun, position=r_sun, velocity=v_sun),
        datamodels.MassiveBody(mass=m_earth, position=r_earth, velocity=v_earth),
        datamodels.MassiveBody(mass=m_moon, position=r_moon, velocity=v_moon)
    )

if __name__=="__main__":
    from matplotlib import pyplot as plt
    from neural_lagrangian_modeling.analytic_simulation import three_body_problem
    from neural_lagrangian_modeling.visualization import three_body_visualizer

    # Initialize simulator and visualizer
    simulator = three_body_problem.ThreeBodyAnalyticSimulator(dt=0.01)  # Smaller dt for accuracy
    visualizer = three_body_visualizer.ThreeBodyVisualizer()

    # Choose initial conditions
    bodies = figure_eight()  # or lagrange_triangle() or sun_earth_moon()

    # Run simulation
    trajectories = simulator.simulate(
        *bodies,
        steps=1000,  # More steps for longer simulation
        save_path="data/figure_eight_sample"
    )

    # Animate results
    visualizer.animate_trajectories(
        trajectories,
        interval=5,      # Faster animation
        trail_length=100,  # Longer trails
        save_path="animation.gif"
    )

    # Plot conservation
    visualizer.plot_conservation_laws(trajectories, simulator)
