
from matplotlib import pyplot as plt

from neural_lagrangian_modeling.analytic_simulation import initial_conditions
from neural_lagrangian_modeling.analytic_simulation import random_initial_conditions
from neural_lagrangian_modeling.analytic_simulation import three_body_problem
from neural_lagrangian_modeling.visualization import three_body_visualizer

def generate_trajectory(seed: int = 42):
    # Initialize simulator and visualizer
    simulator = three_body_problem.ThreeBodyAnalyticSimulator(dt=0.01)  # Smaller dt for accuracy
    visualizer = three_body_visualizer.ThreeBodyVisualizer()

    # Choose initial conditions
    bodies = random_initial_conditions.random_initial_conditions(seed=seed)  # or lagrange_triangle() or sun_earth_moon()

    # Run simulation
    trajectories = simulator.simulate(
        *bodies,
        steps=1000,  # More steps for longer simulation
        save_path=f"data/random_seed_{seed}"
    )

    # Animate results
    visualizer.animate_trajectories(
        trajectories,
        interval=20,      # Faster animation
        trail_length=100,  # Longer trails
        save_path=f"animations/random_{seed}.gif"
    )

    # Plot conservation
    visualizer.plot_conservation_laws(trajectories, simulator)


if __name__=="__main__":
    generate_trajectory(seed=30)
