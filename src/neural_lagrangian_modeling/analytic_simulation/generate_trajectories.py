
from matplotlib import pyplot as plt

from neural_lagrangian_modeling.analytic_simulation import initial_conditions
from neural_lagrangian_modeling.analytic_simulation import random_initial_conditions
from neural_lagrangian_modeling.analytic_simulation import three_body_problem
from neural_lagrangian_modeling.visualization import three_body_visualizer

def generate_trajectory():
    # Initialize simulator and visualizer
    simulator = three_body_problem.ThreeBodyAnalyticSimulator(dt=0.01)  # Smaller dt for accuracy
    visualizer = three_body_visualizer.ThreeBodyVisualizer()

    # Choose initial conditions
    bodies = initial_conditions.lagrange_triangle()

    # Run simulation
    trajectories = simulator.simulate(
        *bodies,
        steps=2000,  # More steps for longer simulation
        save_path=f"data/lagrange"
    )

    # Animate results
    visualizer.animate_trajectories(
        trajectories,
        interval=20,      # Faster animation
        trail_length=100,  # Longer trails
        save_path=f"animations/lagrange.gif"
    )

    # Plot conservation
    visualizer.plot_conservation_laws(trajectories, simulator)


if __name__=="__main__":
    generate_trajectory()
