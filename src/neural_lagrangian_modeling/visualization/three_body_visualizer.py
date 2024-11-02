import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional
from neural_lagrangian_modeling import datamodels

Trajectories = tuple[datamodels.Trajectory, ...]

class ThreeBodyVisualizer:
    def __init__(self):
        plt.style.use('dark_background')  # Makes trajectories stand out better

    def plot_conservation_laws(
        self,
        trajectories: Trajectories,
        simulator,  # ThreeBodyAnalyticSimulator instance
        show: bool = True,
        save_path: Optional[str] = None
    ):
        """Plot energy and momentum conservation over time."""
        steps = len(trajectories[0].position) - 1
        time = np.arange(steps) * simulator.dt
        dims = trajectories[0].position.shape[1]  # 2 or 3

        # Calculate conservation quantities
        energy = np.zeros(steps)
        momentum = np.zeros((steps, dims))
        ang_momentum = np.zeros(steps) if dims == 2 else np.zeros((steps, dims))

        for step in range(steps):
            energy[step] = simulator.energy(trajectories, step)
            momentum[step] = simulator.momentum(trajectories, step)
            ang_momentum[step] = simulator.angular_momentum(trajectories, step)

        # Normalize to initial values
        energy = (energy - energy[0]) / np.abs(energy[0])
        momentum = momentum - momentum[0]
        ang_momentum = ang_momentum - ang_momentum[0]

        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # Energy
        axes[0].plot(time, energy, 'r-', label='Energy')
        axes[0].set_title('Relative Energy Error')
        axes[0].set_ylabel('ΔE/E₀')
        axes[0].grid(True)

        # Linear momentum
        comp_labels = ['x', 'y', 'z'][:dims]
        colors = ['r', 'g', 'b'][:dims]
        for i, (label, color) in enumerate(zip(comp_labels, colors)):
            axes[1].plot(time, momentum[:, i], color=color, label=label)
        axes[1].set_title('Linear Momentum Error')
        axes[1].set_ylabel('ΔP')
        axes[1].legend()
        axes[1].grid(True)

        # Angular momentum
        if dims == 2:
            axes[2].plot(time, ang_momentum, 'r-', label='z')
        else:
            for i, (label, color) in enumerate(zip(comp_labels, colors)):
                axes[2].plot(time, ang_momentum[:, i], color=color, label=label)
        axes[2].set_title('Angular Momentum Error')
        axes[2].set_ylabel('ΔL')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()
        plt.close()

    def animate_trajectories(
        self,
        trajectories: Trajectories,
        interval: int = 50,  # ms between frames
        trail_length: int = 100,  # number of points in trail
        save_path: Optional[str] = None,
    ):
        """Create an animation of the n-body motion in 2D or 3D."""
        dims = trajectories[0].position.shape[1]
        if dims not in (2, 3):
            raise ValueError("Can only animate 2D or 3D trajectories")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d' if dims == 3 else None)

        colors = ['r', 'g', 'b'][:len(trajectories)]
        trails = []  # Store line objects
        points = []  # Store point objects

        # Initialize trails and points
        for traj, color in zip(trajectories, colors):
            if dims == 2:
                trail, = ax.plot([], [], color=color, alpha=0.5)
                point, = ax.plot([], [], color=color, marker='o', markersize=10)
            else:
                trail, = ax.plot([], [], [], color=color, alpha=0.5)
                point, = ax.plot([], [], [], color=color, marker='o', markersize=10)
            trails.append(trail)
            points.append(point)

        # Set axis limits
        all_positions = np.concatenate([t.position for t in trajectories])
        max_range = np.max(all_positions) - np.min(all_positions)
        mid_point = np.mean(all_positions, axis=0)

        margin = max_range * 0.1  # 10% margin
        if dims == 2:
            ax.set_xlim(mid_point[0] - max_range/2 - margin,
                       mid_point[0] + max_range/2 + margin)
            ax.set_ylim(mid_point[1] - max_range/2 - margin,
                       mid_point[1] + max_range/2 + margin)
            ax.set_aspect('equal')
        else:
            ax.set_xlim(mid_point[0] - max_range/2 - margin,
                       mid_point[0] + max_range/2 + margin)
            ax.set_ylim(mid_point[1] - max_range/2 - margin,
                       mid_point[1] + max_range/2 + margin)
            ax.set_zlim(mid_point[2] - max_range/2 - margin,
                       mid_point[2] + max_range/2 + margin)

        def update(frame):
            # Calculate trail start index
            trail_start = max(0, frame - trail_length)

            for trail, point, traj in zip(trails, points, trajectories):
                if dims == 2:
                    # Update trail
                    trail.set_data(
                        traj.position[trail_start:frame, 0],
                        traj.position[trail_start:frame, 1]
                    )
                    # Update point
                    point.set_data(
                        [traj.position[frame, 0]],
                        [traj.position[frame, 1]]
                    )
                else:
                    # Update trail
                    trail.set_data(
                        traj.position[trail_start:frame, 0],
                        traj.position[trail_start:frame, 1]
                    )
                    trail.set_3d_properties(traj.position[trail_start:frame, 2])
                    # Update point
                    point.set_data(
                        [traj.position[frame, 0]],
                        [traj.position[frame, 1]]
                    )
                    point.set_3d_properties([traj.position[frame, 2]])

            return trails + points

        steps = len(trajectories[0].position)
        anim = FuncAnimation(
            fig, update, frames=steps,
            interval=interval, blit=True
        )

        if save_path:
            anim.save(save_path, writer='pillow')
        else:
            plt.show()
        plt.close()
