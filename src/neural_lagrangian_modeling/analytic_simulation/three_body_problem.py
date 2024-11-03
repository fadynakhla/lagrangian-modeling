import numpy as np
from numpy import typing as npt

from neural_lagrangian_modeling import datamodels


Trajectories = tuple[datamodels.Trajectory, datamodels.Trajectory, datamodels.Trajectory]


class ThreeBodyAnalyticSimulator:
    def __init__(self, dt: float = 0.0001):
        self.dt = dt

    def simulate(
        self,
        m1: datamodels.MassiveBody,
        m2: datamodels.MassiveBody,
        m3: datamodels.MassiveBody,
        steps: int,
        dims: int,
        visualize: bool,
    ) -> Trajectories:
        trajectories = tuple(
            datamodels.Trajectory.from_massive_body(o, steps) for o in (m1, m2, m3)
        )

        for step in range(steps):
            accelerations = self.accelerations(*trajectories, step)

            for a, t in zip(accelerations, trajectories):
                t.acceleration[step] = a
                t.velocity[step + 1] = t.velocity[step] + a * self.dt
                t.position[step + 1] = t.position[step] + t.velocity[step + 1] * self.dt

        return trajectories

    def accelerations(
        self,
        t1: datamodels.Trajectory,
        t2: datamodels.Trajectory,
        t3: datamodels.Trajectory,
        step: int
    ) -> tuple[
        npt.NDArray[np.float128], npt.NDArray[np.float128], npt.NDArray[np.float128]
    ]:
        r21 = t2.position[step] - t1.position[step]
        r31 = t3.position[step] - t1.position[step]
        r32 = t3.position[step] - t2.position[step]

        d21 = np.linalg.norm(r21)
        d31 = np.linalg.norm(r31)
        d32 = np.linalg.norm(r32)

        d21_cubed = d21**3
        d31_cubed = d31**3
        d32_cubed = d32**3

        a1 = t2.mass * (r21/d21_cubed) + t3.mass * (r31/d31_cubed)
        a2 = t1.mass * (-r21/d21_cubed) + t3.mass * (r32/d32_cubed)
        a3 = t1.mass * (-r31/d31_cubed) + t2.mass * (-r32/d32_cubed)

        return a1, a2, a3


    def energy(self, trajectories: Trajectories, step: int) -> float:
        """Calculate total energy of the system at given step."""
        # Kinetic energy
        ke = sum(0.5 * t.mass * np.sum(t.velocity[step]**2) for t in trajectories)

        # Potential energy
        t1, t2, t3 = trajectories
        r21 = np.linalg.norm(t2.position[step] - t1.position[step])
        r31 = np.linalg.norm(t3.position[step] - t1.position[step])
        r32 = np.linalg.norm(t3.position[step] - t2.position[step])

        pe = -(t1.mass * t2.mass / r21 +
              t1.mass * t3.mass / r31 +
              t2.mass * t3.mass / r32)

        return ke + pe

    def momentum(self, trajectories: Trajectories, step: int) -> npt.NDArray[np.float128]:
        """Calculate total linear momentum of system."""
        return sum(t.mass * t.velocity[step] for t in trajectories)

    def angular_momentum(self, trajectories: Trajectories, step: int) -> npt.NDArray[np.float128]:
        """Calculate total angular momentum of system."""
        return sum(t.mass * np.cross(t.position[step], t.velocity[step])
                  for t in trajectories)

    def validate_simulation(self, trajectories: Trajectories) -> dict[str, float]:
        """Validate conservation laws over simulation."""
        initial_energy = self.energy(trajectories, 0)
        initial_momentum = self.momentum(trajectories, 0)
        initial_angular_momentum = self.angular_momentum(trajectories, 0)

        steps = len(trajectories[0].position) - 1
        energy_error = []
        momentum_error = []
        angular_momentum_error = []

        for step in range(steps):
            energy_error.append(abs(self.energy(trajectories, step) - initial_energy))
            momentum_error.append(np.linalg.norm(
                self.momentum(trajectories, step) - initial_momentum))
            angular_momentum_error.append(np.linalg.norm(
                self.angular_momentum(trajectories, step) - initial_angular_momentum))

        return {
            'max_energy_error': max(energy_error),
            'max_momentum_error': max(momentum_error),
            'max_angular_momentum_error': max(angular_momentum_error)
        }
