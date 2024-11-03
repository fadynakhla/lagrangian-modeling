import json
import pathlib

import pydantic
import numpy as np
from numpy import typing as npt


class MassiveBody(pydantic.BaseModel):
    mass: float
    position: npt.NDArray[np.float128]
    velocity: npt.NDArray[np.float128]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


class Trajectory(pydantic.BaseModel):
    mass: float
    position: npt.NDArray[np.float128]
    velocity: npt.NDArray[np.float128]
    acceleration: npt.NDArray[np.float128]

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def make_empty(cls, obj_mass: float, steps: int, dims: int) -> "Trajectory":
        return cls(
            mass=obj_mass,
            position=np.zeros((steps + 1, dims), dtype=np.float128),
            velocity=np.zeros((steps + 1, dims), dtype=np.float128),
            acceleration=np.zeros((steps + 1, dims), dtype=np.float128)
        )

    @classmethod
    def from_massive_body(cls, initial_state: MassiveBody, steps: int) -> "Trajectory":
        dims = len(initial_state.position)
        trajectory = cls.make_empty(initial_state.mass, steps, dims)
        trajectory.position[0] = initial_state.position
        trajectory.velocity[0] = initial_state.velocity
        return trajectory



def save_trajectories(
    trajectories: tuple[Trajectory, ...],
    filepath: str,
) -> None:
    """Save trajectories to a compressed npz file.

    Saves:
    - positions, velocities, accelerations as float128 arrays
    - masses as float64 array
    - metadata (timesteps, dimensions) as json
    """
    path = pathlib.Path(filepath)
    if not path.suffix:
        path = path.with_suffix('.npz')

    # Extract arrays and metadata
    data_dict = {}
    for i, traj in enumerate(trajectories):
        prefix = f'body_{i}_'
        data_dict[prefix + 'position'] = traj.position
        data_dict[prefix + 'velocity'] = traj.velocity
        data_dict[prefix + 'acceleration'] = traj.acceleration
        data_dict[prefix + 'mass'] = traj.mass

    # Add metadata
    metadata = {
        'n_bodies': len(trajectories),
        'n_steps': len(trajectories[0].position),
        'dims': trajectories[0].position.shape[1],
    }
    data_dict['metadata'] = json.dumps(metadata)

    # Save everything in a single compressed file
    np.savez_compressed(path, **data_dict)
    print(f"Saved trajectory data to {path}")

def load_trajectories(filepath: str) -> tuple[Trajectory, ...]:
    """Load trajectories from npz file."""
    data = np.load(filepath)
    metadata = json.loads(str(data['metadata']))

    trajectories = []
    for i in range(metadata['n_bodies']):
        prefix = f'body_{i}_'
        traj = Trajectory(
            mass=float(data[prefix + 'mass']),
            position=data[prefix + 'position'],
            velocity=data[prefix + 'velocity'],
            acceleration=data[prefix + 'acceleration']
        )
        trajectories.append(traj)

    return tuple(trajectories)


if __name__=="__main__":
    t = Trajectory.make_empty(100, 2)
    print(t.position)
