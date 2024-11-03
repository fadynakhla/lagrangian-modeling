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



if __name__=="__main__":
    t = Trajectory.make_empty(100, 2)
    print(t.position)
