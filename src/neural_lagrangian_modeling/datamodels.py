import pydantic
import numpy as np
from numpy import typing as npt


class Trajectory(pydantic.BaseModel):
    position: list[npt.NDArray[np.float64]]
    velocity: list[npt.NDArray[np.float64]]
    acceleration: list[npt.NDArray[np.float64]]


class MassiveBody(pydantic.BaseModel):
    mass: float
    position: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]
