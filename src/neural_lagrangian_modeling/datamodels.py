import pydantic
import numpy as np
from numpy import typing as npt


class Trajectory(pydantic.BaseModel):
    position: list[npt.NDArray[np.float128]]
    velocity: list[npt.NDArray[np.float128]]
    acceleration: list[npt.NDArray[np.float128]]


class MassiveBody(pydantic.BaseModel):
    mass: float
    position: npt.NDArray[np.float128]
    velocity: npt.NDArray[np.float128]
