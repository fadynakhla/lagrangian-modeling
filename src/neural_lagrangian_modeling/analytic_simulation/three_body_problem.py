import numpy as np
from numpy import typing as npt

from neural_lagrangian_modeling import datamodels


class ThreeBodyAnalyticSimulator:
    def __init__(self):
        pass

    def simulate(
        self,
        m1: datamodels.MassiveBody,
        m2: datamodels.MassiveBody,
        m3: datamodels.MassiveBody,
        dt: float,
        visualize: bool,
    ) -> datamodels.Trajectory: ...

    def accelerations(
        self,
        m1: datamodels.MassiveBody,
        m2: datamodels.MassiveBody,
        m3: datamodels.MassiveBody,
    ) -> tuple[
        npt.NDArray[np.float128], npt.NDArray[np.float128], npt.NDArray[np.float128]
    ]:
        r21 = m2.position - m1.position
        r31 = m3.position - m1.position
        r32 = m3.position - m2.position

        d21 = np.linalg.norm(r21)
        d31 = np.linalg.norm(r31)
        d32 = np.linalg.norm(r32)

        d21_cubed = d21**3
        d31_cubed = d31**3
        d32_cubed = d32**3

        a1 = m2.mass * (r21/d21_cubed) + m3.mass * (r31/d31_cubed)
        a2 = m1.mass * (-r21/d21_cubed) + m3.mass * (r32/d32_cubed)
        a3 = m1.mass * (-r31/d31_cubed) + m2.mass * (-r32/d32_cubed)

        return a1, a2, a3
