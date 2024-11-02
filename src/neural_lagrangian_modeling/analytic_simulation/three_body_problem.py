import pydantic
import numpy as np

class GeneralCoords(pydantic.BaseModel): ...


class TwoDimCoords(GeneralCoords):
    x: float
    y: float


class ThreeDimCoords(GeneralCoords):
    x: float
    y: float
    z: float


class MassiveBody(pydantic.BaseModel):
    mass: float
    position: GeneralCoords
    velocity: GeneralCoords

class ThreeBodyAnalyticSimulator:
    def __init__(self):
        pass
