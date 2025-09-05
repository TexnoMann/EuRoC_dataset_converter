import glob
import os

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from pydantic import Field, TypeAdapter
from typing import List, Dict
from pydantic.dataclasses import dataclass
from numpydantic import NDArray, Shape

@dataclass
class CameraConfig:
    rate_hz: int
    H: int
    W: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    distortion_coefficients: NDArray[Shape["* x"], np.float64]
    extrinsics: NDArray[Shape["4 x, 4 y"], np.float64]

@dataclass
class IMUConfig:
    rate_hz: int
    gyroscope_noise_density: float     # [ rad / s / sqrt(Hz) ]   ( gyro "white noise" )
    gyroscope_random_walk: float       # [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
    accelerometer_noise_density: float  # [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" )
    accelerometer_random_walk: float # [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion )
    extrinsics: NDArray[Shape["4 x, 4 y"], np.float64]

class BaseDataset(Dataset):
 
    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError()