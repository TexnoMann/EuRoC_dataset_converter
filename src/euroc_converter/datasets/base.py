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
    
def from_camera_config_to_intrinsics_matrix(camera_config: CameraConfig) -> np.ndarray:
    return np.array([
        [camera_config.fx, 0.0, camera_config.cx],
        [0.0, camera_config.fy , camera_config.cy],
        [0.0, 0.0, 1.0]])

def from_dict_to_camera_config(cam_cfg: dict):
    return CameraConfig(
        rate_hz = cam_cfg['rate_hz'],
        H = cam_cfg['resolution'][1],
        W = cam_cfg['resolution'][0],
        fx = cam_cfg['intrinsics'][0],
        fy = cam_cfg['intrinsics'][1],
        cx = cam_cfg['intrinsics'][2],
        cy = cam_cfg['intrinsics'][3],
        distortion_model = cam_cfg['distortion_model'],
        distortion_coefficients = np.array(cam_cfg['distortion_coefficients']),
        extrinsics = np.array(cam_cfg['T_BS']['data']).reshape((4, 4))
    )

def from_camera_config_to_dict(cam_cfg: CameraConfig):
    return {
        'rate_hz': cam_cfg.rate_hz,
        'resolution': [cam_cfg.W, cam_cfg.H],
        'intrinsics': [cam_cfg.fx, cam_cfg.fy, cam_cfg.cx, cam_cfg.cy],
        'distortion_model': cam_cfg.distortion_model,
        'distortion_coefficients': cam_cfg.distortion_coefficients.tolist(),
        'T_BS': {'data': cam_cfg.extrinsics.ravel().tolist()}   
    }

def from_dict_to_imu_config(imu_cfg: dict):
    return IMUConfig(
        rate_hz = imu_cfg['rate_hz'],
        gyroscope_noise_density = imu_cfg['gyroscope_noise_density'],
        gyroscope_random_walk = imu_cfg['gyroscope_random_walk'],
        accelerometer_noise_density = imu_cfg['accelerometer_noise_density'],
        accelerometer_random_walk = imu_cfg['accelerometer_random_walk'],
        extrinsics = np.array(imu_cfg['T_BS']['data']).reshape((4, 4))
    )

def from_imu_config_to_dict(imu_cfg: IMUConfig):
    return {
        'rate_hz': imu_cfg.rate_hz,
        'gyroscope_noise_density': imu_cfg.gyroscope_noise_density,
        'gyroscope_random_walk': imu_cfg.gyroscope_random_walk,
        'accelerometer_noise_density': imu_cfg.accelerometer_noise_density,
        'accelerometer_random_walk': imu_cfg.accelerometer_random_walk,
        'T_BS': {'data': imu_cfg.extrinsics.ravel().tolist()}   
    }