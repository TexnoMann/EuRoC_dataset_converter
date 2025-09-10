import glob
import os
from abc import ABC, abstractmethod
from euroc_converter.datasets.base import CameraConfig
from enum import Enum, auto
import cv2
import numpy as np
from typing import Tuple

class DepthAlignType(Enum):
    LEFT = 'left'
    RIGHT = 'right'

class BaseDepthGenerator:
    
    def __init__(self, 
        left_camera_config: CameraConfig,
        right_camera_config: CameraConfig,
        align_type: DepthAlignType,
        method_config: dict = None
    ):
        self.left_camera_config = left_camera_config
        self.right_camera_config = right_camera_config
        self.align_type = align_type
        self.method_config = method_config

    @abstractmethod
    def generate_depth(self, left_image: cv2.Mat, right_image: cv2.Mat) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        pass