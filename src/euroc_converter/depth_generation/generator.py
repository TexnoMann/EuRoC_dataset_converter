import glob
import os
from abc import ABC, abstractmethod
from euroc_converter.datasets.base import CameraConfig
from enum import Enum, auto
import cv2

class DepthAlignType(Enum):
    LEFT = auto()
    RIGHT = auto()

class BaseDepthGenerator:
    
    def __init__(self, 
        left_camera_config: CameraConfig,
        right_camera_config: CameraConfig,
        align_type: DepthAlignType
    ):
        self.left_camera_config = left_camera_config
        self.right_camera_config = right_camera_config
        self.align_type = align_type

    @abstractmethod
    def generate_depth(self, left_image: cv2.Mat, right_image: cv2.Mat) -> cv2.Mat:
        pass