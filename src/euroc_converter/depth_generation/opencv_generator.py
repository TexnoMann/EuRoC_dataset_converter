import os, sys
import pathlib
from .generator import BaseDepthGenerator, DepthAlignType
from euroc_converter.datasets.base import CameraConfig

import numpy as np
import cv2
from .opencv_stero_depth.opencv_stereo_estimation import create_depth_with_rectification
from typing import Tuple

class OpenCV_DepthGenerator(BaseDepthGenerator):

    def __init__(
        self,
        left_camera_config: CameraConfig,
        right_camera_config: CameraConfig,
        align_type: DepthAlignType,
        method_config: dict = None
    ):
        super().__init__(left_camera_config, right_camera_config, align_type, method_config)

    def generate_depth(self, left_image: cv2.Mat, right_image: cv2.Mat)-> Tuple[np.ndarray, np.ndarray, Tuple]:
        depth, confidence_map, valid_disp_roi = create_depth_with_rectification(
            left_image, 
            right_image, 
            self.left_camera_config, 
            self.right_camera_config,
            self.method_config
        )
        return depth, confidence_map, valid_disp_roi