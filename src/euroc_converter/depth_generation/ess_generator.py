import os, sys
import docker
import pathlib
from .generator import BaseDepthGenerator, DepthAlignType
from euroc_converter.datasets.base import CameraConfig

import cv2
import subprocess

class ESS_ISAAC_DepthGenerator(BaseDepthGenerator):

    def __init__(
        self,
        left_camera_config: CameraConfig,
        right_camera_config: CameraConfig,
        align_type: DepthAlignType,
        isaac_ws_path: str
    ):
        super().__init__(left_camera_config, right_camera_config, align_type)
        self.__isaac_ws_path = pathlib.Path(isaac_ws_path)

        self.docker_client = docker.from_env()

    def __install_isaac_ros_ws(self):
        if not 'ISAAC_ROS_WS' in os.environ:
            if not self.__isaac_ws_path.exists():
                subprocess.run(f'mkdir -p {str(self.__isaac_ws_path/'isaac_ros-dev'/'src')}', shell=True, capture_output=True, text=True)
            subprocess.run(f'echo "export ISAAC_ROS_WS={str(self.__isaac_ws_path/'isaac_ros-dev')}" >> ~/.bashrc', shell=True, capture_output=True, text=True)
        subprocess.run(f'sudo apt-get install -y curl jq tar', shell=True, capture_output=True, text=True)
        subprocess.run('')

    def generate_depth(self, left_image: cv2.Mat, right_image: cv2.Mat) -> cv2.Mat:
        pass