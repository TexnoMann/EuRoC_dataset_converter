import glob
import json
import os
import re
import pathlib
import numpy as np
from .base import (
    BaseDataset, CameraConfig, IMUConfig, 
    from_dict_to_camera_config, from_dict_to_imu_config,
    from_camera_config_to_dict, from_imu_config_to_dict
)
import pandas as pd
import pypose as pp
import torch
import yaml
import cv2
from PIL import Image
import tqdm
from euroc_converter.depth_generation.generator import BaseDepthGenerator, DepthAlignType

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os, shutil

import copy


class EuRoCDataset(BaseDataset):
    def __init__(self, cfg, basedir):
        self.config = cfg
        self.basedir = pathlib.Path(basedir)/'mav0'
        self.basedir_extended = pathlib.Path(basedir)/'mav0_ext'
        if not self.basedir.exists():
            print("Given basedir path invalid, please provide path to parent ov 'mav0' directory")
            exit()
        self.cameras = {}
        for cam_config_path in self.basedir.rglob("cam*/sensor.yaml"):
            cam_name = re.search("cam.", str(cam_config_path))
            cam_name = str(cam_config_path)[cam_name.span()[0]:cam_name.span()[1]]
            try:
                with open(cam_config_path, 'r') as file:
                    cam_cfg = yaml.safe_load(file)
            except FileNotFoundError:
                print(f"Camera config not found by path: '{cam_config_path}'.")
            except yaml.YAMLError as e:
                print(f"Error parsing YAML: {e}")
                exit()
            self.cameras[cam_name] = from_dict_to_camera_config(cam_cfg)
            print(f"- Parsed {self.config['cameras'][cam_name]} ('{cam_name}') camera config:\n {str(self.cameras[cam_name])}")
        imu_config_path = self.basedir/"imu0/sensor.yaml"
        try:
            with open(self.basedir/"imu0/sensor.yaml", 'r') as file:
                imu_cfg = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"IMU config not found by path: '{imu_config_path}'.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            exit()  
        
        self.imu = from_dict_to_imu_config(imu_cfg)
        print(f"- Parsed imu config:\n {str(self.imu)}")
    
    def associate_frames(self, tstamp_images, tstamp_imu, tstamp_gt, max_dt=0.075):
        """ match images, imus, and poses """
        associations = []
        for i, t in enumerate(tstamp_images):
            if tstamp_gt is None:
                j = np.argmin(np.abs(tstamp_imu - t))
                if (np.abs(tstamp_imu[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_imu - t))
                k = np.argmin(np.abs(tstamp_gt - t))

                if (np.abs(tstamp_imu[j] - t) < max_dt) and \
                        (np.abs(tstamp_gt[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def parse(self):
        """ read data in euroc format """
        if not self.basedir_extended.exists():
            self.basedir_extended.mkdir(parents=True, exist_ok=True)
        else:
            rewrite = input("Do you want to rewrite exist extended euroc dataset? [y/n]")
            if rewrite:
                shutil.rmtree(str(self.basedir_extended))
                self.basedir_extended.mkdir(parents=True, exist_ok=True)
        # Parse gt:
        self.tstamp_gt, self.gt_poses = self.__parse_gt(self.basedir)
        self.num_frames = len(self.tstamp_gt)
        
        # Parse color images:
        self.color_data = self.__parse_color_data()

        # Generate depth:
        self.tstamp_depth, self.depth_data = self.__extend_dataset_by_depth(self.config['depth_data'])
    
    def __parse_gt(self, basedir):
        """ read ground truth into list data """
        gt_path = basedir/"state_groundtruth_estimate0"/"data.csv"
        print("Parsing ground truth data ...")
        gt_poses = []
        if os.path.isfile(gt_path):
            gt_data = pd.read_csv(gt_path)
            # print(gt_data)
            tstamp_gt = gt_data['#timestamp'].to_numpy()*1e-9
            for i in range(0, len(tstamp_gt)):
                translation = torch.tensor([
                    gt_data[' p_RS_R_x [m]'][i],
                    gt_data[' p_RS_R_y [m]'][i],
                    gt_data[' p_RS_R_z [m]'][i]], dtype=torch.float32)
                quaternioun = torch.tensor([
                    gt_data[' q_RS_x []'][i],
                    gt_data[' q_RS_y []'][i],
                    gt_data[' q_RS_z []'][i],
                    gt_data[' q_RS_w []'][i]], dtype=torch.float32) # Pypose use 'xyzw' notation
                gt_poses.append(pp.SE3(torch.concat([translation, quaternioun])).matrix().detach().cpu().numpy())
        return tstamp_gt, gt_poses
    
    def __parse_color_data(self):
        color_data = {}
        for cam_name in self.cameras.keys():
            print(f"Reading data for source: {cam_name}")
            color_info_path = self.basedir/cam_name/"data.csv" 
            readed_color_data = pd.read_csv(color_info_path)
            color_data[cam_name] = {}
            color_data[cam_name]["time"] = readed_color_data['#timestamp [ns]'].to_numpy()*1e-9
            color_data[cam_name]["path"] = readed_color_data['filename'].tolist()
        return color_data
    
    def __extend_dataset_by_depth(self, method: str):

        method = self.config['depth_data']['method']['name']
        print(f"Generating depth data using method: {method} ...")

        method_config = pathlib.Path(self.config['depth_data']['method']['config_path'])
        try:
            with open(method_config, 'r') as file:
                method_config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Depth generation method config not found by path: '{method_config}'.")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            exit()

        if method == 'stereo_opencv':
            from euroc_converter.depth_generation.opencv_generator import OpenCV_DepthGenerator
            cam_cfg_by_side = {}
            for cam_names in self.config['cameras'].keys():
                cam_cfg_by_side[self.config['cameras'][cam_names]] = cam_names
            left_camera_config = self.cameras[cam_cfg_by_side['left']]
            right_camera_config = self.cameras[cam_cfg_by_side['right']]
            depth_generator = OpenCV_DepthGenerator(
                left_camera_config=left_camera_config,
                right_camera_config=right_camera_config,
                align_type=DepthAlignType.LEFT,
                method_config=method_config[method]
            )
            
            left_image_paths = self.color_data[cam_cfg_by_side['left']]["path"]
            right_image_paths = self.color_data[cam_cfg_by_side['right']]["path"]
            num_images = min(len(left_image_paths), len(right_image_paths))
            crop_roi = None
            
            (self.basedir_extended/cam_cfg_by_side['left']/'data').mkdir(exist_ok=True, parents=True)
            # Save depth image:
            (self.basedir_extended/'depth'/'data').mkdir(exist_ok=True, parents=True)

            for i in tqdm.tqdm(range(num_images)):
                left_image_path = self.basedir/cam_cfg_by_side['left']/'data'/left_image_paths[i]
                right_image_path = self.basedir/cam_cfg_by_side['right']/'data'/right_image_paths[i]
                left_image = cv2.imread(str(left_image_path))
                right_image = cv2.imread(str(right_image_path))
                depth_map, confidence, roi = depth_generator.generate_depth(left_image, right_image)
                # plt.imshow(depth_map)
                # plt.show()
                if crop_roi is None:
                    crop_roi = roi
                depth_map = depth_map[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
                image_filename = None

                image_filename = self.color_data[cam_cfg_by_side['left']]["path"][i]
                left_image = left_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            
                cv2.imwrite(
                    self.basedir_extended/cam_cfg_by_side['left']/'data'/image_filename,
                    left_image
                )
            
                depth_map = (depth_map*self.config['depth_data']['png_depth_scale']).astype(np.uint32)
                depth_image = Image.fromarray(depth_map)
                depth_image.save(self.basedir_extended/'depth'/'data'/image_filename)
            
            new_camera_config = copy.deepcopy(left_camera_config)
            new_camera_config.cx -= crop_roi[0]
            new_camera_config.cy -= crop_roi[1]

            new_camera_config.H = depth_map.shape[0]
            new_camera_config.W = depth_map.shape[1]

            new_camera_config_dict = from_camera_config_to_dict(new_camera_config)
            # Write fake color sensor config
            with open(self.basedir_extended/cam_cfg_by_side['left']/'sensor.yaml', "w") as file:
                yaml.dump(
                    new_camera_config_dict, file, sort_keys=False, default_flow_style=False
                )
            
            new_camera_config_dict.update( {
                    'png_depth_scale': self.config['depth_data']['png_depth_scale'],
                    'align_type': 'left'
                }
            )
    
            # Write fake depth sensor config
            with open(self.basedir_extended/'depth'/'sensor.yaml', "w") as file:
                yaml.dump(
                    new_camera_config_dict, file, sort_keys=False, default_flow_style=False
                )

            shutil.copy(
                str(self.basedir/cam_cfg_by_side['left']/'data.csv'),
                str(self.basedir_extended/cam_cfg_by_side['left']/'data.csv')
            )
            shutil.copy(
                str(self.basedir/cam_cfg_by_side['left']/'data.csv'),
                str(self.basedir_extended/'depth'/'data.csv')
            )
            shutil.copytree(
                str(self.basedir/'imu0'),
                str(self.basedir_extended/'imu0')
            )
            shutil.copytree(
                str(self.basedir/'state_groundtruth_estimate0'),
                str(self.basedir_extended/'state_groundtruth_estimate0')
            )
            

            tstamp_depth = self.color_data[cam_cfg_by_side['left']]["time"]
            depth_data  = self.color_data[cam_cfg_by_side['left']]["path"]

            return tstamp_depth, depth_data
        else:
            raise NotImplementedError(f"Depth generation method '{method}' not supported.")

    def __len__(self):
        return self.num_frames