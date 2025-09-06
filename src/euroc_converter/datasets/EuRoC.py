import glob
import json
import os
import re
import pathlib
import numpy as np
from .base import BaseDataset, CameraConfig, IMUConfig
import pandas as pd
import pypose as pp
import torch
import yaml

class EuRoCDataset(BaseDataset):
    def __init__(self, cfg, basedir):
        self.config = cfg
        self.basedir = pathlib.Path(basedir)/'mav0'
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
            self.cameras[cam_name] = CameraConfig(
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
        
        self.imu = IMUConfig(
            rate_hz = imu_cfg['rate_hz'],
            gyroscope_noise_density = imu_cfg['gyroscope_noise_density'],
            gyroscope_random_walk = imu_cfg['gyroscope_random_walk'],
            accelerometer_noise_density = imu_cfg['accelerometer_noise_density'],
            accelerometer_random_walk = imu_cfg['accelerometer_random_walk'],
            extrinsics = np.array(imu_cfg['T_BS']['data']).reshape((4, 4))
        )
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

        # Parse gt:
        self.tstamp_gt, self.gt_poses = self.__parse_gt(self.basedir)
        
        # Parse color images:
        self.color_data = self.__parse_color_data(self.basedir)

        # Generate depth:
        self.depth_data = self.__generate_depth_data(self.basedir, self.config['depth_data'])
    
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
            color_data = pd.read_csv(color_info_path)
            color_data[self.config['cameras'][cam_name]] = {}
            color_data[self.config['cameras'][cam_name]]["time"] = color_data['#timestamp [ns]'].to_numpy()*1e-9
            color_data[self.config['cameras'][cam_name]]["path"] = color_data['filename'].to_list()
        return color_data
    
    def __generate_depth_data(self, method: str):


    def parse(self):
        """ read data in euroc format """

        # Parse gt:
        self.tstamp_gt, self.gt_poses = self.__parse_gt(self.basedir)
        
        # Parse images:
        image_data = {}
        for cam_name in self.cameras.keys():
            print(f"Reading data for source: {cam_name}")
            color_info_path = self.basedir/cam_name/"data.csv" 
            color_data = pd.read_csv(color_info_path)
            image_data[self.config['cameras'][cam_name]] = {}
            image_data[self.config['cameras'][cam_name]]["time"] = color_data['#timestamp [ns]'].to_numpy()*1e-9
            image_data[self.config['cameras'][cam_name]]["path"] = color_data['filename'].to_list()
        print(image_data)

        # image_data = self.parse_list(image_list)
        # depth_data = self.parse_list(depth_list)
        # pose_data = self.parse_list(pose_list, skiprows=1)
        # pose_vecs = pose_data[:, 1:].astype(np.float64)

        # tstamp_image = image_data[:, 0].astype(np.float64)
        # tstamp_depth = depth_data[:, 0].astype(np.float64)
        # tstamp_pose = pose_data[:, 0].astype(np.float64)

        # associations = self.associate_frames(
        #     tstamp_image, tstamp_depth, tstamp_pose)

        # indicies = [0]
        # for i in range(1, len(associations)):
        #     t0 = tstamp_image[associations[indicies[-1]][0]]
        #     t1 = tstamp_image[associations[i][0]]
        #     if t1 - t0 > 1.0 / frame_rate:
        #         indicies += [i]

        # images, poses, depths, intrinsics = [], [], [], []
        # inv_pose = None
        # for ix in indicies:
        #     (i, j, k) = associations[ix]
        #     images += [os.path.join(datapath, image_data[i, 1])]
        #     depths += [os.path.join(datapath, depth_data[j, 1])]
        #     c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
        #     # if inv_pose is None:
        #     #     inv_pose = np.linalg.inv(c2w)
        #     #     c2w = np.eye(4)
        #     # else:
        #     #     c2w = inv_pose@c2w
        #     c2w[:3, 1] *= -1
        #     c2w[:3, 2] *= -1
        #     c2w = torch.from_numpy(c2w).float()
        #     poses += [c2w]

        # return images, depths, poses
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, index):
        color_left_path = self.color_left_paths[index]
        color_right_path = self.color_left_paths[index]
        # depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.config['cam']['fx'], 
                                      self.config['cam']['fy'],
                                      self.config['cam']['cx'], 
                                      self.config['cam']['cy']])
            color_data = cv2.undistort(color_data, K, self.distortion)
        
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)  

        if self.rays_d is None:
            self.rays_d =get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)


        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()
        
        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        


        ret = {
            "frame_id": self.frame_ids[index],
            "c2w":  self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d
        }
        return ret
