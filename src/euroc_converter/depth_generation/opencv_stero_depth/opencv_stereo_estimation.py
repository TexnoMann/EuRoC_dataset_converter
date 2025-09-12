import cv2

import numpy as np
import os
from euroc_converter.datasets.base import CameraConfig, from_camera_config_to_intrinsics_matrix
from typing import Tuple

def create_depth_with_rectification(
    left_image: cv2.Mat, 
    right_image: cv2.Mat, 
    left_camera_config: CameraConfig, 
    right_camera_config: CameraConfig,
    method_config: dict = None
) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    """
    Computes and filters a disparity map using rectified stereo images.

    Args:
        left_image (cv2.Mat): Left image from the stereo camera.
        right_image (cv2.Mat): Right image from the stereo camera.
        left_camera_config (CameraConfig): Configuration for the left camera.
        right_camera_config (CameraConfig): Configuration for the right camera.
    """

    image_size = left_image.shape[:2]
    

    # 3. Perform stereo rectification
    # `alpha=-1` will crop the rectified image to only show valid pixels,
    # avoiding black borders. `alpha=0` shows all pixels.
    # `newImageSize` is set to the original image size.

    cam0_intrinsics = from_camera_config_to_intrinsics_matrix(left_camera_config)
    cam1_intrinsics = from_camera_config_to_intrinsics_matrix(right_camera_config)

    R_cam0 = left_camera_config.extrinsics[:3, :3]
    T_cam0 = left_camera_config.extrinsics[:3, 3]

    R_cam1 = right_camera_config.extrinsics[:3, :3]
    T_cam1 = right_camera_config.extrinsics[:3, 3]

    cam0_distortion = left_camera_config.distortion_coefficients
    cam1_distortion = right_camera_config.distortion_coefficients
    
    relative_R_cam1_cam0 = np.linalg.inv(R_cam1) @ R_cam0
    relative_T_cam1_cam0 = np.linalg.inv(R_cam1) @ (T_cam0 - T_cam1) 



    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1 = cam0_intrinsics, 
        distCoeffs1 = cam0_distortion, 
        cameraMatrix2 = cam1_intrinsics, 
        distCoeffs2 = cam1_distortion, 
        imageSize=(image_size[1], image_size[0]), 
        R =relative_R_cam1_cam0, 
        T = relative_T_cam1_cam0,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=-1
    )
    valid_disp_roi = cv2.getValidDisparityROI(roi1, roi2, method_config.get('min_disparity', 0), method_config.get('num_disparities', 128), method_config.get('block_size', 5)) 
    
    # 4. Compute undistortion and rectification maps
    map1x, map1y = cv2.initUndistortRectifyMap(cam0_intrinsics, cam0_distortion, R1, P1, (image_size[1], image_size[0]), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(cam1_intrinsics, cam1_distortion, R2, P2, (image_size[1], image_size[0]), cv2.CV_32FC1)

    # 5. Apply remapping to get the rectified images
    img_left_rectified = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
    img_right_rectified = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)
    
    # Convert to grayscale for stereo matching
    img_left_gray = cv2.cvtColor(img_left_rectified, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right_rectified, cv2.COLOR_BGR2GRAY)
    

    # 6. Configure stereo matchers for left and right views
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=method_config.get('min_disparity', 0),
        numDisparities=method_config.get('num_disparities', 128),  # Must be divisible by 16
        blockSize=method_config.get('block_size', 5),
        P1=8 * 3 * method_config.get('block_size', 5)**2,
        P2=32 * 3 * method_config.get('block_size', 5)**2,
        disp12MaxDiff=method_config.get('disp12_max_diff', -1),
        uniquenessRatio=method_config.get('uniqueness_ratio', 10),
        speckleWindowSize=method_config.get('speckle_window_size', 100),
        speckleRange=method_config.get('speckle_range', 32),
        preFilterCap=method_config.get('pre_filter_cap', 31),
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY if method_config.get('mode', 0) == 0 else cv2.STEREO_SGBM_MODE_HH
    )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # 7. Compute the raw left and right disparity maps from rectified images
    left_disp = left_matcher.compute(img_left_gray, img_right_gray)
    right_disp = right_matcher.compute(img_right_gray, img_left_gray)
    filtered_disp = left_disp
    
    filtration_method = method_config.get('disparity_filter', 'median')

    confidence_map = None
    if filtration_method == 'median':
        filtered_disp = cv2.medianBlur(left_disp, method_config.get('disparity_media_filter_window', 5))
        
    elif filtration_method == 'wls':
        # 8. Create and configure the DisparityWLSFilter
        disparityWLSFilter_lambda = method_config.get('disparity_wls_filter_lambda', 80000.0)
        disparityWLSFilter_sigma = method_config.get('disparity_wls_filter_sigma', 1.8)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(disparityWLSFilter_lambda)
        wls_filter.setSigmaColor(disparityWLSFilter_sigma)

        filtered_disp = wls_filter.filter(left_disp, img_left_gray, disparity_map_right=right_disp)
        confidence_map = wls_filter.getConfidenceMap()

    filtered_disp = filtered_disp.astype(np.float32) / 16.0
    depth = np.array(cv2.reprojectImageTo3D(filtered_disp, Q)[:,:, 2])
    if method_config.get('filter_by_confidence', False):
        if confidence_map is None:
            disparityWLSFilter_lambda = method_config.get('disparity_wls_filter_lambda', 80000.0)
            disparityWLSFilter_sigma = method_config.get('disparity_wls_filter_sigma', 1.8)
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls_filter.setLambda(disparityWLSFilter_lambda)
            wls_filter.setSigmaColor(disparityWLSFilter_sigma)

            _ = wls_filter.filter(left_disp, img_left_gray, disparity_map_right=right_disp)
            confidence_map = wls_filter.getConfidenceMap()
        
        depth[confidence_map < 255*method_config.get('min_confidence', 0.95)]
        
    depth[depth < method_config.get('min_depth', 0.0)] = 0.0
    depth[depth > method_config.get('max_depth', 20.0)] = 0.0

    return depth, confidence_map, valid_disp_roi