import os
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points
import matplotlib.pyplot as plt
import open3d as o3d


def get_image_boxes(nusc, sample, camera_channel):
    """获取指定摄像头图像上的2D边界框"""
    cam_token = sample['data'][camera_channel]
    cam_data = nusc.get('sample_data', cam_token)
    
    # 获取传感器标定和位姿信息
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
    K = np.array(cs_record['camera_intrinsic'])
    
    # 处理所有标注
    boxes_2d = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = nusc.get_box(ann['token'])
        
        # 将框转换到相机坐标系
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        
        # 投影3D框到图像平面
        corners = view_points(box.corners(), K, normalize=True)[:2, :]
        
        # 检查框是否在图像范围内
        if check_corners_in_image(corners, (cam_data['width'], cam_data['height'])):
            x_min = max(0, np.min(corners[0]))
            y_min = max(0, np.min(corners[1]))
            x_max = min(cam_data['width'], np.max(corners[0]))
            y_max = min(cam_data['height'], np.max(corners[1]))
            boxes_2d.append((x_min, y_min, x_max, y_max))
    
    return boxes_2d, cam_data['filename']


def get_lidar_boxes(nusc, sample, lidar_channel='LIDAR_TOP'):
    """获取点云坐标系中的3D边界框"""
    lidar_token = sample['data'][lidar_channel]
    lidar_data = nusc.get('sample_data', lidar_token)
    
    # 获取标定和位姿信息
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    
    boxes_3d = []
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        box = nusc.get_box(ann['token'])
        
        # 转换到激光雷达坐标系
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)
        
        boxes_3d.append(box)
    
    return boxes_3d, lidar_data['filename']


def check_corners_in_image(corners, image_size):
    """检查角点是否在图像范围内"""
    in_x = np.logical_and(corners[0] >= 0, corners[0] < image_size[0])
    in_y = np.logical_and(corners[1] >= 0, corners[1] < image_size[1])
    return np.any(np.logical_and(in_x, in_y))


def visualize_combined(img_paths, boxes_list, camera_channels, lidar_path, boxes_3d):
    """
    将六个相机视角与点云可视化合并到一张图片上。
    """
    # 加载点云
    pc = LidarPointCloud.from_file(lidar_path)
    points = pc.points.T[:, :3]  # 提取点云的 x, y, z 坐标
    
    # 创建可视化窗口
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))  # 2 行 4 列布局
    axs = axs.ravel()  # 将 2D 数组展平为 1D
    
    # 可视化点云
    axs[0].scatter(points[:, 0], points[:, 1], s=0.1, c='gray', alpha=0.8)
    axs[0].set_title("LiDAR Point Cloud", fontsize=14)
    axs[0].set_xlabel("X (meters)")
    axs[0].set_ylabel("Y (meters)")
    axs[0].set_aspect('equal')
    axs[0].set_facecolor('black')  # 设置背景为黑色

    # 添加 3D 边界框到点云图
    for box in boxes_3d:
        corners = box.corners()[:2, :]  # 提取 2D 边界框角点（仅显示 XY 平面）
        for i in range(4):
            axs[0].plot([corners[0, i], corners[0, (i + 1) % 4]],
                        [corners[1, i], corners[1, (i + 1) % 4]], 'r-', linewidth=1)
    
    # 可视化六个相机视角
    for i, (img_path, boxes_2d, channel) in enumerate(zip(img_paths, boxes_list, camera_channels)):
        img = Image.open(img_path)
        axs[i + 1].imshow(img)
        for box in boxes_2d:
            x_min, y_min, x_max, y_max = box
            axs[i + 1].add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                               edgecolor='red', facecolor='none', linewidth=2))
        axs[i + 1].axis('off')
        axs[i + 1].set_title(channel, fontsize=14)

    # 调整布局
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 初始化NuScenes数据集
    nusc = NuScenes(version='v1.0-mini', dataroot='E:\\DataSets\\nuScenes', verbose=True)
    
    # 获取第一个样本
    sample = nusc.sample[0]
    
    # 定义六个摄像头通道
    camera_channels = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    
    # 处理所有摄像头图像
    img_paths = []
    boxes_list = []
    for channel in camera_channels:
        boxes_2d, img_path = get_image_boxes(nusc, sample, channel)
        img_paths.append(os.path.join(nusc.dataroot, img_path))
        boxes_list.append(boxes_2d)
    
    # 处理点云
    boxes_3d, lidar_path = get_lidar_boxes(nusc, sample)
    
    # 可视化相机图像和点云
    visualize_combined(img_paths, boxes_list, camera_channels, os.path.join(nusc.dataroot, lidar_path), boxes_3d)