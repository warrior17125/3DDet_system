import copy
import os
from typing import List, Optional, Tuple
from nuscenes import NuScenes
import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from mmdet3d.structures import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]


OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
    "tricycle": (220, 20, 60),  # 相比原版 mmdet3d 的 visualize 增加 tricycle
    "cyclist": (220, 20, 60)  # 相比原版 mmdet3d 的 visualize 增加 cyclist
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

# 定义类别映射（NuScenes类别名称 -> 自定义类别名称）
nuScenes_to_internal = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'pedestrian',
    'human.pedestrian.stroller': 'pedestrian',
    'vehicle.car': 'car',
    'vehicle.bicycle': 'bicycle',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.bus': 'bus',
    'vehicle.trailer': 'trailer',
    'movable_object.trafficcone': 'traffic_cone',
}

# 自定义类别列表（必须与OBJECT_PALETTE的键一致）
object_classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone'
]

point_cloud_range =  [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# 初始化NuScenes数据集
nusc = NuScenes(version='v1.0-mini', dataroot='E:\\DataSets\\nuScenes', verbose=True)

def quaternion_yaw(q: Quaternion) -> float:
    """
    从四元数中提取偏航角（yaw），范围 [-pi, pi]
    """
    # 使用四元数的旋转矩阵计算欧拉角
    rotation_matrix = q.rotation_matrix
    # 提取偏航角
    yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return yaw


def get_sample_annotations(nusc, sample_token):
    """获取并转换指定样本的3D标注"""
    sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    
    # 获取坐标系转换参数
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    calibrated_sensor = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    
    all_boxes = []
    all_labels = []
    
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        
        # 类别映射
        internal_category = nuScenes_to_internal.get(ann['category_name'], None)
        if internal_category not in object_classes:
            continue
        
        # 创建全局坐标系下的Box
        box = Box(
            ann['translation'],
            ann['size'],
            Quaternion(ann['rotation']),
            name=ann['category_name']
        )
        
        # 全局 -> Ego坐标系
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        # Ego -> LiDAR坐标系
        box.translate(-np.array(calibrated_sensor['translation']))
        box.rotate(Quaternion(calibrated_sensor['rotation']).inverse)
        
        # 提取参数（调整尺寸顺序为l, w, h）
        center = box.center
        size = [box.wlh[1], box.wlh[0], box.wlh[2]]  # wlh -> lwh
        yaw = quaternion_yaw(box.orientation)
        
        # 构建边界框参数
        box_params = np.array([center[0], center[1], center[2], size[0], size[1], size[2], yaw])
        all_boxes.append(box_params)
        all_labels.append(object_classes.index(internal_category))
    
    # 创建LiDARInstance3DBoxes对象
    if len(all_boxes) > 0:
        return LiDARInstance3DBoxes(np.array(all_boxes)), np.array(all_labels)
    else:
        return None, None


def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(np.int),
                    coords[index, end].astype(np.int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()


def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def get_lidar2image(camera_info):
    lidar2camera_r = np.linalg.inv(camera_info["rotation"])
    lidar2camera_t = (
        camera_info["translation"] @ lidar2camera_r.T
    )
    lidar2camera_rt = np.eye(4).astype(np.float32)
    lidar2camera_rt[:3, :3] = lidar2camera_r.T
    lidar2camera_rt[3, :3] = -lidar2camera_t
    camera_intrinsics = np.eye(4).astype(np.float32)
    camera_intrinsics[:3, :3] = camera_info["cam_intrinsic"]
    lidar2image = camera_intrinsics @ lidar2camera_rt.T

    return lidar2image

img_save_path = "img.png"
lidar_save_path = "lidar.png"
sample_token = 'ca9a282c9e77460f8360f564131a8af5'  # 替换为实际样本token
bboxes, labels = get_sample_annotations(nusc, sample_token)
sample = nusc.get('sample', sample_token)
lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
camera_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
camera_info = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
image = mmcv.imread(nusc.get_sample_data_path(camera_data['token']))
points = np.fromfile("E:\\DataSets\\nuScenes\\" + lidar_data['filename'], dtype=np.float32).reshape(-1, 5)[:, :4]
points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)

visualize_camera(
    img_save_path,
    image,
    bboxes=bboxes,
    labels=labels,
    transform=get_lidar2image(camera_info),
    classes=object_classes,
)
visualize_lidar(
    lidar_save_path,
    points,
    bboxes=bboxes,
    labels=labels,
    xlim=[point_cloud_range[d] for d in [0, 3]],
    ylim=[point_cloud_range[d] for d in [1, 4]],
    classes=object_classes,
)