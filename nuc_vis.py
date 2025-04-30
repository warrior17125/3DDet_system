import os
import cv2
import numpy as np
from nuscenes import NuScenes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm 
import io
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

plt.switch_backend('Agg')  # 禁用交互模式

def create_scene_folders(base_path, scene_name):
    """创建场景对应的文件夹结构"""
    scene_path = os.path.join(base_path, scene_name)
    data_types = ['data', 'det', 'res']  # 新增res目录
    folders = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'PointCloud'
    ]
    for dtype in data_types:
        for folder in folders:
            os.makedirs(os.path.join(scene_path, dtype, folder), exist_ok=True)
    return scene_path

def get_camera_annotations(nusc, sample_data_token, imsize):
    """获取摄像头标注信息（2D边界框）"""
    sample_data = nusc.get('sample_data', sample_data_token)
    calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    boxes = nusc.get_boxes(sample_data_token)
    
    annotations = []
    for box in boxes:
        box = box.copy()
        
        # 转换到ego坐标系
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        # 转换到相机坐标系
        box.translate(-np.array(calibrated_sensor['translation']))
        box.rotate(Quaternion(calibrated_sensor['rotation']).inverse)
        
        # 投影到图像平面
        cam_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
        corners_3d = box.corners()
        corners_img = view_points(corners_3d, cam_intrinsic, normalize=True)[:2, :]
        
        # 过滤无效的框
        in_front = np.any(corners_3d[2, :] > 0)
        if not in_front:
            continue
            
        # 计算2D边界框
        min_x = np.min(corners_img[0])
        min_y = np.min(corners_img[1])
        max_x = np.max(corners_img[0])
        max_y = np.max(corners_img[1])
        
        # 调整到图像范围内
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(imsize[0]-1, max_x)
        max_y = min(imsize[1]-1, max_y)
        
        # 计算归一化坐标
        x_center = ((min_x + max_x) / 2) / imsize[0]
        y_center = ((min_y + max_y) / 2) / imsize[1]
        width = (max_x - min_x) / imsize[0]
        height = (max_y - min_y) / imsize[1]
        
        annotations.append(f"{box.name} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return annotations

def get_lidar_annotations(nusc, sample_data_token):
    """获取点云标注信息（3D边界框）"""
    sample_data = nusc.get('sample_data', sample_data_token)
    calibrated_sensor = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    boxes = nusc.get_boxes(sample_data_token)
    
    annotations = []
    for box in boxes:
        box = box.copy()
        
        # 转换到ego坐标系
        box.translate(-np.array(ego_pose['translation']))
        box.rotate(Quaternion(ego_pose['rotation']).inverse)
        
        # 转换到雷达坐标系
        box.translate(-np.array(calibrated_sensor['translation']))
        box.rotate(Quaternion(calibrated_sensor['rotation']).inverse)
        
        # 获取框参数
        center = box.center
        wlh = box.wlh
        yaw = box.orientation.yaw_pitch_roll[0]
        
        annotations.append(
            f"{box.name} {center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
            f"{wlh[0]:.6f} {wlh[1]:.6f} {wlh[2]:.6f} {yaw:.6f}"
        )
    
    return annotations

def fig_to_cv_no_margin(fig):
    """将 matplotlib 图形转换为 OpenCV 格式，同时去除白边"""
    buf = io.BytesIO()
    # 提高 dpi 值以提升图片像素
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    return img

def get_scene_samples(nusc, scene_name):
    """根据场景名称获取所有样本"""
    # 查找匹配的场景
    scene = None
    for s in nusc.scene:
        if s['name'] == scene_name:
            scene = s
            break
    
    if not scene:
        raise ValueError(f"'{scene_name}' not found in the dataset.")
    
    # 遍历场景中的所有样本
    samples = []
    current_sample_token = scene['first_sample_token']
    
    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)
        samples.append(sample)
        current_sample_token = sample['next']
    
    return samples

def save_scene_data(nusc, scene_name, output_base):
    """保存指定场景的所有数据"""
    samples = get_scene_samples(nusc, scene_name)
    scene_path = create_scene_folders(output_base, scene_name)
    print(f"场景 '{scene_name}' 包含 {len(samples)} 帧数据")
    
    for idx, sample in enumerate(tqdm(samples, desc=f'处理场景 {scene_name}')):
        # 处理摄像头数据
        for cam_name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
            cam_token = sample['data'][cam_name]
            sample_data = nusc.get('sample_data', cam_token)
            
            # 保存图像数据
            for dtype, with_anns in [('data', False), ('det', True)]:
                fig = plt.figure(figsize=(6, 3))
                ax = fig.add_subplot(111)
                nusc.render_sample_data(cam_token, with_anns=with_anns, ax=ax)
                ax.axis('off')
                ax.set_title('')
                save_path = os.path.join(
                    scene_path,
                    dtype,
                    cam_name,
                    f"{sample_data['timestamp']:018d}.jpg"
                )
                cv2.imwrite(save_path, fig_to_cv_no_margin(fig))
                plt.close(fig)
            
            # 保存摄像头标注信息
            res_path = os.path.join(
                scene_path,
                'res',
                cam_name,
                f"{sample_data['timestamp']:018d}.txt"
            )
            imsize = (sample_data['width'], sample_data['height'])
            annotations = get_camera_annotations(nusc, cam_token, imsize)
            with open(res_path, 'w') as f:
                f.write('\n'.join(annotations))
        
        # 处理点云数据
        lidar_token = sample['data']['LIDAR_TOP']
        sample_data = nusc.get('sample_data', lidar_token)
        
        # 保存点云图像
        for dtype, with_anns in [('data', False), ('det', True)]:
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            nusc.render_sample_data(lidar_token, with_anns=with_anns, ax=ax)
            ax.axis('off')
            ax.set_title('')
            save_path = os.path.join(
                scene_path,
                dtype,
                'PointCloud',
                f"{sample_data['timestamp']:018d}.jpg"
            )
            cv2.imwrite(save_path, fig_to_cv_no_margin(fig))
            plt.close(fig)
        
        # 保存点云标注信息
        res_path = os.path.join(
            scene_path,
            'res',
            'PointCloud',
            f"{sample_data['timestamp']:018d}.txt"
        )
        annotations = get_lidar_annotations(nusc, lidar_token)
        with open(res_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(samples)} 帧")

if __name__ == '__main__':
    # 配置参数
    DATAROOT = 'E:/DataSets/nuScenes'
    VERSION = 'v1.0-mini'
    OUTPUT_DIR = './scene_output'
    SCENE_NAME = 'scene-0061'  # 修改为目标场景名称
    
    # 初始化数据集
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    
    try:
        save_scene_data(nusc, SCENE_NAME, OUTPUT_DIR)
        print(f"场景 '{SCENE_NAME}' 数据保存完成")
    except ValueError as e:
        print(f"错误: {str(e)}")