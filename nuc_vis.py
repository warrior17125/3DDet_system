import os
import cv2
import numpy as np
from nuscenes import NuScenes
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm 
import io

plt.switch_backend('Agg')  # 禁用交互模式

def create_scene_folders(base_path, scene_name):
    """创建场景对应的文件夹结构"""
    scene_path = os.path.join(base_path, scene_name)
    folders = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'PointCloud'
    ]
    for folder in folders:
        os.makedirs(os.path.join(scene_path, folder), exist_ok=True)
    return scene_path

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
    # 获取场景样本
    samples = get_scene_samples(nusc, scene_name)
    
    # 创建场景文件夹
    scene_path = create_scene_folders(output_base, scene_name)
    print(f"场景 '{scene_name}' 包含 {len(samples)} 帧数据")
    
    # 遍历保存每个样本
    for idx, sample in enumerate(tqdm(samples, desc=f'处理场景 {scene_name}')):
        # 保存相机数据
        for cam_name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
            cam_token = sample['data'][cam_name]
            sample_data = nusc.get('sample_data', cam_token)
            
            fig = plt.figure(figsize=(6, 3))
            ax = fig.add_subplot(111)
            nusc.render_sample_data(cam_token, with_anns=True, ax=ax)
            ax.axis('off')
            ax.set_title('')

            save_path = os.path.join(
                scene_path,
                cam_name,
                f"{sample_data['timestamp']:018d}.jpg"
            )
            # 使用新函数转换图形，去除白边
            cv2.imwrite(save_path, fig_to_cv_no_margin(fig))
            plt.close(fig)
        
        # 保存点云数据
        lidar_token = sample['data']['LIDAR_TOP']
        sample_data = nusc.get('sample_data', lidar_token)
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        nusc.render_sample_data(lidar_token, with_anns=True, ax=ax)
        ax.axis('off')
        ax.set_title('')
        save_path = os.path.join(
            scene_path,
            'PointCloud',
            f"{sample_data['timestamp']:018d}.jpg"
        )
        # 使用新函数转换图形，去除白边
        cv2.imwrite(save_path, fig_to_cv_no_margin(fig))
        plt.close(fig)
        
        if (idx + 1) % 10 == 0:
            print(f"已处理 {idx + 1}/{len(samples)} 帧")

if __name__ == '__main__':
    # 配置参数
    DATAROOT = 'E:/DataSets/nuScenes'
    VERSION = 'v1.0-mini'
    OUTPUT_DIR = './scene_output'
    SCENE_NAME = 'scene-0655'  # 修改为目标场景名称
    
    # 初始化数据集
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    
    try:
        save_scene_data(nusc, SCENE_NAME, OUTPUT_DIR)
        print(f"场景 '{SCENE_NAME}' 数据保存完成")
    except ValueError as e:
        print(f"错误: {str(e)}")