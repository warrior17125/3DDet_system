from nuscenes.nuscenes import NuScenes
import numpy as np
import json

class NuScenesLoader:
    def __init__(self, dataroot):
        self.nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)
        
    def get_sample_data(self, sample_token):
        sample = self.nusc.get('sample', sample_token)
        data = {
            'images': [],
            'pointcloud': None,
            'annotations': []
        }
        
        # 处理摄像头数据
        for cam_channel in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
            cam_data = self.nusc.get_sample_data_path(sample['data'][cam_channel])
            data['images'].append({
                'url': f"/images/{cam_data.split('/')[-1]}",
                'calibration': self.get_camera_calibration(cam_channel)
            })
        
        # 处理点云数据
        lidar_data = self.nusc.get_sample_data(sample['data']['LIDAR_TOP'])
        data['pointcloud'] = {
            'path': lidar_data[0].split('/')[-1],
            'points': lidar_data[1].tolist()
        }
        
        # 处理标注数据
        annotations = self.nusc.get_boxes(sample_token)
        for ann in annotations:
            data['annotations'].append({
                'token': ann.token,
                'category': ann.name,
                'bbox3d': self._get_3d_bbox(ann),
                'color': self._get_color(ann.name)
            })
        
        return data

    def _get_3d_bbox(self, ann):
        return {
            'translation': ann.center.tolist(),
            'size': ann.wlh.tolist(),
            'rotation': ann.orientation.q.tolist()
        }
    
    def _get_color(self, category):
        # 定义类别颜色映射
        color_map = {
            'car': '#FF0000',
            'pedestrian': '#00FF00',
            # ...其他类别
        }
        return color_map.get(category, '#FFFFFF')