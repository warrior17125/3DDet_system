
import cv2
import numpy as np
import matplotlib.pyplot as plt
from nuscenes import NuScenes
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def fig2array(fig):
    """ 将 matplotlib figure 转换为 numpy 数组 """
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    return img

def combine_visualization(nusc, sample_token, save_path=None):
    # 获取样本数据
    sample = nusc.get('sample', sample_token)
    
    # 相机名称映射关系
    camera_mapping = {
        'CAM_FRONT': 'cam_front',
        'CAM_FRONT_LEFT': 'cam_front_left',
        'CAM_FRONT_RIGHT': 'cam_front_right',
        'CAM_BACK': 'cam_back',
        'CAM_BACK_LEFT': 'cam_back_left',
        'CAM_BACK_RIGHT': 'cam_back_right'
    }
    
    # ================= 1. 渲染所有相机图像 =================
    image_dict = {}
    for cam_name in camera_mapping.keys():
        # 获取相机数据
        cam_token = sample['data'][cam_name]
        data_path, _, _ = nusc.get_sample_data(cam_token)
        
        # 读取原始图像尺寸
        with Image.open(data_path) as img:
            orig_w, orig_h = img.size
        aspect_ratio = orig_w / orig_h
        
        # 动态设置 figure 尺寸（保持原始宽高比）
        fig_h = 3  # 高度设为3英寸
        fig_w = fig_h * aspect_ratio
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)
        
        # 渲染带标注的图像
        nusc.render_sample_data(cam_token, with_anns=True, ax=ax)
        ax.axis('off')
        ax.set_title('')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # 转换为 OpenCV 格式
        img_array = fig2array(fig)
        plt.close(fig)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        image_dict[camera_mapping[cam_name]] = img_bgr
    
    # ================= 2. 渲染激光雷达BEV =================
    lidar_token = sample['data']['LIDAR_TOP']
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    nusc.render_sample_data(lidar_token, with_anns=True, ax=ax, show_lidarseg=False)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    lidar_img = fig2array(fig)
    plt.close(fig)
    lidar_img = cv2.cvtColor(lidar_img, cv2.COLOR_RGBA2BGR)
    
    # ================= 3. 图像拼接 =================
    # 前视拼接（左-中-右）
    front = cv2.hconcat([
        image_dict['cam_front_left'],
        image_dict['cam_front'],
        image_dict['cam_front_right']
    ])
    
    # 后视拼接（右-中-左）并翻转
    back = cv2.hconcat([
        image_dict['cam_back_right'],
        image_dict['cam_back'],
        image_dict['cam_back_left']
    ])
    back = cv2.flip(back, 1)  # 水平翻转
    
    # 垂直拼接前后视图
    cams_combined = cv2.vconcat([front, back])
    
    # 缩放至与激光雷达同高度
    target_h = lidar_img.shape[0]
    scale = target_h / cams_combined.shape[0]
    target_w = int(cams_combined.shape[1] * scale)
    cams_resized = cv2.resize(cams_combined, (target_w, target_h))
    
    # 最终拼接
    final_img = cv2.hconcat([cams_resized, lidar_img])
    
    # 保存或显示
    if save_path:
        cv2.imwrite(save_path, final_img)
    return final_img

# 使用示例
if __name__ == "__main__":
    # 初始化数据集
    nusc = NuScenes(version='v1.0-mini', dataroot='E:\\DataSets\\nuScenes', verbose=True)
    
    # 生成可视化结果
    result = combine_visualization(nusc, nusc.sample[0]['token'], 'combined.jpg')
    
    # 显示结果
    cv2.imshow('NuScenes Visualization', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()