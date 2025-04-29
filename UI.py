import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import base64
from io import BytesIO

app = Flask(__name__)

# 固定目录路径
FIXED_DIRECTORY = r"E:\Code\3DDet_system\scene_output"

def get_image_base64(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    # 获取固定目录下的所有子文件夹
    scene_folders = []
    if os.path.exists(FIXED_DIRECTORY):
        scene_folders = [f.name for f in os.scandir(FIXED_DIRECTORY) if f.is_dir()]
    
    selected_scene = request.form.get('scene_folder') if request.method == 'POST' else None
    groups = []
    
    if selected_scene and selected_scene != "请选择场景":
        scene_path = os.path.join(FIXED_DIRECTORY, selected_scene)
        
        # 检查必须存在的子目录
        required_dirs = ['data', 'det']
        if not all(os.path.exists(os.path.join(scene_path, d)) for d in required_dirs):
            return render_template('index.html', 
                               scene_folders=scene_folders,
                               error="所选场景需要包含data和det目录")
        
        # 名称映射字典
        name_mapping = {
            'CAM_FRONT': '前视角',
            'CAM_FRONT_RIGHT': '右前视角',
            'CAM_BACK_RIGHT': '右后视角',
            'CAM_BACK': '后视角',
            'CAM_BACK_LEFT': '左后视角',
            'CAM_FRONT_LEFT': '左前视角',
            'PointCloud': '点云鸟瞰视角'
        }
        
        # 处理两组数据
        for group_name in ['data', 'det']:
            group_path = os.path.join(scene_path, group_name)
            sequences = []
            
            # 获取7个子文件夹
            subfolders = []
            for f in os.scandir(group_path):
                if f.is_dir():
                    subfolders.append(f.path)
                    if len(subfolders) >= 7:
                        break
            
            if len(subfolders) < 7:
                return render_template('index.html', 
                                   scene_folders=scene_folders,
                                   error=f"{group_name}目录需要至少7个子文件夹")
            
            for sub in subfolders:
                images = sorted([f for f in os.listdir(sub) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
                if not images:
                    continue
                    
                original_name = os.path.basename(sub)
                display_name = name_mapping.get(original_name, original_name)
                
                sequences.append({
                    'original_name': original_name,
                    'name': display_name,
                    'images': [os.path.join(sub, img) for img in images]
                })
            
            # 重新排序前6个摄像头
            if len(sequences) >= 6:
                # 第一步：交换特定摄像头
                front_index = next((i for i, seq in enumerate(sequences[:6]) if seq['original_name'] == 'CAM_FRONT'), None)
                front_left_index = next((i for i, seq in enumerate(sequences[:6]) if seq['original_name'] == 'CAM_FRONT_LEFT'), None)
                back_index = next((i for i, seq in enumerate(sequences[:6]) if seq['original_name'] == 'CAM_BACK'), None)
                back_left_index = next((i for i, seq in enumerate(sequences[:6]) if seq['original_name'] == 'CAM_BACK_LEFT'), None)
                
                if front_index is not None and front_left_index is not None:
                    sequences[front_index], sequences[front_left_index] = sequences[front_left_index], sequences[front_index]
                
                if back_index is not None and back_left_index is not None:
                    sequences[back_index], sequences[back_left_index] = sequences[back_left_index], sequences[back_index]
                
                # 第二步：交换FRONT排和BACK排
                front_indices = [i for i, seq in enumerate(sequences[:6]) if 'FRONT' in seq['original_name']]
                back_indices = [i for i, seq in enumerate(sequences[:6]) if 'BACK' in seq['original_name']]
                
                # 交换两排位置
                if len(front_indices) == 3 and len(back_indices) == 3:
                    sequences[0], sequences[3] = sequences[3], sequences[0]
                    sequences[1], sequences[4] = sequences[4], sequences[1]
                    sequences[2], sequences[5] = sequences[5], sequences[2]
        
            groups.append({
                'name': '采集数据' if group_name == 'data' else '检测结果',  # 修改组名显示
                'sequences': sequences
            })
    
    return render_template('index.html',
                         scene_folders=scene_folders,
                         groups=groups,
                         selected_scene=selected_scene)

@app.route('/get_image')
def get_image():
    image_path = request.args.get('path')
    return jsonify({'image': get_image_base64(image_path)})

if __name__ == '__main__':
    app.run(debug=True)