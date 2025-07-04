import os
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# 固定目录路径
FIXED_DIRECTORY = r"E:\Code\3DDet_system\scene_output"

NAME_MAPPING = {
    'CAM_FRONT': '前视角',
    'CAM_FRONT_RIGHT': '右前视角',
    'CAM_BACK_RIGHT': '右后视角',
    'CAM_BACK': '后视角', 
    'CAM_BACK_LEFT': '左后视角',
    'CAM_FRONT_LEFT': '左前视角',
    'PointCloud': '点云鸟瞰视角'
}

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
    res_data = {}

    if selected_scene and selected_scene != "请选择场景":
        scene_path = os.path.join(FIXED_DIRECTORY, selected_scene)
        
        # 检查必须存在的子目录（增加res目录检查）
        required_dirs = ['data', 'det', 'res']
        if not all(os.path.exists(os.path.join(scene_path, d)) for d in required_dirs):
            return render_template('index.html', 
                               scene_folders=scene_folders,
                               error="所选场景需要包含data、det和res目录")
    
        
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
                display_name = NAME_MAPPING.get(original_name, original_name)
                
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

        res_path = os.path.join(scene_path, 'res')
        res_dirs = sorted([d.name for d in os.scandir(res_path) if d.is_dir()][:7])
        res_data = {}
        
        for sub in res_dirs:
            sub_path = os.path.join(res_path, sub)
            files = sorted([f for f in os.listdir(sub_path) if f.endswith('.txt')])
            frame_data = []
            
            for file in files:
                with open(os.path.join(sub_path, file), 'r') as f:
                    lines = f.readlines()
                    objects = []
                    for line in lines:
                        parts = line.strip().split()
                        if sub == 'PointCloud':
                            if len(parts) >= 8:
                                obj = {
                                    '类别': parts[0],
                                    'x': parts[1],
                                    'y': parts[2],
                                    'z': parts[3],
                                    'l': parts[4],
                                    'w': parts[5],
                                    'h': parts[6],
                                    'yaw': parts[7]
                                }
                        else:
                            if len(parts) >= 5:
                                obj = {
                                    '类别': parts[0],
                                    'x': parts[1],
                                    'y': parts[2],
                                    'h': parts[3],
                                    'w': parts[4]
                                }
                        if obj:
                            objects.append(obj)
                    frame_data.append({
                        'filename': file,
                        'objects': objects
                    })
            res_data[sub] = frame_data
    
    return render_template('index.html',
                         scene_folders=scene_folders,
                         groups=groups,
                         selected_scene=selected_scene,
                         res_data=res_data)


@app.route('/export_data', methods=['POST'])
def export_data():
    try:
        data = request.json
        res_data = data['resData']
        scene_name = data['scene']
        
        # 创建内存文件对象
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for sensor_type in res_data:
                # 转换数据为DataFrame
                rows = []
                for frame in res_data[sensor_type]:
                    timestamp = os.path.splitext(frame['filename'])[0]  # 提取不带后缀的文件名
                    for obj in frame['objects']:
                        row = {'时间戳': timestamp}
                        row.update(obj)
                        rows.append(row)
                
                df = pd.DataFrame(rows)
                sheet_name = NAME_MAPPING.get(sensor_type, sensor_type)
                
                # 添加数据类型转换（示例）
                if sensor_type == 'PointCloud':
                    numeric_cols = ['x', 'y', 'z', 'l', 'w', 'h', 'yaw']
                else:
                    numeric_cols = ['x', 'y', 'h', 'w']
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                
        # 生成带场景名称和当前时间的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{scene_name}_检测结果_{timestamp}.xlsx"
        
        output.seek(0)
        return send_file(
            output,
            download_name=filename,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/get_image')
def get_image():
    image_path = request.args.get('path')
    return jsonify({'image': get_image_base64(image_path)})

if __name__ == '__main__':
    app.run(debug=True)