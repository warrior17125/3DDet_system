from flask import Flask, jsonify, send_file
from nuscenes_utils import NuScenesLoader
import config

app = Flask(__name__)
nusc_loader = NuScenesLoader(config.DATA_PATH)

@app.route('/api/sample/<sample_token>')
def get_sample_data(sample_token):
    data = nusc_loader.get_sample_data(sample_token)
    return jsonify(data)

@app.route('/pointcloud/<path:filename>')
def serve_pointcloud(filename):
    return send_file(f"{config.DATA_PATH}/{filename}")