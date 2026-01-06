import os
import sys
import base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from pathlib import Path
from flask import Flask
from flask import request,render_template, jsonify, send_file

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.img_transform import transform_predict, predict_bbox
from src.models_creator import ResNetCardiacDetectorModel

checkpoint_file = "app/static/resnet_cardiac_detection.ckpt"
model = ResNetCardiacDetectorModel()
last_result_image = None

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("upload_file.html")

@app.route('/', methods=['POST'])
def predicted():
    if 'dicom_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['dicom_file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    img_tensor, dcm_pixel_array = transform_predict(file)
    bbox = predict_bbox(img_tensor=img_tensor, checkpoint_file=checkpoint_file, model=model)

    fig, axis = plt.subplots(1, 1, figsize=(8, 8))
    x,y,w,h = bbox[0],bbox[1],bbox[2],bbox[3] 
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
    axis.imshow(dcm_pixel_array, cmap='bone')
    axis.add_patch(rect)

    # Format plot BEFORE saving
    axis.axis('off')
    plt.tight_layout()

    # Save to BytesIO
    img_io = BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight', dpi=100)
    img_io.seek(0)
    
    # Store the bytes for download
    last_result_image = img_io.read()
    
    plt.close(fig)
    
    # Encode to base64 for display
    image_base64 = base64.b64encode(last_result_image).decode('utf-8')
    
    print(f"Image stored: {len(last_result_image)} bytes")  # Debug line

    return render_template(
        'results.html',
        image_data=image_base64,
        bbox=bbox 
    )
  


if __name__ == '__main__':
    app.run(debug=True)
