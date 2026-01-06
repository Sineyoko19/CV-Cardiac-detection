import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from pathlib import Path
from flask import Flask
from flask import request,render_template, jsonify, Response
from app.img_transform import transform_predict, predict_bbox
from src.models_creator import ResNetCardiacDetectorModel

checkpoint_file = "app/static/resnet_cardiac_detection.ckpt"
model = ResNetCardiacDetectorModel()

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("upload_file.html")

@app.route('/predicted', methods=['POST'])
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

    img_io = BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    img_io.seek(0)

    # Return image as response
    return Response(img_io.getvalue(), mimetype='image/png')

   
if __name__ == '__main__':
    app.run(debug=True)