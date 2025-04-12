from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")  # Your custom model

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Run YOLO detection
    results = model.predict(filepath, save=True, project=RESULTS_FOLDER, name="detection")
    
    # Path to the saved result image
    result_filename = f"detection/{file.filename}"
    return jsonify({"result": result_filename})

@app.route('/results/<path:filename>')
def get_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)