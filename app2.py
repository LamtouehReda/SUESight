from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import os
import uuid  # To generate unique filenames

app = Flask(__name__)
CORS(app)  # Enable CORS

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = os.path.join('static', 'results')  # Store in 'static' for Flask
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Load model
model = YOLO("best.pt")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Generate unique filename
        unique_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{file.filename}")
        file.save(upload_path)

        # Run detection (save to 'static/results')
        results = model.predict(
            upload_path,
            save=True,
            project=RESULTS_FOLDER,
            name="detections",
            exist_ok=True
        )

        # Get the saved image path
        result_filename = f"detections/{os.path.basename(upload_path)}"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)

        if not os.path.exists(result_path):
            return jsonify({"error": "Detection failed (no output)"}), 500

        # Return relative URL (Flask serves 'static' automatically)
        return jsonify({
            "result_url": f"/static/results/{result_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)