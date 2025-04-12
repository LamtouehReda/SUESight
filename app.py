from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
from ultralytics import YOLO
import os
import logging
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
DETECTION_SUBFOLDER = 'detection'
DETECTION_FOLDER = os.path.join(RESULTS_FOLDER, DETECTION_SUBFOLDER)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Load YOLO model
try:
    model = YOLO("best.pt")
    app.logger.info("✅ YOLO model loaded successfully.")
except Exception as e:
    app.logger.error(f"❌ Error loading YOLO model: {e}")
    model = None

@app.route('/upload', methods=['POST'])
def upload_file():
    if not model:
        return jsonify({"error": "Model not loaded."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        app.logger.info(f"✅ File saved to: {filepath}")

        # Run YOLO detection
        app.logger.info("🧠 Running YOLO prediction...")
        results = model.predict(
            source=filepath,
            save=True,
            project=RESULTS_FOLDER,
            name=DETECTION_SUBFOLDER,
            exist_ok=True
        )

        # Log result path
        result_path = os.path.join(DETECTION_FOLDER, filename)
        app.logger.info(f"🔍 Expected result path: {result_path}")

        # Check if the file exists
        if not os.path.exists(result_path):
            app.logger.warning("⚠️ Detection result not found.")
            # Find actual file name from results
            output_file = None
            for f in os.listdir(DETECTION_FOLDER):
                if f.endswith(".jpg") or f.endswith(".png"):
                    output_file = f
                    break
            if output_file:
                filename = output_file
            else:
                return jsonify({"error": "Detection output not found."}), 500

        # Create full URL
        result_url = url_for('get_result', filename=f"{DETECTION_SUBFOLDER}/{filename}", _external=True)
        app.logger.info(f"✅ Returning result URL: {result_url}")

        return jsonify({
            "message": "Detection successful!",
            "result": f"{DETECTION_SUBFOLDER}/{filename}",
            "result_url": result_url
        })

    except Exception as e:
        app.logger.error("❌ Exception during detection:")
        app.logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Detection failed: {str(e)}",
            "details": traceback.format_exc()
        }), 500

@app.route('/results/<path:filename>')
def get_result(filename):
    try:
        return send_from_directory(RESULTS_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({"error": "Result file not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
from ultralytics import YOLO
import os
import logging
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
DETECTION_SUBFOLDER = 'detection'
DETECTION_FOLDER = os.path.join(RESULTS_FOLDER, DETECTION_SUBFOLDER)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Load YOLO model
try:
    model = YOLO("best.pt")
    app.logger.info("✅ YOLO model loaded successfully.")
except Exception as e:
    app.logger.error(f"❌ Error loading YOLO model: {e}")
    model = None

@app.route('/upload', methods=['POST'])
def upload_file():
    if not model:
        return jsonify({"error": "Model not loaded."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        app.logger.info(f"✅ File saved to: {filepath}")

        # Run YOLO detection
        app.logger.info("🧠 Running YOLO prediction...")
        results = model.predict(
            source=filepath,
            save=True,
            project=RESULTS_FOLDER,
            name=DETECTION_SUBFOLDER,
            exist_ok=True
        )

        # Log result path
        result_path = os.path.join(DETECTION_FOLDER, filename)
        app.logger.info(f"🔍 Expected result path: {result_path}")

        # Check if the file exists
        if not os.path.exists(result_path):
            app.logger.warning("⚠️ Detection result not found.")
            # Find actual file name from results
            output_file = None
            for f in os.listdir(DETECTION_FOLDER):
                if f.endswith(".jpg") or f.endswith(".png"):
                    output_file = f
                    break
            if output_file:
                filename = output_file
            else:
                return jsonify({"error": "Detection output not found."}), 500

        # Create full URL
        result_url = url_for('get_result', filename=f"{DETECTION_SUBFOLDER}/{filename}", _external=True)
        app.logger.info(f"✅ Returning result URL: {result_url}")

        return jsonify({
            "message": "Detection successful!",
            "result": f"{DETECTION_SUBFOLDER}/{filename}",
            "result_url": result_url
        })

    except Exception as e:
        app.logger.error("❌ Exception during detection:")
        app.logger.error(traceback.format_exc())
        return jsonify({
            "error": f"Detection failed: {str(e)}",
            "details": traceback.format_exc()
        }), 500

@app.route('/results/<path:filename>')
def get_result(filename):
    try:
        return send_from_directory(RESULTS_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({"error": "Result file not found"}), 404

if __name__ == "__main__":
    app.run()
