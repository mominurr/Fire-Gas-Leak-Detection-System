from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import random
import threading
import time
import tempfile
import os,traceback
import joblib
import pandas as pd
import numpy as np
from ultralytics import YOLO
# from gradio_client import Client, handle_file
from queue import Empty, Queue
import warnings
warnings.filterwarnings("ignore")
vision_queue = Queue()
app = Flask(__name__)

MAX_RETRIES = 3  # Maximum number of retries
BASE_TIMEOUT = 1  # Base timeout for the first attempt (1 second)
MAX_TIMEOUT = 22  # Maximum timeout between retries (22 seconds)

# Initialize models path and load the models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vision_model_path = os.path.join(BASE_DIR, "models", "fire_gas_leak_detection_vision_model", "best.pt")
threat_model_label_path = os.path.join(BASE_DIR, "models", "threat_prediction_rf_model", "label_encoder.pkl")
threat_model_scaler_path = os.path.join(BASE_DIR, "models", "threat_prediction_rf_model", "scaler.pkl")
threat_model_path = os.path.join(BASE_DIR, "models", "threat_prediction_rf_model", "threat_model.pkl")

#load the models
VISION_MODEL = YOLO(vision_model_path)  # YOLOv10 model trained for smoke/fire
THREAT_MODEL = joblib.load(threat_model_path)
LABEL_ENCODER_MODEL = joblib.load(threat_model_label_path)
SCALER_MODEL = joblib.load(threat_model_scaler_path)

# print(VISION_MODEL.names)
# # Print the expected input size and stride
# print(f"Model stride: {VISION_MODEL.stride}")

# Shared data structure
shared_data = {
    "rf_prediction": "System Initializing",
    "vision_prediction": "System Initializing",
    "combined_prediction": "System Initializing",
    "inputs": {
        "temperature": 0.0,
        "humidity": 0.0,
        "mq2_smoke": 0,
        "mq135_gas": 0,
        "flame_detected": 0,
        "cv_flame_score": 0.0,
        "cv_smoke_score": 0.0,
        "person_detected": 0
    }
}

data_lock = threading.Lock()

# Video capture configuration
video_source = 0  # 0 for webcam, path for file
cap = cv2.VideoCapture(video_source)
cap_lock = threading.Lock()



def detect_fire_smoke_from_image(image_path):
    global VISION_MODEL
    try:
        # # Read the image from file path
        # image = cv2.imread(image_path)
        # # Resize the input image before passing it to the model
        # image = cv2.resize(image, (416, 416))  # or try 416, 320, etc.
        results = VISION_MODEL(image_path)
        flame_score = 0.0
        smoke_score = 0.0
        person_detected = 0

        for result in results:
            # print(f"Model detected classes: {result.names}")
            for box in result.boxes:
                label = result.names[int(box.cls)]
                conf = float(box.conf)
                # print(f"Detected: {label} with confidence {conf}")
                if "fire" in label.lower():
                    flame_score = max(flame_score, conf)
                elif "smoke" in label.lower():
                    smoke_score = max(smoke_score, conf)
                elif "person" in label.lower():
                    person_detected = 1

        return {
            "cv_flame_score": round(flame_score, 3),
            "cv_smoke_score": round(smoke_score, 3),
            "person_detected": person_detected
        }
    except Exception as e:
        print(f"error in fire smoke dtect function: {e}")
        return {}


def predict_threat(rf_input):
    global THREAT_MODEL, LABEL_ENCODER_MODEL, SCALER_MODEL
    try:
        # Pass the dictionary as a list to DataFrame
        data = pd.DataFrame([rf_input])

        # Scale the data
        scaled_data = SCALER_MODEL.transform(data)

        # Get prediction
        prediction = THREAT_MODEL.predict(scaled_data)

        # Decode the prediction
        threat_label = LABEL_ENCODER_MODEL.inverse_transform(prediction)
        threat_label = str(threat_label[0])
        
        return {"prediction": threat_label}
    except Exception as e:
        print(f"error in threat prediction: {e}")
        return {}



# Generate synthetic sensor data
def get_synthetic_sensor_data():
    return {
            "temperature": round(random.uniform(20, 50), 1),
            "humidity": round(random.uniform(10, 90), 1),
            "mq2_smoke": random.randint(100, 1000),
            "mq135_gas": random.randint(100, 1000),
            "flame_detected": random.randint(0, 1)
        }


def generate_frames():
    while True:
        # Read frame with minimal lock duration
        success, frame = False, None
        # Get frame with minimal locking
        with cap_lock:
            if not cap.isOpened():
                time.sleep(0.1)
                continue
                
            success, frame = cap.read()
            if not success:
                if isinstance(video_source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

        # Rest of processing without lock ----------------------------------
        # Process frame for vision model
        vision_pred = "Not Detected"
        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_file.write(buffer)
                temp_file_path = temp_file.name

            vision_result = detect_fire_smoke_from_image(temp_file_path)
            # print(vision_result)
            os.unlink(temp_file_path)
            flame_score = vision_result.get('cv_flame_score', 0.0)
            smoke_score = vision_result.get('cv_smoke_score', 0.0)
            person_detected = vision_result.get('person_detected', 0.0)

            if person_detected >= 0.5 and (flame_score > 0.5 or smoke_score > 0.5):
                vision_pred = "Evacuate Immediately" 
            elif flame_score > 0.5 or smoke_score > 0.5:
                vision_pred = "Fire Detected"
            elif flame_score > 0.3 and smoke_score > 0.3:
                vision_pred = "Warning"
            else:
                vision_pred = "Safe"

            sensor_data = get_synthetic_sensor_data()
            # Get Random Forest prediction
            rf_input = {
                "temperature": sensor_data["temperature"],
                "humidity": sensor_data["humidity"],
                "mq2_smoke": sensor_data["mq2_smoke"],
                "mq135_gas": sensor_data["mq135_gas"],
                "flame_detected": sensor_data["flame_detected"],
                "cv_flame_score": flame_score,
                "cv_smoke_score": smoke_score,
                "person_detected": person_detected
            }
            rf_result = predict_threat(rf_input)
            rf_pred = rf_result.get('prediction', 'Not Detected')

        except Exception as e:
            print(f"[Error] Model processing failed: {e}")
            rf_pred = "⚠️ Unable to analyze sensor data at the moment. Please try again later."
            vision_pred = "⚠️ Unable to process image data at the moment. Please try again later."
            rf_input = {}

        # Define severity scores
        severity = {
            "Evacuate Immediately": 4,
            "Fire Detected": 3,
            "Gas Leak": 2,
            "Warning": 1,
            "Safe": 0
        }

        # Get the prediction with higher severity
        try:
            rf_severity = severity.get(rf_pred, 0)
            vision_severity = severity.get(vision_pred, 0)

            if rf_severity >= vision_severity:
                combined_pred = rf_pred
            else:
                combined_pred = vision_pred
            with data_lock:
                shared_data["inputs"].update(rf_input)
                shared_data["vision_prediction"] = vision_pred
                shared_data["rf_prediction"] = rf_pred
                shared_data["combined_prediction"] = combined_pred
        except Exception as er:
            print(f"error in update prediction result: {er}")
            pass
 
        # Encode frame after processing
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')    


@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def handle_upload():
    global cap, video_source
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or file.filename == '' or not file.filename.endswith('.mp4'):
        return jsonify({"error": "Invalid file type. Please upload a .mp4 file."}), 400

    try:
        temp_path = tempfile.mkstemp(suffix='.mp4')[1]
        file.save(temp_path)

        with cap_lock:
            # Cleanup existing resources
            if cap.isOpened():
                cap.release()
            
            # Create new capture
            new_cap = cv2.VideoCapture(temp_path)
            if not new_cap.isOpened():
                os.unlink(temp_path)
                return jsonify({"error": "Failed to open video file"}), 400

            # Atomic update
            cap = new_cap
            video_source = temp_path

        return jsonify({
            "message": f"Loaded {file.filename}",
            "path": temp_path,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    

@app.route('/webcam')
def enable_webcam():
    with cap_lock:
        if cap.isOpened():
            cap.release()
        global video_source
        video_source = 0
        cap.open(0)
    return jsonify({"message": "Webcam activated"})

@app.route('/get_data')
def get_data():
    with data_lock:
        return jsonify(shared_data)

if __name__ == '__main__':
    app.run()

