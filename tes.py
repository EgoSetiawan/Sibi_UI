from flask import Flask, render_template, Response, jsonify, request, url_for,redirect
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pandas as pd
import logging
import threading
import time
from collections import deque
import json
import gc
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

mp_holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    model_complexity=0
)

# Configuration class
class AppConfig:
    def __init__(self):
        self.threshold = 0.5
        self.available_models = {
            ### Load model here
            'lstm': '',
            'gru': ''
        }
        self.current_model = 'lstm'
        self.lock = threading.Lock()

app_config = AppConfig()

# load label from csv
df = pd.read_csv("summary.csv")
actions = np.array(df['label'].unique())
logger.info(f"Loaded actions: {actions}")

class ModelSystem:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()


    def load_model(self, model_name):
        with self.lock:
            if model_name not in app_config.available_models:
                raise ValueError(f"Model {model_name} not found")
                
            model_path = app_config.available_models[model_name]
            try:
                if self.model:
                    tf.keras.backend.clear_session()
                    self.model = None
                
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"Loaded {model_name.upper()} model")

            except Exception as e:
                logger.error(f"Model load error: {str(e)}")
                raise

model_system = ModelSystem()
model_system.load_model(app_config.current_model)

class CameraManager:
    def __init__(self):
        self.camera_index = 0 ### change it for different camera
        self.cap = None
        self.active = False
        self.lock = threading.Lock()
        self.last_frame = None
        self.backend = cv2.CAP_DSHOW
        self.frame_counter = 0

    def start(self):
        with self.lock:
            if not self.active:
                try:
                    self.cap = cv2.VideoCapture(self.camera_index, self.backend)
                    if not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(self.camera_index)
                    
                    if self.cap.isOpened():
                        self._configure_camera()
                        self.active = True
                        logger.info("Camera initialized successfully")
                        return True
                    logger.error("Failed to initialize camera")
                    return False
                except Exception as e:
                    logger.error(f"Camera error: {str(e)}")
                    return False
            return True

    def read(self):
        with self.lock:
            if self.active and self.cap.isOpened():
                success = False
                frame = None
                try:
                    success, frame = self.cap.read()
                    if success:
                        self.frame_counter += 1
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        return success, frame
                except Exception as e:
                    logger.error(f"Frame read error: {str(e)}")
                return success, None
            return False, None

    def _configure_camera(self):
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) ## for external camera if u want
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        logger.info("Camera configured: 640x480 @ 30fps")

    def stop(self):
        with self.lock:
            if self.active:
                self.cap.release()
                self.active = False
                logger.info("Camera fully stopped")

camera_manager = CameraManager()

class PredictionTracker:
    def __init__(self, max_history=5):
        self.history = deque(maxlen=max_history)
        self.lock = threading.Lock()
        
    def update(self, frame_time, predictions):
        with self.lock:
            self.history.append({
                'time': frame_time,  
                'predictions': predictions 
            })
    
    def get_latest(self):
        with self.lock:
            return self.history[-1] if self.history else None

prediction_tracker = PredictionTracker()

class ProcessingState:
    def __init__(self):
        self.lock = threading.Lock()
        self.sentence = []
        self.predictions = []
        self.sequence = []

processing_state = ProcessingState()



def draw_landmarks(image, results):
    """Draw holistic landmarks on image"""
    # draw landmark for pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=1)
        )
    
    ### Uncomment this if you want to display face too
    # draw landmark for Face
    # if results.face_landmarks:
    #     mp_drawing.draw_landmarks(
    #         image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #         mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #         mp_drawing.DrawingSpec(color=(80,200,121), thickness=1, circle_radius=1)
    #     )
    
    # draw landmark for Left Hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=1)
        )
    
    # draw landmark for Right Hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
        )
    
    return image

def create_error_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, message, (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def extract_coordinates(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, face, lh, rh])

def mediapipe_detection(image, model):
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results

def process_frame(frame):
    try:
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        processed_frame, results = mediapipe_detection(frame, mp_holistic_model)

        if any([results.pose_landmarks, results.face_landmarks, 
                results.left_hand_landmarks, results.right_hand_landmarks]):
            processed_frame = draw_landmarks(processed_frame, results)
            keypoints = extract_coordinates(results)
        else:
            keypoints = None

        # Prediction logic
        if keypoints is not None:
            input_data = np.expand_dims(keypoints, axis=0)
            
            with tf.device('/CPU:0'):
                res = model_system.model.predict(input_data, verbose=0)[0]
            
            # Update predictions and UI
            sorted_preds = sorted(zip(actions, res), key=lambda x: x[1], reverse=True)[:5]
            
            prediction_tracker.update(time.time(), {
            'scores': {action: float(confidence) for action, confidence in sorted_preds}
            })
            

        return processed_frame

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return np.zeros((480, 640, 3), dtype=np.uint8)

@app.route('/')
def homepage():
    return render_template('home.html')


@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/switch_model', methods=['POST'])
def switch_model():
    model_name = request.json.get('model')
    logger.info(print(f'INI ADA ADALAH :{model_name}'))
    if not model_name or model_name not in app_config.available_models:
        return jsonify(success=False, error="Invalid model"), 400
    try:
        model_system.load_model(model_name)
        app_config.current_model = model_name
        logger.info(print(f'INI ADA ADALAH :{model_name}'))
        return jsonify(
            success=True,
            current_model=model_name,
            message=f"Switched to {model_name.upper()} model"
        )
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    try:
        new_threshold = float(request.json.get('threshold', 0.5))
        if not 0 <= new_threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
            
        app_config.threshold = new_threshold
        logger.info(f"Updated confidence threshold to {new_threshold}")
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 400

@app.route('/predictions')
def get_predictions():
    latest = prediction_tracker.get_latest()
    if latest:
        sorted_preds = sorted(latest['predictions']['scores'].items(), 
                        key=lambda x: x[1], 
                        reverse=True)
        
        return jsonify({
            'timestamp': latest['time'],
            'predictions': [{'action': k, 'confidence': v} for k, v in sorted_preds]
        })
    return jsonify({'predictions': []})

@app.route('/video_feed')
def video_feed():
    def generate():
        camera_manager.start()
        while True:
            success, frame = camera_manager.read()
            logger.info("Frame read - Success: %s, Frame present: %s", success, frame is not None)
            if not success:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' 
                       + cv2.imencode('.jpg', np.zeros((480,640,3), dtype=np.uint8)[1].tobytes() 
                       + b'\r\n'))
                continue
            
            processed_frame = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    success = camera_manager.start()
    return jsonify(success=success)

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    camera_manager.stop()
    return jsonify(success=True)

@app.teardown_appcontext
def cleanup(exception=None):
    global sequence
    sequence = []

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        camera_manager.stop()
