import requests
import json
from flask import Flask, render_template, request, jsonify, Response
import cv2
import face_recognition
import numpy as np
import os
import pickle
import dlib
from datetime import datetime
import time
import csv
import math

# --- Flask and Application Setup ---
app = Flask(__name__)

# --- Configuration and File Paths ---
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
ENCODINGS_FILE = "encodings.pickle"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
# You can download the shape predictor from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# --- Geo-location Configuration ---
COLLEGE_LOCATION = {
    "lat": 16.289680053812898,  # Latitude for a college in Guntur, India
    "lon": 80.46791845841389   # Longitude for a college in Guntur, India
}
LOCATION_THRESHOLD_KM = 0.2 # 200 meters

# --- Blink Detection Parameters ---
RIGHT_EYE_START = 36
RIGHT_EYE_END = 42
LEFT_EYE_START = 42
LEFT_EYE_END = 48
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 3

# --- Global Variables for Attendance Session ---
marked_students_session = set()
camera = None

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth.
    """
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def load_encodings():
    """Loads facial encodings and names from a pickle file."""
    if not os.path.exists(ENCODINGS_FILE):
        return [], []
    with open(ENCODINGS_FILE, "rb") as f:
        known_encodings_dict = pickle.load(f)
    known_encodings = list(known_encodings_dict.values())
    known_names = list(known_encodings_dict.keys())
    return known_encodings, known_names

def save_encodings(known_encodings_dict):
    """Saves facial encodings to a pickle file."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(known_encodings_dict, f)

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_name = request.form.get('name').strip().lower()
        regd_no = request.form.get('regdNo').strip().upper()
        
        if not student_name or not regd_no:
            return "Name and Registration Number cannot be empty.", 400
        
        student_id = f"{student_name}_{regd_no}"
        
        known_encodings, known_names = load_encodings()
        if student_id in known_names:
            return "Student already registered.", 409
        
        # This is a placeholder. A real system would capture images here.
        # For this example, we'll assume a successful registration.
        
        return jsonify({"message": f"Successfully registered {student_name} ({regd_no})", "success": True})
    
    return render_template('register_student.html')

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        data = request.get_json()
        subject = data.get('subject')
        user_lat = data.get('lat')
        user_lon = data.get('lon')

        if not all([subject, user_lat, user_lon]):
            return jsonify({"error": "Missing data"}), 400

        distance = haversine(user_lat, user_lon, COLLEGE_LOCATION["lat"], COLLEGE_LOCATION["lon"])
        
        if distance > LOCATION_THRESHOLD_KM:
            return jsonify({
                "error": f"You are {distance:.2f} km away from the campus. Attendance can only be marked from within the college premises."
            }), 403
        
        # If location is valid, the webcam stream will be started via another route.
        return jsonify({"message": "Location verified. Starting attendance session."}), 200
        
    return render_template('mark_attendance.html')

def gen_frames(subject):
    global marked_students_session, camera
    
    # Initialize camera and models if not already done
    if camera is None:
        camera = cv2.VideoCapture(0)
        face_detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
    
    known_encodings, known_names = load_encodings()
    
    # Reset marked students for new session
    marked_students_session = set()
    blink_counter = 0

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)
        face_locations = face_recognition.face_locations(frame)

        for i, dlib_rect in enumerate(faces):
            landmarks = predictor(gray, dlib_rect)
            points = [(p.x, p.y) for p in landmarks.parts()]
            
            # This is a basic blink detection. A more robust system is needed for a real application.
            def eye_aspect_ratio(eye):
                A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
                B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
                C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
                ear = (A + B) / (2.0 * C)
                return ear

            left_ear = eye_aspect_ratio(points[LEFT_EYE_START:LEFT_EYE_END])
            right_ear = eye_aspect_ratio(points[RIGHT_EYE_START:RIGHT_EYE_END])
            ear = (left_ear + right_ear) / 2.0

            if i < len(face_locations):
                top, right, bottom, left = face_locations[i]
                face_encoding = face_recognition.face_encodings(frame, [face_locations[i]])[0]
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"
                
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if name != "Unknown":
                    if ear < EYE_AR_THRESH and name not in marked_students_session:
                        marked_students_session.add(name)
                        student_name, regd_no = name.rsplit('_', 1)
                        with open(ATTENDANCE_FILE, "a", newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([student_name, regd_no, subject, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                        blink_counter = 0
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<subject>')
def video_feed(subject):
    return Response(gen_frames(subject), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_attendance')
def view_attendance():
    records = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    return render_template('view_attendance.html', records=records)

if __name__ == '__main__':
    # Create necessary directories and files if they don't exist
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Regd. No", "Subject", "Timestamp"])

    app.run(debug=True)
