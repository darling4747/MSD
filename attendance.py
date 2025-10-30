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

# --- Configuration and File Paths ---
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
ENCODINGS_FILE = "encodings.pickle"
SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"

# --- Geo-location Configuration ---
# IMPORTANT: Replace YOUR_LATITUDE and YOUR_LONGITUDE with the coordinates you got.
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

# --- Haversine Formula for Distance Calculation ---
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth using their latitude and longitude.
    Returns distance in kilometers.
    """
    R = 6371  # Radius of Earth in kilometers
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

# --- Helper Functions for Face Detection and Encoding ---
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

# --- Main Feature Functions ---

def register_student():
    """
    Registers a new student by capturing and encoding their face.
    It prompts for the student's name, captures 10 images, and saves their face encoding.
    """
    print("\n--- Student Registration ---")
    student_name = input("Enter student's full name: ").strip().lower()
    if not student_name:
        print("Name cannot be empty. Returning to main menu.")
        return

    known_encodings, known_names = load_encodings()
    known_encodings_dict = dict(zip(known_names, known_encodings))

    if student_name in known_encodings_dict:
        print(f"Error: A student with the name '{student_name}' already exists.")
        return

    print("Please look directly at the camera. Capturing images...")
    print("Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    images_captured = 0
    while images_captured < 10:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing: {images_captured + 1}/10", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            face_encoding = face_recognition.face_encodings(rgb_frame, [face_locations[0]])[0]

            known_encodings.append(face_encoding)
            known_names.append(student_name)
            images_captured += 1
            print(f"Captured image {images_captured}/10")
            time.sleep(0.5)

        cv2.imshow("Register Student", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if images_captured > 0:
        known_encodings_dict = dict(zip(known_names, known_encodings))
        save_encodings(known_encodings_dict)
        print(f"Successfully registered student: {student_name}")
    else:
        print("No faces captured. Registration failed.")
        
def mark_attendance():
    """
    Marks attendance for students in a specific subject.
    Requires a student to blink to confirm their presence and a location check.
    """
    print("\n--- Mark Attendance ---")
    
    # --- Location Check (FIXED) ---
    print("Checking your location... Please wait.")
    
    # We will use the college location itself for the test.
    user_lat, user_lon = COLLEGE_LOCATION["lat"], COLLEGE_LOCATION["lon"]
    
    distance = haversine(user_lat, user_lon, COLLEGE_LOCATION["lat"], COLLEGE_LOCATION["lon"])
    
    if distance > LOCATION_THRESHOLD_KM:
        print(f"Location check failed. You are {distance:.2f} km away from the campus.")
        print("Attendance can only be marked from within the college premises.")
        return

    print("Location confirmed. You are on campus.")
    # --- End Location Check ---
    
    subject = input("Enter subject name: ").strip()
    if not subject:
        print("Subject cannot be empty. Returning to main menu.")
        return

    known_encodings, known_names = load_encodings()
    if not known_encodings:
        print("No student data found. Please register students first.")
        return

    face_detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR)

    def eye_aspect_ratio(eye):
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        ear = (A + B) / (2.0 * C)
        return ear

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    marked_students_session = set()
    blink_counter = 0

    print(f"Attendance for subject: {subject}")
    print("Look at the camera. Blink to mark your attendance.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_detector(gray, 0)
        face_locations = face_recognition.face_locations(frame)

        for i, dlib_rect in enumerate(faces):
            landmarks = predictor(gray, dlib_rect)
            points = [(p.x, p.y) for p in landmarks.parts()]
            left_eye = points[LEFT_EYE_START:LEFT_EYE_END]
            right_eye = points[RIGHT_EYE_START:RIGHT_EYE_END]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
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

                if ear < EYE_AR_THRESH and name != "Unknown":
                    blink_counter += 1
                else:
                    if blink_counter >= EYE_AR_CONSEC_FRAMES and name not in marked_students_session:
                        print(f"Attendance marked for: {name}")
                        marked_students_session.add(name)
                        
                        with open(ATTENDANCE_FILE, "a", newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([name, subject, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                        blink_counter = 0
            
            for (x, y) in points:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.imshow("Mark Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Attendance session ended.")

def view_attendance():
    """
    Displays the attendance records from the attendance.csv file.
    """
    print("\n--- View Attendance ---")
    if not os.path.exists(ATTENDANCE_FILE):
        print("No attendance records found.")
        return

    with open(ATTENDANCE_FILE, "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    if not data:
        print("No attendance records to display.")
        return

    print("{:<20} {:<20} {:<30}".format("Name", "Subject", "Timestamp"))
    print("-" * 70)
    for row in data:
        if len(row) >= 3:
            name, subject, timestamp = row
            print("{:<20} {:<20} {:<30}".format(name, subject, timestamp))
    print("-" * 70)

# --- Main Menu Function ---

def main():
    """
    The main function that runs the attendance system menu.
    """
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
    
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Subject", "Timestamp"])

    while True:
        print("\n--- Attendance Management System Menu ---")
        print("1. Register a new student")
        print("2. Mark attendance by subject")
        print("3. View attendance records")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            register_student()
        elif choice == '2':
            mark_attendance()
        elif choice == '3':
            view_attendance()
        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    if not os.path.exists(SHAPE_PREDICTOR):
        print(f"Error: Required file '{SHAPE_PREDICTOR}' not found.")
        print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract the .dat file and place it in the same directory as this script.")
    else:
        main()
