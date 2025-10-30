import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import cv2
import pandas as pd
import os
from datetime import datetime

def subjectChoose(text_to_speech_func):
    """
    Creates a modern UI for entering a subject and starting attendance.
    """
    subject_window = ttk.Toplevel()
    subject_window.title("Punjab Rural School - Take Attendance")
    subject_window.geometry("650x450")
    subject_window.resizable(False, False)

    # --- UI Elements with new styling ---
    header_label = ttk.Label(
        subject_window,
        text="Enter Subject Name",
        font=("Helvetica", 28, "bold"),
        style='primary.TLabel',
        anchor="center"
    )
    header_label.pack(fill=X, pady=(50, 30))

    entry_frame = ttk.Frame(subject_window)
    entry_frame.pack(pady=20, fill=X, padx=60)

    ttk.Label(entry_frame, text="Subject:", font=("Helvetica", 16)).pack(side=LEFT, padx=(0, 15))
    subject_entry = ttk.Entry(entry_frame, font=("Helvetica", 16), width=25)
    subject_entry.pack(side=LEFT, fill=X, expand=True, ipady=5)
    subject_entry.focus()


    def on_fill_attendance_click():
        subject = subject_entry.get()
        if not subject:
            messagebox.showerror("Error", "Please enter a subject name.", parent=subject_window)
            return

        subject_window.destroy()
        recognize_attendance(subject, text_to_speech_func)

    # --- Button ---
    buttons_frame = ttk.Frame(subject_window)
    buttons_frame.pack(pady=40, expand=True)

    fill_button = ttk.Button(
        buttons_frame,
        text="Start Attendance",
        command=on_fill_attendance_click,
        style='success.TButton',
        width=20
    )
    fill_button.pack(pady=10, ipady=15)

def recognize_attendance(subject, text_to_speech_func):
    """
    Recognizes faces and marks attendance for the given subject.
    """
    haar_cascade_path = "haarcascade_frontalface_default.xml"
    trainer_path = "./TrainingImageLabel/Trainner.yml"
    student_details_file = "./StudentDetails/studentdetails.csv"
    attendance_folder = "Attendance"

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainer_path)
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        df_students = pd.read_csv(student_details_file)

        id_map = {index + 1: (str(row['ID']), row['NAME']) for index, row in df_students.iterrows()}

    except Exception as e:
        messagebox.showerror("Error", f"Failed to load necessary files: {e}")
        return

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    col_names = ["ID", "NAME", "TIME"]
    date_str = datetime.now().strftime("%Y-%m-%d")
    attendance_file_path = os.path.join(attendance_folder, f"Attendance_{date_str}_{subject}.csv")

    if not os.path.exists(attendance_folder):
        os.makedirs(attendance_folder)

    attendance = pd.DataFrame(columns=col_names)
    if os.path.exists(attendance_file_path):
        attendance = pd.read_csv(attendance_file_path)


    while True:
        ret, im = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (85, 127, 255), 2)

            serial, conf = recognizer.predict(gray[y : y + h, x : x + w])

            if conf < 50:
                if serial in id_map:
                    alnum_id, name = id_map[serial]
                    ts = datetime.now().strftime("%H:%M:%S")

                    if alnum_id not in attendance['ID'].values:
                        new_entry = pd.DataFrame([{"ID": alnum_id, "NAME": name, "TIME": ts}])
                        attendance = pd.concat([attendance, new_entry], ignore_index=True)
                        attendance.to_csv(attendance_file_path, index=False)
                        text_to_speech_func(f"Attendance marked for {name}")

                    display_text = f"{name}"
                    cv2.putText(im, display_text, (x, y - 10), font, 1, (255, 255, 255), 2)
                else:
                    cv2.putText(im, "Unknown", (x, y - 10), font, 1, (0, 0, 255), 2)
            else:
                cv2.putText(im, "Unknown", (x, y - 10), font, 1, (0, 0, 255), 2)

        cv2.imshow("Taking Attendance - Press 'q' to exit", im)
        if cv2.waitKey(1) == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    
    if os.path.exists(attendance_file_path):
        messagebox.showinfo("Success", f"Attendance has been saved to:\n{attendance_file_path}")
        os.startfile(attendance_file_path)
    else:
        messagebox.showinfo("Info", "No new attendance was recorded.")

