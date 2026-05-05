import cv2
import time
import os
import numpy as np
import pandas as pd
import sounddevice as sd
from datetime import datetime
from detector import ProctorVision

# Setup
OS_FOLDERS = ['evidence', 'logs']
for folder in OS_FOLDERS: os.makedirs(folder, exist_ok=True)

class ProctorAnalytics:
    def __init__(self):
        self.logs, self.gaze_points = [], []
        self.last_log_time, self.audio_level = 0, 0

    def audio_callback(self, indata, frames, time, status):
        self.audio_level = np.linalg.norm(indata) * 10

    def log_event(self, event_type, frame=None):
        now = datetime.now()
        self.logs.append({"Timestamp": now.strftime("%H:%M:%S"), "Event": event_type})
        if frame is not None:
            cv2.imwrite(f"evidence/{event_type.replace(' ', '_')}_{now.strftime('%H%M%S')}.jpg", frame)
        pd.DataFrame(self.logs).to_csv("logs/proctor_report.csv", index=False)

def draw_text(img, text, pos, color):
    # Adds a black outline/shadow so text is visible on light backgrounds
    x, y = pos
    cv2.putText(img, text, (x+2, y+2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def main():
    vision = ProctorVision()
    analytics = ProctorAnalytics()
    cap = cv2.VideoCapture(0)
    stream = sd.InputStream(callback=analytics.audio_callback)
    stream.start()

    look_away_start, ALARM_THRESHOLD = None, 5
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        h, w, _ = frame.shape
        yolo_results, mesh_results = vision.get_frame_data(frame)

        # 1. Detection Loop (Person, Phone, Material)
        person_count = 0
        for box in yolo_results.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            
            if cls == 0: person_count += 1
            
            # Phone Detection (Class 67)
            if cls == 67:
                draw_text(frame, "FLAG: PHONE DETECTED", (50, 50), (0, 0, 255))
                if time.time() - analytics.last_log_time > 5:
                    analytics.log_event("Phone Detected", frame)
                    analytics.last_log_time = time.time()

            # Unauthorized Material 
            if cls == 73 and conf > 0.3:
                draw_text(frame, "FLAG: UNAUTHORIZED MATERIAL", (50, 200), (0, 165, 255))
                if time.time() - analytics.last_log_time > 5:
                    analytics.log_event("Material Detected", frame)
                    analytics.last_log_time = time.time()

        # 2. Gaze Logic
        is_looking_away = False
        if mesh_results.face_landmarks:
            for face_landmarks in mesh_results.face_landmarks:
                # Use Right Eye for calculation
                p_out, p_in, p_iris = face_landmarks[33], face_landmarks[133], face_landmarks[468]
                analytics.gaze_points.append((int(p_iris.x * w), int(p_iris.y * h)))

                eye_width = abs(p_in.x - p_out.x)
                if eye_width > 0:
                    iris_pos = abs(p_iris.x - p_out.x) / eye_width
                    # STRICTER RANGE: 0.4 - 0.6
                    if iris_pos < 0.42 or iris_pos > 0.58: is_looking_away = True
                
                # Visual Feedback: Iris dots
                for idx in range(468, 478):
                    pt = face_landmarks[idx]
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (0, 255, 0), -1)
        else:
            is_looking_away = True # Flag if face is hidden/missing

        # 3. UI Timer & Status
        if is_looking_away:
            if look_away_start is None: look_away_start = time.time()
            elapsed = int(time.time() - look_away_start)
            
            color = (0, 165, 255) if elapsed < ALARM_THRESHOLD else (0, 0, 255)
            draw_text(frame, f"WARNING: Look Away ({elapsed}s)", (50, 100), color)
            
            if elapsed >= ALARM_THRESHOLD:
                draw_text(frame, "CRITICAL: GAZE DEVIATION", (50, 150), (0, 0, 255))
                if time.time() - analytics.last_log_time > 5:
                    analytics.log_event("Gaze Deviation", frame)
                    analytics.last_log_time = time.time()
        else:
            look_away_start = None
            draw_text(frame, "Status: Monitoring", (50, 100), (0, 255, 0))

        # 4. Audio Bar
        bar_color = (0, 255, 0) if analytics.audio_level < 5 else (0, 0, 255)
        cv2.rectangle(frame, (w-160, h-20), (w-160 + int(analytics.audio_level * 10), h-40), bar_color, -1)
        cv2.putText(frame, "Audio", (w-160, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow('Proctor AI Professional', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    stream.stop()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()
