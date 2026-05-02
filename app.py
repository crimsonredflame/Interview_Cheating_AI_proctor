import cv2
import time
from detector import ProctorVision

def main():
    # Initialize the Edge-AI Vision system
    vision = ProctorVision()
    cap = cv2.VideoCapture(0)
    
    # Timer variables for gaze deviation
    look_away_start = None
    ALARM_THRESHOLD = 5 # 5-second threshold
    
    print("Proctoring System Active. Press 'q' to stop.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w, _ = frame.shape
        yolo_results, mesh_results = vision.get_frame_data(frame)

        # 1. YOLOv8 Detection: Phones and Multiple People
        person_count = 0
        for box in yolo_results.boxes:
            cls = int(box.cls[0])
            
            # Detect Phones (Class 67)
            if cls == 67:
                cv2.putText(frame, "FLAG: PHONE DETECTED", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Detect Persons (Class 0)
            if cls == 0:
                person_count += 1
                # Optional: Draw blue boxes around all detected people
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if cls == 73:
                cv2.putText(frame, "FLAG: UNAUTHORIZED MATERIAL", (50, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

        # Alert if more than one person is present
        if person_count > 1:
            cv2.putText(frame, f"FLAG: {person_count} PEOPLE DETECTED", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 2. Gaze Deviation Logic (Iris Ratio)
        is_looking_away = False

        if mesh_results.face_landmarks:
            for face_landmarks in mesh_results.face_landmarks:
                # Extract landmarks for the right eye
                p_outer = face_landmarks[33]
                p_inner = face_landmarks[133]
                p_iris = face_landmarks[468]

                # Iris Ratio Calculation
                eye_width = abs(p_inner.x - p_outer.x)
                if eye_width > 0:
                    iris_pos = abs(p_iris.x - p_outer.x) / eye_width
                    
                    # Threshold for "Looking Away"
                    if iris_pos < 0.38 or iris_pos > 0.62:
                        is_looking_away = True

                # Draw green iris dots for visual confirmation
                for idx in range(468, 478):
                    pt = face_landmarks[idx]
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (0, 255, 0), -1)
        else:
            # Face missing from frame triggers the timer
            is_looking_away = True

        # 3. Warning & Critical Flag Management
        if is_looking_away:
            if look_away_start is None:
                look_away_start = time.time()
            
            elapsed = time.time() - look_away_start
            color = (0, 165, 255) if elapsed < ALARM_THRESHOLD else (0, 0, 255)
            
            cv2.putText(frame, f"WARNING: Look Away ({int(elapsed)}s)", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if elapsed > ALARM_THRESHOLD:
                cv2.putText(frame, "CRITICAL FLAG: GAZE DEVIATION", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            look_away_start = None
            cv2.putText(frame, "Status: Monitoring", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Agentic AI Proctor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
