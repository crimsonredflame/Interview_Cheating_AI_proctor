import cv2
import time
from detector import ProctorVision

def main():
    # Initialize the Edge-AI Vision system
    vision = ProctorVision()
    cap = cv2.VideoCapture(0)
    
    # Timer variables for gaze deviation
    look_away_start = None
    ALARM_THRESHOLD = 5 # Set to 5 seconds as requested
    
    print("Proctoring System Active. Press 'q' to stop.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Get frame dimensions for coordinate conversion
        h, w, _ = frame.shape

        # Retrieve AI data from YOLOv8 and MediaPipe Tasks
        yolo_results, mesh_results = vision.get_frame_data(frame)

        # 1. Phone Detection (YOLOv8)
        for box in yolo_results.boxes:
            cls = int(box.cls[0])
            if cls == 67: # COCO Class 67 is 'cell phone'
                cv2.putText(frame, "FLAG: PHONE DETECTED", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 2. Gaze Deviation Logic
        is_looking_away = False

        if mesh_results.face_landmarks:
            for face_landmarks in mesh_results.face_landmarks:
                # Coordinate extraction for Iris Ratio calculation
                # Landmarks: 33 (Right Eye Outer), 133 (Right Eye Inner), 468 (Right Iris Center)
                p_outer = face_landmarks[33]
                p_inner = face_landmarks[133]
                p_iris = face_landmarks[468]

                # Calculate relative iris position
                # Formula: Ratio = |x_iris - x_outer| / |x_inner - x_outer|
                eye_width = abs(p_inner.x - p_outer.x)
                if eye_width > 0:
                    iris_pos = abs(p_iris.x - p_outer.x) / eye_width
                    
                    # Thresholds: 0.38 to 0.62 is the 'center' safe zone
                    if iris_pos < 0.38 or iris_pos > 0.62:
                        is_looking_away = True

                # Visualization: Draw Iris Landmarks
                for idx in range(468, 478):
                    pt = face_landmarks[idx]
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 1, (0, 255, 0), -1)
        else:
            # If no face is detected at all, it counts as looking away
            is_looking_away = True

        # 3. Timer and Alert Management
        if is_looking_away:
            if look_away_start is None:
                look_away_start = time.time()
            
            elapsed = time.time() - look_away_start
            
            # Visual warnings
            color = (0, 165, 255) if elapsed < ALARM_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, f"WARNING: Look Away ({int(elapsed)}s)", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if elapsed > ALARM_THRESHOLD:
                cv2.putText(frame, "CRITICAL FLAG: GAZE DEVIATION", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            # Reset timer when user looks back at the screen
            look_away_start = None
            cv2.putText(frame, "Status: Monitoring", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the proctoring feed
        cv2.imshow('Agentic AI Proctor', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()