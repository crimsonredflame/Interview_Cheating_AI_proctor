import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

class ProctorVision:
    def __init__(self):
        # 1. Load YOLOv8 for phone detection (the file from your Desktop)
        self.yolo = YOLO('yolov8n.pt') 

        # 2. Setup MediaPipe Face Landmarker (New API)
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_frame_data(self, frame):
        # YOLO detection
        results = self.yolo(frame, verbose=False, conf=0.5)[0]

        # MediaPipe detection (Convert OpenCV BGR to MediaPipe Image)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = self.detector.detect(mp_image)

        return results, detection_result