A real-time, Edge-AI Video Analytics system designed for automated exam proctoring. This system detects common cheating behaviors—such as using a mobile phone or deviating gaze—using high-precision deep learning models optimized for local hardware performance

->Key Features
Phone Detection: Utilizes YOLOv8 to identify mobile devices in the frame with high confidence.

Gaze & Iris Tracking: Employs the MediaPipe Tasks API to monitor iris positions in real-time.

Automated Flagging: Features a state-based timer that triggers a "CRITICAL FLAG" if a user's gaze deviates from the screen for more than 5 seconds.

Edge Optimized: Specifically configured for XNNPACK acceleration, ensuring smooth performance on Apple Silicon (M1/M2/M3) and modern CPUs.

->Tech Stack
Language: Python 3.11

AI Frameworks: Ultralytics (YOLOv8), MediaPipe (Face Landmarker Tasks)

Vision: OpenCV

Hardware Acceleration: Apple M1 Neural Engine / CPU XNNPACK

