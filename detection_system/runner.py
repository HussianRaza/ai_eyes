!pip install ultralytics

from google.colab import drive
drive.mount('/content/drive')


from detection_system.app import DetectionApp

# Initialize the application
app = DetectionApp(
    currency_model_path = "/content/drive/MyDrive/currencydetectyolo/detect/train/weights/best.pt",
    object_model_path = 'yolo11m.pt'    # Replace with actual path to general object model
    conf_threshold=0.25
)

# Process webcam feed
app.process_webcam(fps_target=10)

# Or process a video file
app.process_video(
    input_path="input.mp4",
    output_path="output.mp4"
)



