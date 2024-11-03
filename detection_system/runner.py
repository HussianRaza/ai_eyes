# detection_system/runner.py
def main():
    from detection_system.app import DetectionApp
    
    print("Initializing Real-time Currency and Object Detection System...")
    app = DetectionApp(
        currency_model_path="/content/drive/MyDrive/currencydetectyolo/detect/train/weights/best.pt",
        object_model_path="yolo11m.pt",
        conf_threshold=0.25
    )
    
    print("\nStarting webcam detection... Press Ctrl+C to stop.")
    try:
        app.run_detection(fps_target=10)
    except KeyboardInterrupt:
        print("\nDetection stopped.")

if __name__ == "__main__":
    main()