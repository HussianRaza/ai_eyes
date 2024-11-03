def setup_colab():
    # Install required packages
    !pip install ultralytics opencv-python
    
    # Mount Google Drive
    from google.colab import drive
    drive.mount('/content/drive')
    
  

def main():
    from detection_system.app import DetectionApp
    
    print("Initializing Detection System...")
    app = DetectionApp(
        currency_model_path="/content/drive/MyDrive/currencydetectyolo/detect/train/weights/best.pt",
        object_model_path="yolov8m.pt",
        conf_threshold=0.25
    )
    
    while True:
        print("\nSelect operation mode:")
        print("1. Process Webcam Feed")
        print("2. Process Video File")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            print("\nStarting webcam processing... Press Ctrl+C to stop.")
            try:
                app.process_webcam(fps_target=10)
            except KeyboardInterrupt:
                print("\nWebcam processing stopped.")
                
        elif choice == "2":
            input_path = input("\nEnter input video path (in Google Drive): ")
            output_path = input("Enter output video path (in Google Drive): ")
            print("\nProcessing video...")
            app.process_video(input_path, output_path)
            print("Video processing completed!")
            
        elif choice == "3":
            print("\nExiting...")
            break
            
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    print("Setting up Colab environment...")
    setup_colab()
    main()