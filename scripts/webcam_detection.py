"""
Real-time face mask detection using webcam.

Uses mask_detector_best.h5 by default.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from tensorflow.keras.models import load_model


def get_project_root() -> Path:
    """Get the root directory of the project."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def preprocess_image(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess an image for model input."""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    image_resized = cv2.resize(image_rgb, target_size)
    # Normalize to [0, 1]
    image_normalized = image_resized.astype('float32') / 255.0
    # Add batch dimension
    return np.expand_dims(image_normalized, axis=0)


def main():
    """Main function for real-time webcam detection."""
    project_root = get_project_root()
    
    # Try to load the best model, fallback to regular model
    best_model_path = project_root / 'models' / 'mask_detector_best.h5'
    regular_model_path = project_root / 'models' / 'mask_detector.h5'
    
    if best_model_path.exists():
        model_path = best_model_path
        print(f"Loading best model: {model_path.name}")
    elif regular_model_path.exists():
        model_path = regular_model_path
        print(f"Loading model: {model_path.name}")
    else:
        print("Error: No model found. Please make sure mask_detector_best.h5 or mask_detector.h5 exists in models/")
        sys.exit(1)
    
    # Load model
    try:
        model = load_model(str(model_path))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Make sure your webcam is connected and not being used by another application.")
        sys.exit(1)
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\n" + "=" * 70)
    print("Real-time Face Mask Detection")
    print("=" * 70)
    print(f"Webcam resolution: {width}x{height}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("=" * 70 + "\n")
    
    frame_count = 0
    save_counter = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Preprocess frame for model
            preprocessed = preprocess_image(frame)
            
            # Predict (only every few frames for better performance)
            if frame_count % 2 == 0:  # Process every 2nd frame for speed
                prediction = model.predict(preprocessed, verbose=0)[0][0]
                has_mask = prediction > 0.5
                confidence = prediction if has_mask else (1 - prediction)
            
            # Draw result on frame
            label = "MASK" if has_mask else "NO MASK"
            color = (0, 255, 0) if has_mask else (0, 0, 255)  # Green for mask, Red for no mask
            confidence_text = f"{confidence*100:.1f}%"
            
            # Create background rectangle for text
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw main label
            cv2.putText(frame, label, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Draw confidence
            cv2.putText(frame, confidence_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw colored border
            border_color = color
            cv2.rectangle(frame, (5, 5), (width-5, height-5), border_color, 5)
            
            # Show frame
            cv2.imshow('Face Mask Detection - Press Q to quit', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_counter += 1
                save_path = project_root / f'webcam_capture_{save_counter}.jpg'
                cv2.imwrite(str(save_path), frame)
                print(f"✓ Frame saved: {save_path.name}")
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam closed. Thank you for using Face Mask Detection!")


if __name__ == "__main__":
    main()

