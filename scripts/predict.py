"""
Face mask detection inference script.

Supports single images, directories, or webcam streams.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tensorflow.keras.models import load_model


def get_project_root() -> Path:
    """Get the root directory of the project."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def load_trained_model(model_path: Path):
    """
    Load the trained face mask detection model.
    
    Args:
        model_path: Path to the saved model file.
        
    Returns:
        Loaded Keras model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = load_model(str(model_path))
    print("✓ Model loaded successfully")
    return model


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess an image for model input.
    
    Args:
        image: Input image as numpy array (BGR format from OpenCV).
        target_size: Target size for resizing (height, width).
        
    Returns:
        Preprocessed image array ready for model input.
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    image_resized = cv2.resize(image_rgb, target_size)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype('float32') / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch


def predict_image(model, image_path: Path, threshold: float = 0.5) -> Tuple[bool, float]:
    """
    Predict if a face in an image is wearing a mask.
    
    Args:
        model: Trained Keras model.
        image_path: Path to the image file.
        threshold: Classification threshold (default: 0.5).
        
    Returns:
        Tuple of (has_mask: bool, confidence: float).
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocess
    image_preprocessed = preprocess_image(image)
    
    # Predict
    prediction = model.predict(image_preprocessed, verbose=0)[0][0]
    
    # Interpret result
    has_mask = prediction > threshold
    confidence = prediction if has_mask else (1 - prediction)
    
    return has_mask, float(confidence)


def predict_webcam(model, output_path: Optional[Path] = None):
    """
    Real-time face mask detection using webcam.
    
    Args:
        model: Trained Keras model.
        output_path: Optional path to save output video.
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print("Starting webcam... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            preprocessed = preprocess_image(frame)
            
            # Predict
            prediction = model.predict(preprocessed, verbose=0)[0][0]
            has_mask = prediction > 0.5
            confidence = prediction if has_mask else (1 - prediction)
            
            # Draw result on frame
            label = "Mask" if has_mask else "No Mask"
            color = (0, 255, 0) if has_mask else (0, 0, 255)
            confidence_text = f"{confidence*100:.1f}%"
            
            # Draw rectangle and text
            cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
            cv2.putText(frame, label, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, confidence_text, (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Face Mask Detection', frame)
            
            # Save frame if writer is set
            if writer:
                writer.write(frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Webcam closed")


def predict_directory(model, directory: Path, output_dir: Optional[Path] = None):
    """
    Predict masks for all images in a directory.
    
    Args:
        model: Trained Keras model.
        directory: Directory containing images.
        output_dir: Optional directory to save annotated images.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in directory.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No images found in {directory}")
        return
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(image_files)} images...")
    
    results = []
    for img_path in image_files:
        try:
            has_mask, confidence = predict_image(model, img_path)
            results.append((img_path.name, has_mask, confidence))
            
            status = "✓ MASK" if has_mask else "✗ NO MASK"
            print(f"  {img_path.name}: {status} ({confidence*100:.1f}%)")
            
            # Save annotated image if output directory is specified
            if output_dir:
                image = cv2.imread(str(img_path))
                label = "Mask" if has_mask else "No Mask"
                color = (0, 255, 0) if has_mask else (0, 0, 255)
                
                cv2.putText(image, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(image, f"{confidence*100:.1f}%", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                output_path = output_dir / img_path.name
                cv2.imwrite(str(output_path), image)
        
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
    
    # Print summary
    mask_count = sum(1 for _, has_mask, _ in results if has_mask)
    print(f"\nSummary: {mask_count}/{len(results)} images with mask")


def main():
    """Main function for face mask detection inference."""
    parser = argparse.ArgumentParser(
        description="Face Mask Detection Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a single image
  python scripts/predict.py --image path/to/image.jpg
  
  # Predict on all images in a directory
  python scripts/predict.py --directory path/to/images/
  
  # Use webcam for real-time detection
  python scripts/predict.py --webcam
  
  # Save webcam output to video
  python scripts/predict.py --webcam --output output.mp4
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/mask_detector_best.h5',
        help='Path to trained model (default: models/mask_detector.h5)'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image file'
    )
    parser.add_argument(
        '--directory',
        type=str,
        help='Path to directory containing images'
    )
    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Use webcam for real-time detection'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for video (webcam mode) or directory (directory mode)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Get project root and model path
    project_root = get_project_root()
    model_path = project_root / args.model
    
    # Load model
    try:
        model = load_trained_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Process based on input mode
    if args.webcam:
        output_path = Path(args.output) if args.output else None
        predict_webcam(model, output_path)
    
    elif args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image not found: {image_path}")
            sys.exit(1)
        
        try:
            has_mask, confidence = predict_image(model, image_path, args.threshold)
            label = "Mask" if has_mask else "No Mask"
            print(f"\nResult: {label}")
            print(f"Confidence: {confidence*100:.2f}%")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.directory:
        directory = Path(args.directory)
        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            sys.exit(1)
        
        output_dir = Path(args.output) if args.output else None
        predict_directory(model, directory, output_dir)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

