"""
Quick model testing script.

Tests model accuracy using random samples from the dataset.
"""

import random
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
    # Resize
    image_resized = cv2.resize(image_rgb, target_size)
    # Normalize
    image_normalized = image_resized.astype('float32') / 255.0
    # Add batch dimension
    return np.expand_dims(image_normalized, axis=0)


def test_model():
    """Test the model with sample images from the dataset."""
    project_root = get_project_root()
    model_path = project_root / 'models' / 'mask_detector.h5'
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please make sure the model file exists.")
        return
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(str(model_path))
    print("✓ Model loaded successfully\n")
    
    # Get dataset directories
    with_mask_dir = project_root / 'dataset' / 'with_mask'
    without_mask_dir = project_root / 'dataset' / 'without_mask'
    
    # Get sample images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    with_mask_images = [f for f in with_mask_dir.iterdir() 
                       if f.suffix.lower() in image_extensions]
    without_mask_images = [f for f in without_mask_dir.iterdir() 
                          if f.suffix.lower() in image_extensions]
    
    if not with_mask_images or not without_mask_images:
        print("Error: No images found in dataset directories")
        return
    
    # Select random samples (5 from each category)
    num_samples = min(5, len(with_mask_images), len(without_mask_images))
    sample_with_mask = random.sample(with_mask_images, num_samples)
    sample_without_mask = random.sample(without_mask_images, num_samples)
    
    print("=" * 70)
    print("Testing Model with Dataset Images")
    print("=" * 70)
    print(f"\nTesting {num_samples} images from each category...\n")
    
    # Test images WITH mask (should predict True)
    print("Testing images WITH mask (expected: MASK):")
    print("-" * 70)
    correct_with_mask = 0
    for img_path in sample_with_mask:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        preprocessed = preprocess_image(image)
        prediction = model.predict(preprocessed, verbose=0)[0][0]
        has_mask = prediction > 0.5
        confidence = prediction if has_mask else (1 - prediction)
        
        status = "✓ CORRECT" if has_mask else "✗ WRONG"
        if has_mask:
            correct_with_mask += 1
        
        print(f"  {img_path.name[:40]:40} | {status} | {confidence*100:5.1f}% | {'MASK' if has_mask else 'NO MASK'}")
    
    # Test images WITHOUT mask (should predict False)
    print(f"\nTesting images WITHOUT mask (expected: NO MASK):")
    print("-" * 70)
    correct_without_mask = 0
    for img_path in sample_without_mask:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        preprocessed = preprocess_image(image)
        prediction = model.predict(preprocessed, verbose=0)[0][0]
        has_mask = prediction > 0.5
        confidence = prediction if has_mask else (1 - prediction)
        
        status = "✓ CORRECT" if not has_mask else "✗ WRONG"
        if not has_mask:
            correct_without_mask += 1
        
        print(f"  {img_path.name[:40]:40} | {status} | {confidence*100:5.1f}% | {'MASK' if has_mask else 'NO MASK'}")
    
    # Summary
    total_correct = correct_with_mask + correct_without_mask
    total_tested = num_samples * 2
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Images WITH mask:    {correct_with_mask}/{num_samples} correct ({correct_with_mask/num_samples*100:.1f}%)")
    print(f"Images WITHOUT mask: {correct_without_mask}/{num_samples} correct ({correct_without_mask/num_samples*100:.1f}%)")
    print(f"Overall accuracy:    {total_correct}/{total_tested} correct ({total_correct/total_tested*100:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    test_model()

