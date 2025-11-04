"""
Data preprocessing script.

Loads images, resizes to 224x224, normalizes to [0,1], and splits into
train/validation/test sets (80/10/10). Saves preprocessed data to data/processed/.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def get_project_root() -> Path:
    """Get the root directory of the project."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def is_image_file(filename: str) -> bool:
    """
    Check if a file is a valid image file based on its extension.
    
    Args:
        filename: Name of the file to check.
        
    Returns:
        bool: True if the file has a valid image extension, False otherwise.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    return Path(filename).suffix.lower() in valid_extensions


def load_images_from_directory(
    directory: Path,
    label: int,
    image_size: Tuple[int, int] = (224, 224)
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Load and preprocess images from a directory.
    
    Args:
        directory: Path to the directory containing images.
        label: Label to assign to images from this directory (0 or 1).
        image_size: Target size for resizing images (width, height).
        
    Returns:
        Tuple containing:
            - List of preprocessed image arrays
            - List of labels corresponding to each image
    """
    images = []
    labels = []
    
    if not directory.exists():
        print(f"Warning: Directory not found: {directory}")
        return images, labels
    
    image_files = [f for f in directory.iterdir() 
                   if f.is_file() and is_image_file(f.name)]
    
    print(f"Loading images from {directory.name}...")
    for img_file in image_files:
        try:
            # Read image
            img = cv2.imread(str(img_file))
            
            # Check if image was loaded successfully
            if img is None:
                print(f"Warning: Could not read image {img_file.name}")
                continue
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, image_size)
            
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            print(f"Error processing {img_file.name}: {e}")
            continue
    
    print(f"  Loaded {len(images)} images with label {label}")
    return images, labels


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        X: Image data array.
        y: Label array.
        train_ratio: Proportion of data for training (default: 0.8).
        val_ratio: Proportion of data for validation (default: 0.1).
        test_ratio: Proportion of data for testing (default: 0.1).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple containing (X_train, y_train, X_val, y_val, X_test, y_test).
        
    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.0")
    
    # First split: train (80%) and temp (20%)
    test_size = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # Second split: validation (10%) and test (10%) from temp (20%)
    val_size = val_ratio / test_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_size,
        random_state=random_state,
        stratify=y_temp
    )
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def save_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    metadata: dict = None
) -> None:
    """
    Save preprocessed data arrays to NumPy files in an organized structure.
    
    Args:
        X_train: Training images array.
        y_train: Training labels array.
        X_val: Validation images array.
        y_val: Validation labels array.
        X_test: Test images array.
        y_test: Test labels array.
        output_dir: Base directory where files will be saved.
        metadata: Optional dictionary with metadata about the processing.
    """
    # Create main output directory structure
    processed_dir = output_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for better organization
    train_dir = processed_dir / 'train'
    val_dir = processed_dir / 'validation'
    test_dir = processed_dir / 'test'
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save training data
    print(f"\nSaving training data to {train_dir}...")
    np.save(train_dir / 'X_train.npy', X_train)
    np.save(train_dir / 'y_train.npy', y_train)
    print(f"  ✓ X_train.npy: {X_train.shape}")
    print(f"  ✓ y_train.npy: {y_train.shape}")
    
    # Save validation data
    print(f"\nSaving validation data to {val_dir}...")
    np.save(val_dir / 'X_val.npy', X_val)
    np.save(val_dir / 'y_val.npy', y_val)
    print(f"  ✓ X_val.npy: {X_val.shape}")
    print(f"  ✓ y_val.npy: {y_val.shape}")
    
    # Save test data
    print(f"\nSaving test data to {test_dir}...")
    np.save(test_dir / 'X_test.npy', X_test)
    np.save(test_dir / 'y_test.npy', y_test)
    print(f"  ✓ X_test.npy: {X_test.shape}")
    print(f"  ✓ y_test.npy: {y_test.shape}")
    
    # Save metadata if provided
    if metadata:
        metadata_path = processed_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ metadata.json saved to {metadata_path}")
    
    print(f"\nAll data saved successfully to: {processed_dir}")


def main() -> None:
    """
    Main function to prepare the dataset for training.
    
    This function:
    1. Loads images from both categories
    2. Preprocesses and normalizes images
    3. Splits data into train/validation/test sets
    4. Saves preprocessed data as NumPy arrays
    """
    # Configuration
    IMAGE_SIZE = (224, 224)
    
    # Get project root and define paths
    project_root = get_project_root()
    dataset_with_mask = project_root / 'dataset' / 'with_mask'
    dataset_without_mask = project_root / 'dataset' / 'without_mask'
    output_dir = project_root  # Base directory for creating data/processed/
    
    print("=" * 60)
    print("Face Mask Detection - Data Preparation")
    print("=" * 60)
    
    # Load images with mask (label = 1)
    images_with_mask, labels_with_mask = load_images_from_directory(
        dataset_with_mask,
        label=1,
        image_size=IMAGE_SIZE
    )
    
    # Load images without mask (label = 0)
    images_without_mask, labels_without_mask = load_images_from_directory(
        dataset_without_mask,
        label=0,
        image_size=IMAGE_SIZE
    )
    
    # Combine all images and labels
    all_images = images_with_mask + images_without_mask
    all_labels = labels_with_mask + labels_without_mask
    
    if len(all_images) == 0:
        print("Error: No images were loaded. Please check your dataset paths.")
        return
    
    # Convert to NumPy arrays and normalize
    print("\nPreprocessing images...")
    X = np.array(all_images, dtype='float32') / 255.0
    y = np.array(all_labels)
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"  - Total images:        {len(X):,}")
    print(f"  - Images with mask:    {np.sum(y == 1):,} ({np.sum(y == 1)*100/len(y):.1f}%)")
    print(f"  - Images without mask: {np.sum(y == 0):,} ({np.sum(y == 0)*100/len(y):.1f}%)")
    print(f"  - Image shape:         {X[0].shape}")
    
    # Split dataset
    print("\nSplitting dataset into train/validation/test sets...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X, y,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42
    )
    
    print("\nDataset Split:")
    print(f"  - Training:   {len(X_train):,} images ({len(X_train)*100/len(X):.1f}%)")
    print(f"  - Validation: {len(X_val):,} images ({len(X_val)*100/len(X):.1f}%)")
    print(f"  - Test:       {len(X_test):,} images ({len(X_test)*100/len(X):.1f}%)")
    
    # Prepare metadata
    metadata = {
        'image_size': IMAGE_SIZE,
        'total_images': len(X),
        'images_with_mask': int(np.sum(y == 1)),
        'images_without_mask': int(np.sum(y == 0)),
        'train_count': len(X_train),
        'validation_count': len(X_val),
        'test_count': len(X_test),
        'train_ratio': 0.8,
        'validation_ratio': 0.1,
        'test_ratio': 0.1,
        'random_state': 42,
        'normalization': 'pixel_values_scaled_to_[0,1]',
        'image_shape': list(X[0].shape)
    }
    
    # Save preprocessed data
    print(f"\nSaving preprocessed data...")
    save_data(X_train, y_train, X_val, y_val, X_test, y_test, output_dir, metadata)
    
    print("\n" + "=" * 60)
    print("Data preparation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
