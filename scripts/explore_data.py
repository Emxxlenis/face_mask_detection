"""
Dataset exploration and visualization script.

Generates dataset statistics and preview images.
"""

import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
from PIL import Image


def get_project_root() -> Path:
    """Get the root directory of the project."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def count_dataset_images(dataset_path: Path) -> int:
    """
    Count the number of image files in a dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory containing images.
        
    Returns:
        int: Number of image files in the directory.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    return len(list(dataset_path.iterdir()))


def load_sample_images(dataset_path: Path, num_samples: int = 5) -> list[Image.Image]:
    """
    Load a sample of images from a dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory.
        num_samples: Number of sample images to load (default: 5).
        
    Returns:
        list[Image.Image]: List of PIL Image objects.
    """
    image_files = sorted(list(dataset_path.iterdir()))[:num_samples]
    images = []
    for img_file in image_files:
        try:
            img = Image.open(img_file)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {img_file}: {e}")
    return images


def visualize_dataset_samples(
    with_mask_images: list[Image.Image],
    without_mask_images: list[Image.Image],
    output_path: Path,
) -> None:
    """
    Create a visualization grid showing sample images from both dataset categories.
    
    Args:
        with_mask_images: List of PIL Images from the 'with_mask' category.
        without_mask_images: List of PIL Images from the 'without_mask' category.
        output_path: Path where the visualization will be saved.
    """
    num_samples = max(len(with_mask_images), len(without_mask_images))
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    # Handle case when there's only one sample (axes becomes 1D)
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    # Display images with mask
    for idx, img in enumerate(with_mask_images):
        axes[0, idx].imshow(img)
        axes[0, idx].set_title('With Mask', fontsize=10)
        axes[0, idx].axis('off')
    
    # Hide empty subplots in the first row
    for idx in range(len(with_mask_images), num_samples):
        axes[0, idx].axis('off')
    
    # Display images without mask
    for idx, img in enumerate(without_mask_images):
        axes[1, idx].imshow(img)
        axes[1, idx].set_title('Without Mask', fontsize=10)
        axes[1, idx].axis('off')
    
    # Hide empty subplots in the second row
    for idx in range(len(without_mask_images), num_samples):
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Preview image saved to: {output_path}")
    plt.show()


def main() -> None:
    """
    Main function to explore and visualize the face mask detection dataset.
    
    This function:
    1. Sets up paths to dataset directories
    2. Counts images in each category
    3. Loads sample images
    4. Creates and saves a visualization
    """
    # Get project root directory
    project_root = get_project_root()
    
    # Define dataset paths
    dataset_with_mask = project_root / 'dataset' / 'with_mask'
    dataset_without_mask = project_root / 'dataset' / 'without_mask'
    output_path = project_root / 'dataset_preview.png'
    
    # Count images in each category
    print("=" * 60)
    print("Face Mask Detection Dataset Exploration")
    print("=" * 60)
    
    try:
        with_mask_count = count_dataset_images(dataset_with_mask)
        without_mask_count = count_dataset_images(dataset_without_mask)
        total_count = with_mask_count + without_mask_count
        
        print(f"\nDataset Statistics:")
        print(f"  - Images with mask:    {with_mask_count:,}")
        print(f"  - Images without mask: {without_mask_count:,}")
        print(f"  - Total images:        {total_count:,}")
        
        # Load sample images for visualization
        print(f"\nLoading sample images...")
        with_mask_samples = load_sample_images(dataset_with_mask, num_samples=5)
        without_mask_samples = load_sample_images(dataset_without_mask, num_samples=5)
        
        if with_mask_samples and without_mask_samples:
            print(f"Visualizing dataset samples...")
            visualize_dataset_samples(
                with_mask_samples,
                without_mask_samples,
                output_path
            )
        else:
            print("Warning: Could not load enough sample images for visualization.")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset directories exist.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
