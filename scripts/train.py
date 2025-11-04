"""
CNN model training script.

Trains a convolutional neural network for binary face mask classification.
Supports GPU training with mixed precision. Saves models and evaluation metrics.
"""

import json
from pathlib import Path
import argparse
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_project_root() -> Path:
    """Get the root directory of the project."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def load_preprocessed_data(data_dir: Path) -> Tuple[np.ndarray, ...]:
    """
    Load preprocessed training, validation, and test data.
    
    Args:
        data_dir: Path to the processed data directory.
        
    Returns:
        Tuple containing (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'validation'
    test_dir = data_dir / 'test'
    
    print("Loading preprocessed data...")
    X_train = np.load(train_dir / 'X_train.npy')
    y_train = np.load(train_dir / 'y_train.npy')
    X_val = np.load(val_dir / 'X_val.npy')
    y_val = np.load(val_dir / 'y_val.npy')
    X_test = np.load(test_dir / 'X_test.npy')
    y_test = np.load(test_dir / 'y_test.npy')
    
    print(f"  ✓ Training set:   {X_train.shape[0]:,} samples")
    print(f"  ✓ Validation set: {X_val.shape[0]:,} samples")
    print(f"  ✓ Test set:       {X_test.shape[0]:,} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_metadata(data_dir: Path) -> Dict:
    """
    Load metadata from the processed data directory.
    
    Args:
        data_dir: Path to the processed data directory.
        
    Returns:
        Dictionary containing metadata information.
    """
    metadata_path = data_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    return {}


def create_data_augmentation() -> ImageDataGenerator:
    """
    Create an ImageDataGenerator for data augmentation.
    
    Returns:
        Configured ImageDataGenerator instance.
    """
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )


def build_cnn_model(input_shape: Tuple[int, int, int] = (224, 224, 3)) -> Sequential:
    """
    Build a Convolutional Neural Network model for binary classification.
    
    Architecture:
    - 3 Convolutional blocks with increasing filters (32, 64, 128)
    - Batch Normalization after each convolutional layer
    - MaxPooling for downsampling
    - Fully connected layers with dropout for regularization
    
    Args:
        input_shape: Shape of input images (height, width, channels).
        
    Returns:
        Compiled Keras Sequential model.
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        # Ensure numerical stability under mixed precision by forcing float32 output
        Dense(1, activation='sigmoid', dtype='float32')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(
    history: Dict,
    output_path: Path
) -> None:
    """
    Plot training history (accuracy and loss curves).
    
    Args:
        history: Training history dictionary from model.fit().
        output_path: Path where the plot will be saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Training history saved to: {output_path}")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    class_names: Tuple[str, str] = ('Without Mask', 'With Mask')
) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        output_path: Path where the plot will be saved.
        class_names: Names of the classes for display.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Confusion matrix saved to: {output_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN for Face Mask Detection")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision (float16) for faster training on supported GPUs",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to train the face mask detection CNN model.
    
    This function:
    1. Loads preprocessed data
    2. Creates data augmentation pipeline
    3. Builds and compiles CNN model
    4. Trains the model with callbacks
    5. Evaluates on test set
    6. Generates and saves visualizations
    7. Saves the trained model
    """
    # Get project paths
    project_root = get_project_root()
    data_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Face Mask Detection - Model Training")
    print("=" * 70)
    
    # Load data and metadata
    X_train, y_train, X_val, y_val, X_test, y_test = load_preprocessed_data(data_dir)
    metadata = load_metadata(data_dir)
    
    # Get image shape from metadata or data
    if metadata:
        input_shape = tuple(metadata['image_shape'])
        print(f"\nImage shape from metadata: {input_shape}")
    else:
        input_shape = X_train.shape[1:]
        print(f"\nImage shape from data: {input_shape}")
    
    # Parse CLI args
    args = parse_args()

    # Optionally enable mixed precision
    if args.mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')
        print("\nMixed precision enabled (policy = mixed_float16)")

    # Create data augmentation
    print("\nCreating data augmentation pipeline...")
    datagen = create_data_augmentation()
    print("  ✓ Data augmentation configured")
    
    # Build model
    print("\nBuilding CNN model...")
    model = build_cnn_model(input_shape=input_shape)
    print("\nModel Architecture:")
    model.summary()
    
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        filepath=str(models_dir / 'mask_detector_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [early_stopping, model_checkpoint]
    
    # Train model
    print("\n" + "=" * 70)
    print("Training Model")
    print("=" * 70)
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs} (with early stopping)")
    print("=" * 70 + "\n")
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=max(1, len(X_train) // args.batch_size),
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on Test Set")
    print("=" * 70)
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype('int').flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nTest Set Results:")
    print(f"  - Accuracy:  {accuracy*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives (Without Mask):  {cm[0, 0]}")
    print(f"  False Positives:                {cm[0, 1]}")
    print(f"  False Negatives:                {cm[1, 0]}")
    print(f"  True Positives (With Mask):     {cm[1, 1]}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Without Mask', 'With Mask'],
        digits=4
    ))
    
    # Check if metrics meet requirements
    print("\n" + "=" * 70)
    print("Metrics Check")
    print("=" * 70)
    
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
    
    metrics_met = []
    if accuracy >= 0.95:
        metrics_met.append("✓ Accuracy: 95%+ (Required: >=95%)")
    else:
        metrics_met.append(f"✗ Accuracy: {accuracy*100:.2f}% (Required: >=95%)")
    
    if precision >= 0.90:
        metrics_met.append("✓ Precision: >90% (Required: >90%)")
    else:
        metrics_met.append(f"✗ Precision: {precision*100:.2f}% (Required: >90%)")
    
    if recall >= 0.90:
        metrics_met.append("✓ Recall: >90% (Required: >90%)")
    else:
        metrics_met.append(f"✗ Recall: {recall*100:.2f}% (Required: >90%)")
    
    for metric in metrics_met:
        print(f"  {metric}")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    plot_training_history(history, project_root / 'training_history.png')
    plot_confusion_matrix(
        y_test, y_pred,
        project_root / 'confusion_matrix.png',
        class_names=('Without Mask', 'With Mask')
    )
    
    # Save final model
    print("\n" + "=" * 70)
    print("Saving Model")
    print("=" * 70)
    
    final_model_path = models_dir / 'mask_detector.h5'
    model.save(str(final_model_path))
    print(f"  ✓ Model saved to: {final_model_path}")
    
    print("\n" + "=" * 70)
    print("Training Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

