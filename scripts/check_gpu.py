"""
GPU setup verification script.

Checks if TensorFlow can detect and use NVIDIA GPU.
"""

import sys

print("=" * 70)
print("GPU Setup Verification")
print("=" * 70)
print()

# Check TensorFlow installation
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError:
    print("✗ TensorFlow is not installed")
    print("  Please install: pip install tensorflow")
    sys.exit(1)

# Check for GPU
print("\nChecking for available GPUs...")
print("-" * 70)

gpus = tf.config.list_physical_devices('GPU')

if len(gpus) > 0:
    print(f"✓ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # Get GPU details
    print("\nGPU Details:")
    print("-" * 70)
    try:
        for gpu in gpus:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  Name: {gpu.name}")
            if 'device_name' in details:
                print(f"  Device: {details['device_name']}")
            print()
        
        # Test GPU computation
        print("Testing GPU computation...")
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print("✓ GPU computation test successful!")
        
        print("\n" + "=" * 70)
        print("GPU SETUP: SUCCESS ✓")
        print("TensorFlow can use your GPU for training!")
        print("=" * 70)
        
    except Exception as e:
        print(f"✗ Error testing GPU: {e}")
        print("\nGPU may be detected but not fully functional.")
        print("Make sure CUDA and cuDNN are properly installed.")
        
else:
    print("✗ No GPUs detected")
    print("\nPossible reasons:")
    print("  1. CUDA Toolkit is not installed")
    print("  2. cuDNN is not installed")
    print("  3. CUDA/cuDNN versions are incompatible with TensorFlow")
    print("  4. NVIDIA drivers are not up to date")
    print("\nFor TensorFlow 2.13.0, you need:")
    print("  - CUDA 11.8")
    print("  - cuDNN 8.6")
    print("\nSee GPU_SETUP.md for installation instructions.")
    print("\nTraining will continue with CPU (slower).")

print("\nTensorFlow Build Information:")
print("-" * 70)
print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"  Built with cuDNN: {tf.test.is_built_with_gpu_support()}")



