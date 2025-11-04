# Face Mask Detection

A deep learning project for detecting face masks in real-time using Convolutional Neural Networks (CNN). The model achieves 97%+ accuracy on test data.

## 🚀 Live Demo

Try the application live: **[Face Mask Detection App](https://facemaskdetection-emxx.streamlit.app/)**

## Features

- **High Accuracy**: 97.09% test accuracy with >95% precision and recall
- **Real-time Detection**: Webcam-based detection with live feedback
- **Batch Processing**: Process single images or entire directories
- **GPU Support**: Optimized for NVIDIA GPUs via CUDA/cuDNN

## Dataset

This project uses the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) from Kaggle.

The dataset contains images organized into two categories:
- `with_mask/`: Images of people wearing face masks
- `without_mask/`: Images of people not wearing face masks

After downloading the dataset, organize it in the following structure:
```
dataset/
├── with_mask/     # Images with masks
└── without_mask/  # Images without masks
```

## Project Structure

```
face_mask_detection/
├── scripts/
│   ├── explore_data.py      # Dataset exploration and visualization
│   ├── prepare_data.py      # Data preprocessing and train/val/test split
│   ├── train.py             # Model training script
│   ├── predict.py           # Inference on images/directories/webcam
│   ├── webcam_detection.py  # Real-time webcam detection
│   ├── test_model.py        # Quick model testing with dataset samples
│   └── check_gpu.py         # GPU setup verification
├── models/                   # Trained model files (not in repo)
├── dataset/                  # Training dataset (not in repo)
├── data/processed/          # Preprocessed data (not in repo)
└── requirements.txt         # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ and cuDNN 8.6+ (for GPU training)
- NVIDIA GPU with compute capability 7.0+ (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd face_mask_detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/WSL
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

### 1. Download and Prepare Dataset

Download the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) from Kaggle and organize it as shown in the Dataset section above.

### 2. Explore Dataset

```bash
python scripts/explore_data.py
```

This generates a preview of your dataset and statistics.

### 3. Preprocess Data

```bash
python scripts/prepare_data.py
```

This script:
- Loads and resizes images to 224x224
- Normalizes pixel values to [0, 1]
- Splits data into train/validation/test (80/10/10)
- Saves preprocessed data to `data/processed/`

### 4. Train Model

```bash
# Basic training
python scripts/train.py

# With custom parameters
python scripts/train.py --epochs 50 --batch_size 64 --mixed_precision
```

**Arguments:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 32)
- `--mixed_precision`: Enable mixed precision for faster training on GPU

The trained model will be saved to `models/mask_detector.h5` and the best checkpoint to `models/mask_detector_best.h5`.

### 5. Test Model

```bash
# Quick test with dataset samples
python scripts/test_model.py

# Test on specific image
python scripts/predict.py --image path/to/image.jpg

# Test on directory
python scripts/predict.py --directory path/to/images/
```

### 6. Real-time Webcam Detection

```bash
python scripts/webcam_detection.py
```

**Controls:**
- `Q`: Quit
- `S`: Save current frame

## Model Architecture

The CNN model consists of:
- **3 Convolutional Blocks**: 32, 64, 128 filters with Batch Normalization
- **MaxPooling**: Downsampling after each convolutional block
- **Fully Connected Layers**: 256 neurons with 50% dropout
- **Output**: Binary classification (sigmoid activation)

## Performance

- **Accuracy**: 97.09%
- **Precision**: 98.35% (with mask), 95.93% (without mask)
- **Recall**: 95.71% (with mask), 98.43% (without mask)
- **F1-Score**: 97.01% (weighted average)

## GPU Setup (Optional)

For GPU training on WSL2/Linux:

1. Install CUDA 12.2 and cuDNN
2. Verify GPU detection:
   ```bash
   python scripts/check_gpu.py
   ```
3. Training will automatically use GPU if available

## Requirements

- numpy >= 1.24.0
- opencv-python >= 4.8.0
- tensorflow >= 2.13.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- pillow >= 10.0.0

## Web Application

A Streamlit web app is included for easy deployment and testing.

### Local Deployment

```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

**Note:** The model file (`mask_detector_best.h5`) is too large (>100MB) for GitHub. You have two options:

#### Option 1: Host Model Externally (Recommended)

1. **Upload model to Google Drive or Dropbox**
   - Upload `models/mask_detector_best.h5` to a cloud storage service
   - Get a public download link
   
2. **Deploy to Streamlit Cloud:**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Click "New app"
   - Select repository: `Emxxlenis/face_mask_detection`
   - Main file path: `app.py`
   - Python version: **3.10** (specified in `runtime.txt`)
   - In "Advanced settings", add secret:
     - Key: `MODEL_URL`
     - Value: Your model download URL
   - Click "Deploy"

#### Option 2: Local Deployment Only

For local testing, simply place `mask_detector_best.h5` in the `models/` directory and run:

```bash
streamlit run app.py
```

The app will automatically detect and load the local model.

## License

This project is open source and available under the MIT License.

