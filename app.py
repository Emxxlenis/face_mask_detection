"""
Face Mask Detection Web App

Streamlit web application for real-time face mask detection.
Uses the trained CNN model to classify images.
"""

import streamlit as st
from pathlib import Path
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Try to import cv2, fallback to PIL if not available
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Page config
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 1rem 0;
    }
    .mask-detected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .no-mask-detected {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_():
    """Load the trained model (cached)."""
    # Try local path first
    model_path = Path("models/mask_detector_best.h5")
    
    if model_path.exists():
        return load_model(str(model_path))
    
    # Get model URL from secrets
    model_url = None
    
    try:
        if hasattr(st, 'secrets') and 'MODEL_URL' in st.secrets:
            model_url = st.secrets["MODEL_URL"].strip()
    except Exception:
        pass
    
    if not model_url:
        st.error("Model not found. Please configure MODEL_URL in secrets.")
        st.stop()
    
    # Download model from URL (silently)
    try:
        import tempfile
        import os
        import re
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Handle Google Drive URLs with gdown
            if "drive.google.com" in model_url:
                import gdown
                
                # Extract file ID
                file_id = None
                patterns = [
                    r'/d/([a-zA-Z0-9_-]+)',
                    r'id=([a-zA-Z0-9_-]+)',
                    r'[-\w]{25,}',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, model_url)
                    if match:
                        file_id = match.group(1) if match.lastindex else match.group()
                        break
                
                if file_id:
                    download_url = f"https://drive.google.com/uc?id={file_id}"
                    gdown.download(download_url, tmp_path, quiet=True, fuzzy=True)
                else:
                    raise ValueError("Could not extract Google Drive file ID")
            else:
                # For other URLs, use urllib
                import urllib.request
                urllib.request.urlretrieve(model_url, tmp_path)
            
            # Verify file is valid HDF5
            file_size = os.path.getsize(tmp_path)
            with open(tmp_path, 'rb') as f:
                header = f.read(8)
                valid_signatures = [b'\x89HDF', b'\x0e\x03\x13\x01']
                is_valid = any(header.startswith(sig) for sig in valid_signatures)
                
                if not is_valid:
                    raise ValueError("Downloaded file is not a valid model file")
            
            # Load model
            model = load_model(tmp_path)
            os.unlink(tmp_path)
            return model
            
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def preprocess_image(image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess image for model input."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is RGB
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]
    elif len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    
    # Resize
    if HAS_CV2:
        image_resized = cv2.resize(image, target_size)
    else:
        pil_image = Image.fromarray(image.astype('uint8'))
        pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        image_resized = np.array(pil_image)
    
    # Normalize to [0, 1]
    image_normalized = image_resized.astype('float32') / 255.0
    
    # Add batch dimension
    return np.expand_dims(image_normalized, axis=0)

def main():
    """Main Streamlit app."""
    st.markdown('<h1 class="main-header">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Real-Time Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses a Convolutional Neural Network (CNN) to detect 
        face masks in images with **97%+ accuracy**.
        
        **Model Performance:**
        - Accuracy: 97.09%
        - Precision: >95%
        - Recall: >95%
        
        Upload an image or use your webcam to detect face masks!
        """)
        
        st.markdown("---")
        st.markdown("**Dataset:** [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)")
    
    # Load model (show loading spinner)
    with st.spinner("Loading AI model..."):
        try:
            model = load_model_()
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
    
    # Main content
    tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Webcam Capture"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose an image containing a face",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            col_img, col_result = st.columns([1, 1])
            
            with col_img:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            
            with col_result:
                img_array = np.array(image)
                preprocessed = preprocess_image(img_array)
                
                with st.spinner("🔍 Analyzing..."):
                    prediction = model.predict(preprocessed, verbose=0)[0][0]
                    has_mask = prediction > 0.5
                    confidence = float(prediction if has_mask else (1 - prediction))
                
                if has_mask:
                    st.markdown(
                        f'<div class="prediction-card mask-detected">'
                        f'<h2>✅ Mask Detected</h2>'
                        f'<h1>{confidence*100:.1f}%</h1>'
                        f'<p>Confidence Level</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-card no-mask-detected">'
                        f'<h2>⚠️ No Mask Detected</h2>'
                        f'<h1>{confidence*100:.1f}%</h1>'
                        f'<p>Confidence Level</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                st.progress(confidence)
                st.caption(f"Prediction confidence: {confidence*100:.2f}%")
    
    with tab2:
        camera_input = st.camera_input("📷 Capture from your webcam")
        
        if camera_input is not None:
            col_img, col_result = st.columns([1, 1])
            
            with col_img:
                image = Image.open(camera_input)
                st.image(image, use_container_width=True)
            
            with col_result:
                img_array = np.array(image)
                preprocessed = preprocess_image(img_array)
                
                with st.spinner("🔍 Analyzing..."):
                    prediction = model.predict(preprocessed, verbose=0)[0][0]
                    has_mask = prediction > 0.5
                    confidence = float(prediction if has_mask else (1 - prediction))
                
                if has_mask:
                    st.markdown(
                        f'<div class="prediction-card mask-detected">'
                        f'<h2>✅ Mask Detected</h2>'
                        f'<h1>{confidence*100:.1f}%</h1>'
                        f'<p>Confidence Level</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-card no-mask-detected">'
                        f'<h2>⚠️ No Mask Detected</h2>'
                        f'<h1>{confidence*100:.1f}%</h1>'
                        f'<p>Confidence Level</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                st.progress(confidence)
                st.caption(f"Prediction confidence: {confidence*100:.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Face Mask Detection System | CNN Model with 97% Accuracy<br>"
        "Made with ❤️ by <strong>Emily Lenis</strong>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
