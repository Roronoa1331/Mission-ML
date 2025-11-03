"""
Flask Web Application for Casting Defect Detection - FULLY FIXED VERSION
Compatible with all NumPy versions
Run: python app.py
Access: http://localhost:5000
"""

import os
import io
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import base64
import cv2
import json

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global variable for model
model = None
IMG_SIZE = (224, 224)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types - UPDATED FOR MODERN NUMPY"""
    def default(self, obj):
        # Handle integer types
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        # Handle float types
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        # Handle boolean types - FIXED for modern NumPy
        elif hasattr(obj, 'dtype') and obj.dtype == np.bool_:
            return bool(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Use the custom encoder
app.json_encoder = NumpyEncoder

def load_model():
    """Load the trained model"""
    global model
    try:
        # Try to load the model (use your actual model path)
        model_paths = [
            'optimized_transfer_model.h5',
            'optimized_custom_cnn_model.h5',
            'xception_transfer_model.h5',
            'custom_cnn_model.h5', 
            'quick_start_model.h5'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                model = keras.models.load_model(path)
                print(f"✅ Model loaded from {path}")
                return True
        
        print("❌ No trained model found. Please train a model first.")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def allowed_file(filename):
    """Check if file type is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(IMG_SIZE)
    
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_array, axis=0)
    
    return image_batch, image_array

def safe_convert_to_native_types(obj):
    """
    Safely convert NumPy types to Python native types for JSON serialization
    COMPATIBLE WITH ALL NUMPY VERSIONS
    """
    try:
        # Handle numpy arrays
        if hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
            return obj.tolist()
        
        # Handle numpy scalars by checking dtype
        if hasattr(obj, 'dtype'):
            if np.issubdtype(obj.dtype, np.integer):
                return int(obj)
            elif np.issubdtype(obj.dtype, np.floating):
                return float(obj)
            elif np.issubdtype(obj.dtype, np.bool_):
                return bool(obj)
        
        # Handle specific type instances
        if isinstance(obj, (np.int32, np.int64, np.int8, np.int16)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # Handle dictionaries recursively
        elif isinstance(obj, dict):
            return {key: safe_convert_to_native_types(value) for key, value in obj.items()}
        
        # Handle lists recursively
        elif isinstance(obj, list):
            return [safe_convert_to_native_types(item) for item in obj]
        
        # Handle tuples
        elif isinstance(obj, tuple):
            return tuple(safe_convert_to_native_types(item) for item in obj)
        
        # For any other type, return as is
        return obj
        
    except Exception as e:
        print(f"Warning: Could not convert {type(obj)}: {e}")
        return str(obj)  # Fallback to string representation

def create_visualization(original_image, prediction, confidence, save_path=None):
    """Create visualization of prediction result"""
    plt.figure(figsize=(10, 8))
    
    # Plot original image
    plt.imshow(original_image)
    
    # Set title color based on prediction
    color = 'green' if prediction == 'OK' else 'red'
    status = "✅ DEFECT-FREE" if prediction == 'OK' else "❌ DEFECTIVE"
    
    plt.title(f"CASTING QUALITY INSPECTION\n{status}\nConfidence: {confidence:.3f}", 
              color=color, fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    
    # Add border color based on prediction
    for spine in plt.gca().spines.values():
        spine.set_color(color)
        spine.set_linewidth(4)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    # Convert to base64 for web display
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

def safe_predict(image_batch):
    """
    Safe prediction function that handles all type conversions
    """
    # Get prediction
    prediction = model.predict(image_batch, verbose=0)[0][0]
    
    # Convert to native Python types safely
    try:
        # Method 1: Direct conversion
        prediction_float = float(prediction)
    except:
        # Method 2: Via numpy then conversion
        prediction_float = float(np.array(prediction).item())
    
    # Determine if defective
    is_defective = prediction_float > 0.5
    
    # Calculate confidence
    if is_defective:
        confidence = prediction_float
    else:
        confidence = 1.0 - prediction_float
    
    # Ensure all values are native Python types
    result = {
        'prediction_value': prediction_float,
        'is_defective': bool(is_defective),  # Explicit bool conversion
        'confidence': float(confidence),     # Explicit float conversion
        'prediction_label': "Defective" if is_defective else "OK"
    }
    
    return result

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', model_loaded=model is not None)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction - FULLY FIXED"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Secure filename and save
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and preprocess image
            image = Image.open(filepath)
            image_batch, original_image = preprocess_image(image)
            
            # Make prediction
            if model is None:
                return jsonify({'error': 'Model not loaded. Please train a model first.'}), 500
            
            # Use safe prediction function
            prediction_result = safe_predict(image_batch)
            
            # Create visualization
            viz_data = create_visualization(
                original_image, 
                prediction_result['prediction_label'], 
                prediction_result['confidence']
            )
            
            # Prepare response with native Python types
            result = {
                'filename': filename,
                'prediction': prediction_result['prediction_label'],
                'confidence': prediction_result['confidence'],
                'raw_score': prediction_result['prediction_value'],
                'visualization': viz_data,
                'is_defective': prediction_result['is_defective']
            }
            
            # Final safety check - convert all values
            result = safe_convert_to_native_types(result)
            
            return jsonify(result)
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """Handle batch file upload - FULLY FIXED"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files')
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                # Secure filename and save
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Load and preprocess image
                image = Image.open(filepath)
                image_batch, _ = preprocess_image(image)
                
                # Make prediction
                if model is None:
                    return jsonify({'error': 'Model not loaded'}), 500
                
                # Use safe prediction
                prediction_result = safe_predict(image_batch)
                
                results.append({
                    'filename': filename,
                    'prediction': prediction_result['prediction_label'],
                    'confidence': prediction_result['confidence'],
                    'is_defective': prediction_result['is_defective']
                })
                
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e)
                })
    
    # Calculate statistics
    total = len(results)
    defective_count = sum(1 for r in results if r.get('is_defective', False))
    ok_count = total - defective_count
    defect_rate = (defective_count / total * 100) if total > 0 else 0
    
    response_data = {
        'results': safe_convert_to_native_types(results),
        'statistics': safe_convert_to_native_types({
            'total': total,
            'defective': defective_count,
            'ok': ok_count,
            'defect_rate': round(defect_rate, 2)
        })
    }
    
    return jsonify(response_data)

@app.route('/demo')
def demo():
    """Demo page with sample images"""
    return render_template('demo.html', model_loaded=model is not None)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions - FULLY FIXED"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file and allowed_file(file.filename):
        try:
            # Load and preprocess image
            image = Image.open(file.stream)
            image_batch, _ = preprocess_image(image)
            
            # Make prediction
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            
            # Use safe prediction
            prediction_result = safe_predict(image_batch)
            
            response_data = {
                'prediction': 'defective' if prediction_result['is_defective'] else 'ok',
                'confidence': prediction_result['confidence'],
                'raw_score': prediction_result['prediction_value']
            }
            
            return jsonify(safe_convert_to_native_types(response_data))
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/models')
def list_models():
    """List available models"""
    models = []
    model_files = [
        'optimized_transfer_model.h5',
        'optimized_custom_cnn_model.h5', 
        'xception_transfer_model.h5',
        'custom_cnn_model.h5',
        'quick_start_model.h5'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            models.append({
                'name': model_file,
                'size': f"{os.path.getsize(model_file) / 1024 / 1024:.1f} MB"
            })
    
    return jsonify({'models': models})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify(safe_convert_to_native_types({
        'status': 'healthy',
        'model_loaded': model is not None,
        'numpy_version': np.__version__
    }))

@app.route('/debug_types')
def debug_types():
    """Debug endpoint to check type conversions"""
    test_values = [
        np.float32(0.75),
        np.float64(0.85),
        np.bool_(True),
        np.int32(42),
        np.array([0.1, 0.2, 0.3]),
        True,
        False,
        0.99,
        1
    ]
    
    debug_info = []
    for val in test_values:
        debug_info.append({
            'original': str(val),
            'type': str(type(val)),
            'converted': safe_convert_to_native_types(val),
            'converted_type': str(type(safe_convert_to_native_types(val)))
        })
    
    return jsonify({'debug_info': debug_info})

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File size too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Casting Defect Detection Web Application...")
    print(f"📦 NumPy version: {np.__version__}")
    print("📁 Current directory:", os.getcwd())
    
    model_files = [f for f in os.listdir('.') if f.endswith('.h5')]
    print(f"🤖 Available models: {model_files}")
    
    if load_model():
        print("✅ Model loaded successfully!")
    else:
        print("⚠️  No model found. Please train a model using the training scripts.")
        print("   You can still run the app, but predictions will not work.")
    
    print("🌐 Starting Flask development server...")
    print("📱 Access the application at: http://localhost:5000")
    print("🔧 Debug mode: ON")
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)