from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import logging
import os
import shutil


app = Flask(__name__)

# Enable CORS
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the model path
MODEL_PATH = 'Customer_churning.keras'  # Change this to your model's location

# Load the pre-trained TensorFlow model
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        
        # Get the input shape from the model
        input_shape = model.input_shape[1:]
        logger.info(f"Model expects input shape: {input_shape}")
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")
        model = None
        input_shape = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    input_shape = None

@app.route('/')
def home():
    """Serve the frontend HTML file"""
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
        
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid input: 'features' field is required"}), 400
        
        # Check if it's a batch prediction (list of lists) or single prediction (single list)
        features = data['features']
        is_batch = isinstance(features[0], list) if features else False
        
        if is_batch:
            # Batch prediction
            logger.info(f"Processing batch prediction with {len(features)} samples")
            return handle_batch_prediction(features)
        else:
            # Single prediction
            logger.info("Processing single prediction")
            return handle_single_prediction(features)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

def handle_single_prediction(features):
    """Handle single customer prediction"""
    features = np.array(features)
    
    # Log the received features
    logger.info(f"Received features: {features}")
    logger.info(f"Feature count: {len(features)}")
    
    # Expecting 10 features based on frontend
    if len(features) != 10:
        logger.warning(f"Unexpected feature count: {len(features)}. Expected 10 based on frontend.")
    
    # Reshape features for the model
    features = features.reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # For binary classification
    churn_probability = float(prediction[0][0])
    will_churn = "Yes user will most likely leave your establishment" if churn_probability > 0.5 else "No the user won't leave your establishment"
    
    logger.info(f"Prediction made: {will_churn} with probability {churn_probability:.4f}")
    
    return jsonify({
        "prediction": will_churn,
        "probability": float(churn_probability)
    })

def handle_batch_prediction(batch_features):
    """Handle batch prediction for multiple customers"""
    # Convert to numpy array
    features_array = np.array(batch_features)
    
    logger.info(f"Batch shape: {features_array.shape}")
    
    # Validate feature count
    if features_array.shape[1] != 11:
        return jsonify({
            "error": f"Invalid feature count. Expected 10, got {features_array.shape[1]}"
        }), 400
    
    # Make predictions for all samples
    predictions = model.predict(features_array)
    
    # Process results
    results = []
    churn_count = 0
    no_churn_count = 0
    
    for i, pred in enumerate(predictions):
        churn_probability = float(pred[0])
        will_churn = churn_probability > 0.5
        
        logger.info(f"Sample {i+1}: probability = {churn_probability:.4f}")
        
        if will_churn:
            churn_count += 1
            prediction_text = "Yes user will most likely leave your establishment"
        else:
            no_churn_count += 1
            prediction_text = "No the user won't leave your establishment"
        
        results.append({
            "sample_id": i + 1,
            "prediction": prediction_text,
            "probability": churn_probability,
            "will_churn": will_churn
        })
        
    
    # Calculate percentages
    total_samples = len(predictions)
    churn_percentage = (churn_count / total_samples) * 100
    no_churn_percentage = (no_churn_count / total_samples) * 100
    
    # Add in handle_batch_prediction function:
    
    logger.info(f"Batch prediction complete: {churn_count} will churn, {no_churn_count} won't churn")
    
    return jsonify({
        "batch_results": results,
        "summary": {
            "total_samples": total_samples,
            "churn_count": churn_count,
            "no_churn_count": no_churn_count,
            "churn_percentage": round(churn_percentage, 2),
            "no_churn_percentage": round(no_churn_percentage, 2)
        }
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint to get information about the model"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "input_shape": str(input_shape),
        "output_shape": str(model.output_shape),
        "expected_features": [
            "Credit Score", "Geography (Spain)", "Geography (Germany)", 
            "Gender (Male)", "Age", "Tenure", "Balance", 
            "Number of Products", "Has Credit Card", "Is Active Member",
            "Estimated Salary"
        ],
        "model_file": MODEL_PATH
    })

@app.route('/test-connection', methods=['GET'])
def test_connection():
    """Simple endpoint to test if the API is running"""
    return jsonify({"status": "API is running", "model_loaded": model is not None})

@app.route('/api/status')
def api_status():
    """Check if the API is running"""
    return jsonify({"status": "API is running", "model_loaded": model is not None})

# Create a directory for templates if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Move frontend.html to templates directory if it's not already there
if os.path.exists('frontend.html') and not os.path.exists('templates/frontend.html'):
    shutil.move('frontend.html', 'templates/frontend.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)