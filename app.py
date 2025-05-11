from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import logging
import os

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
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.route('/')
def home():
    return "Customer Churn Prediction API is running. Send POST requests to /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500
        
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({"error": "Invalid input: 'features' field is required"}), 400
            
        features = np.array(data['features'])
        
        # Log the received features
        logger.info(f"Received features: {features}")
        logger.info(f"Feature count: {len(features)}")
        
        # Based on the frontend, expecting 10 features:
        # [creditScore, Spain, Germany, Male, age, tenure, balance, numOfProducts, hasCrCard, isActiveMember, estimatedSalary]
        if len(features) != 10:
            logger.warning(f"Unexpected feature count: {len(features)}. Expected 10 based on frontend.")
        
        # Reshape features for the model
        features = features.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # For binary classification (churn prediction is typically binary)
        churn_probability = float(prediction[0][0])
        will_churn = "Yes user will most likely leave your establishment" if churn_probability > 0.5 else "No the user won't leave your establishment"
        
        logger.info(f"Prediction made: {will_churn} with probability {churn_probability:.4f}")
        
        # Return the prediction in the format expected by the frontend
        return jsonify({
            "prediction": will_churn,
            "probability": float(churn_probability)
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)