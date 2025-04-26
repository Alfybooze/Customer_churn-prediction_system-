from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('bank_churn_model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [[
        data['creditScore'],
        data['geography'],  # Ensure encoding is done beforehand
        data['gender'],     # Ensure encoding is done beforehand
        data['age'],
        data['tenure'],
        data['balance'],
        data['numOfProducts'],
        data['hasCrCard'],
        data['isActiveMember'],
        data['estimatedSalary']
    ]]
    prediction = model.predict(features)
    return jsonify({'exited': bool(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000)
