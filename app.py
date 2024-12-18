from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Load the trained model, scaler, and encoders
model = joblib.load('student_admission_model.pkl')  # Trained model
scaler = joblib.load('scaler.pkl')  # Saved StandardScaler object

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the request
        data = request.json
        features = data.get('features', [])

        if not features or len(features) != 5:
            return jsonify({"error": "Invalid input. Please provide all required features."}), 400

        # Extract and preprocess the input features
        age = int(features[0])
        gender = int(features[1])  # Ensure input is an integer
        admission_test_score = float(features[2])
        high_school_percentage = float(features[3])
        city = int(features[4])  # Ensure input is an integer

        # Scale numerical features
        input_data = pd.DataFrame([[admission_test_score, high_school_percentage, age]],
                                  columns=['Admission Test Score', 'High School Percentage', 'Age'])
        input_features_scaled = scaler.transform(input_data)

        print("Input features (scaled):", input_features_scaled, flush=True)

        # Combine scaled and encoded features in the correct order
        input_features = np.array([
            [input_features_scaled[0][2], gender, input_features_scaled[0][0], input_features_scaled[0][1], city]
        ])

        print("Final input for prediction:", input_features, flush=True)

        # Make prediction
        prediction = model.predict(input_features)

        # Return prediction result
        return jsonify({"prediction": int(prediction[0])})

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)