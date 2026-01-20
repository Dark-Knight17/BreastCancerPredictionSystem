from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load Model and Scaler
# Ensure these paths are correct relative to where you run app.py
# If your model is in a /model folder, change path to 'model/breast_cancer_model.pkl'
MODEL_PATH = './model/breast_cancer_model.pkl'
SCALER_PATH = './model/scaler.pkl'

# Helper function to load resources
def load_resources():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        print("Error: Model or Scaler file not found. Please run model_building code first.")
        return None, None

model, scaler = load_resources()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    
    if request.method == 'POST':
        if not model or not scaler:
            return render_template('index.html', prediction_text="Error: Model not loaded.")

        try:
            # 1. Get data from form
            radius = float(request.form['radius_mean'])
            texture = float(request.form['texture_mean'])
            perimeter = float(request.form['perimeter_mean'])
            area = float(request.form['area_mean'])
            smoothness = float(request.form['smoothness_mean'])

            # 2. Prepare data for model
            # Create a numpy array with the inputs
            input_data = np.array([[radius, texture, perimeter, area, smoothness]])
            
            # 3. Scale the data (using the same scaler from training)
            input_scaled = scaler.transform(input_data)

            # 4. Predict
            prediction = model.predict(input_scaled)
            
            # Sklearn Breast Cancer dataset default: 0 = Malignant, 1 = Benign
            if prediction[0] == 0:
                result = "Malignant"
                css_class = "malignant"
            else:
                result = "Benign"
                css_class = "benign"

            prediction_text = f"Prediction: Tumor is {result}"
            
            return render_template('index.html', prediction_text=prediction_text, result_class=css_class)

        except Exception as e:
            return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)