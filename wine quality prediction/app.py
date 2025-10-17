from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("model", "wine_model.pkl")
model = joblib.load(MODEL_PATH)

FEATURE_NAMES = [
    'fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
    'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from form
        features = [float(request.form[f]) for f in [
            'fixed_acidity','volatile_acidity','citric_acid','residual_sugar','chlorides',
            'free_sulfur_dioxide','total_sulfur_dioxide','density','pH','sulphates','alcohol'
        ]]

        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        prediction = model.predict(input_df)[0]

        # Feature importance chart
        importances = model.feature_importances_
        plt.figure(figsize=(8,5))
        plt.barh(FEATURE_NAMES, importances, color='#4CAF50')
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()

        # Save plot to PNG image in memory
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        chart_base64 = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        return render_template('result.html', prediction=prediction, chart_base64=chart_base64)

if __name__ == '__main__':
    app.run(debug=True)
