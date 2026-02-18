from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("crash_detection_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data], columns=[
        "maxAcc", "maxGyro", "accVar", "gyroVar"
    ])

    probability = model.predict_proba(df)[0][1]

    # Print probability for debugging
    print("Predicted probability:", probability)
    print("Input features:", df)
    

    return jsonify({
        "crash_probability": float(probability)
    })

if __name__ == "__main__":
    app.run()

