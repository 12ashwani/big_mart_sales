import os
import sys
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from src.big_mart_sales.exception import CustomException
from src.big_mart_sales.logger import logging
from src.big_mart_sales.pipelines.prediction_pipeline import PredictionPipeline
from src.big_mart_sales.components.data_transformation import DataTransformation

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Define directory for saving predictions
PREDICTION_DIR = "artifacts"
os.makedirs(PREDICTION_DIR, exist_ok=True)  # Ensure the directory exists

# ✅ Load Prediction Pipeline
MODEL_PATH = "artifacts/best_model.pkl"
prediction_pipeline = PredictionPipeline(MODEL_PATH)

@app.route('/')
def home():
    return jsonify({"message": "Welcome to Big Mart Sales Prediction API. Upload a CSV file to get predictions."})

@app.route('/predict', methods=['POST'])
def predict_sales():
    try:
        logging.info("🔮 Prediction Started")
        
        # ✅ Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded!"}), 400
        
        file = request.files['file']
        df = pd.read_csv(file)
        logging.info(f"✅ Data loaded successfully. Shape: {df.shape}")

        # ✅ Check if Model Exists
        if not os.path.exists(MODEL_PATH):
            logging.error("❌ Model file not found!")
            return jsonify({"error": "Model file not found!"}), 500

        # ✅ Transform Data
        df_transformed = DataTransformation().transform_data(df)
        logging.info(f"✅ Data transformed successfully. Shape: {df_transformed.shape}")
        
        if df_transformed.empty:
            return jsonify({"error": "Transformed DataFrame is empty!"}), 400

        # ✅ Make Predictions
        predictions = prediction_pipeline.predict(df_transformed)
        logging.info("✅ Predictions made successfully")

        # ✅ Create Output DataFrame
        submission_df = pd.DataFrame({
            'Item_Identifier': df['Item_Identifier'],
            'Outlet_Identifier': df['Outlet_Identifier'],
            'Item_Outlet_Sales': predictions.astype(float)
        })

        # ✅ Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_file = os.path.join(PREDICTION_DIR, f"predictions_{timestamp}.csv")

        # ✅ Save Predictions
        submission_df.to_csv(prediction_file, index=False)
        logging.info(f"✅ Predictions saved at {prediction_file}")

        # ✅ Return File for Download
        return send_file(prediction_file, as_attachment=True)

    except Exception as e:
        logging.error(f"❌ Prediction failed: {str(e)}")
        return jsonify({"error": str(CustomException(e, sys))}), 500

if __name__ == '__main__':
    app.run(debug=True)
