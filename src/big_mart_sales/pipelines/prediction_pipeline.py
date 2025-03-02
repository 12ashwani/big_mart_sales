import sys
import os
import pandas as pd
import numpy as np
import joblib
from src.big_mart_sales.exception import CustomException
from src.big_mart_sales.logger import logging
from src.big_mart_sales.utils import load_object
from src.big_mart_sales.components.data_transformation import DataTransformation

import joblib  # ✅ Import joblib instead of pickle

class PredictionPipeline:
    """Prediction Pipeline to make predictions on new data."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """Load the trained model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"❌ Model file not found: {self.model_path}")
            
            # ✅ Load model using joblib
            self.model = joblib.load(self.model_path)
            logging.info("✅ Model loaded successfully!")
            print("✅ Model loaded successfully!")  # Debugging
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            raise CustomException(e, sys)


    def predict(self, data):
        """
        Make predictions on new data.

        Args:
            data (str or pd.DataFrame): Path to the CSV file or DataFrame.

        Returns:
            np.ndarray: Predictions made by the model.
        """
        try:
            if self.model is None:
                self.load_model()

            logging.info("🚀 Processing input data...")

            # ✅ Ensure data is a DataFrame before transformation
            if isinstance(data, str):  
                data = pd.read_csv(data)  # Read CSV if path is provided
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError("❌ Invalid data format. Provide a CSV file path or a DataFrame.")

            processed_df = DataTransformation().transform_data(data)

            predictions = self.model.predict(processed_df)
            return predictions.astype(float)

        except Exception as e:
            logging.error(f"❌ Error during prediction: {e}")
            raise CustomException(e, sys)
