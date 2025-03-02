import os
import sys
import pandas as pd
from datetime import datetime
from src.big_mart_sales.exception import CustomException
from src.big_mart_sales.logger import logging
from src.big_mart_sales.pipelines.prediction_pipeline import PredictionPipeline

# ✅ Define the artifacts directory for saving predictions
PREDICTION_DIR = "artifacts"
os.makedirs(PREDICTION_DIR, exist_ok=True)  # Ensure the directory exists

if __name__ == "__main__":
    try:
        model_path = "artifacts/best_model.pkl"
        data_path = "Data/test.csv"

        prediction_pipeline = PredictionPipeline(model_path)

        # ✅ Load test data
        sample_data = pd.read_csv(data_path)

        # ✅ Keep a copy before dropping unnecessary columns
        sample_data_copy = sample_data.copy()

        # ✅ Make predictions
        predictions = prediction_pipeline.predict(sample_data)

        # ✅ Create a submission dataframe
        submission_df = pd.DataFrame({
            'Item_Identifier': sample_data_copy['Item_Identifier'],  
            'Outlet_Identifier': sample_data_copy['Outlet_Identifier'],  
            'Item_Outlet_Sales': predictions  
        })

        # ✅ Ensure correct output type
        submission_df['Item_Outlet_Sales'] = submission_df['Item_Outlet_Sales'].astype(float)

        print("🔮 Predicted Sales:\n", submission_df.head())

        # ✅ Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_file = os.path.join(PREDICTION_DIR, f"predictions_{timestamp}.csv")

        # ✅ Save predictions to CSV
        submission_df.to_csv(prediction_file, index=False)
        logging.info(f"✅ Predictions saved successfully at {prediction_file}")

    except Exception as e:
        logging.error("❌ Prediction failed")
        print(e)
        raise CustomException(e, sys)
