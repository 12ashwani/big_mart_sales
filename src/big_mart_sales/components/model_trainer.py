import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.big_mart_sales.exception import CustomException
from src.big_mart_sales.logger import logging
from src.big_mart_sales.utils import save_object  # ✅ Updated save function

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'best_model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_and_evaluate_models(self, data):
        try:
            logging.info("🚀 Model Training and Evaluation Started")
            X = data.drop(columns=['Item_Outlet_Sales'], axis=1)
            y = data['Item_Outlet_Sales']

            # ✅ Splitting dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
            }

            model_scores = {}

            for name, model in models.items():
                logging.info(f"⚡ Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                model_scores[name] = r2
                logging.info(f"✅ {name} R2 Score: {r2:.4f}")

            # ✅ Selecting the best model
            best_model_name = max(model_scores, key=model_scores.get)
            best_model = models[best_model_name]

            logging.info(f"🏆 Best Model: {best_model_name} with R2 Score: {model_scores[best_model_name]:.4f}")

            # ✅ Save the best model using joblib
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"✅ Best model saved at {self.model_trainer_config.trained_model_file_path}")

            return best_model_name, model_scores[best_model_name]

        except Exception as e:
            logging.error("❌ Error in Model Training and Evaluation")
            raise CustomException(e, sys)
