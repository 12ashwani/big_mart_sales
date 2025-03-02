from src.big_mart_sales.exception import CustomException
import sys
import os
from src.big_mart_sales.components.data_transformation import DataTransformation
from src.big_mart_sales.components.model_trainer import ModelTrainer
from src.big_mart_sales.utils import save_object
from src.big_mart_sales.logger import logging
from src.big_mart_sales.pipelines.prediction_pipeline import PredictionPipeline

data_path="Data/train.csv"

data=DataTransformation().transform_data(data_path)
ModelTrainer().train_and_evaluate_models(data)
# PredictionPipeline().predict(data)
# model_path="model.pkl"