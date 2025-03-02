from src.big_mart_sales.exception import CustomException
import sys
import os
import pandas as pd
from src.big_mart_sales.utils import load_object
from src.big_mart_sales.logger import logging
from src.big_mart_sales.components.data_ingestion import DataIngestion
# from data_ingestion import DataIngestion  # Import the class from your script

# Define source details where the raw data is stored
source_details = {
    "path": "Data/Train.csv"  # Ensure this path is correct
}

# Define ingestion configuration
class IngestionConfig:
    raw_data_path = "Data_New/raw_data.csv"
    train_data_path = "Data_New/train_data.csv"
    test_data_path = "Data_New/test_data.csv"

ingestion_config = IngestionConfig()

# Initialize and call data ingestion
data_ingestion = DataIngestion(source_type="csv", source_details=source_details, ingestion_config=ingestion_config)
data_ingestion.initiate_data_ingestion()


# DataIngestion().initiate_data_ingestion(source_type='csv', target_path='Data/Train.csv')    
# Compare this snippet from src/big_mart_sales/logger.py:

