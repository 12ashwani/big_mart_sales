import os
import sys
from src.big_mart_sales.exception import CustomException
from src.big_mart_sales.logger import logging

import joblib
import os

def save_object(file_path, obj):
    """Save an object using joblib."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as f:
            joblib.dump(obj, f)
        
        print(f"✅ Model saved successfully at {file_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        raise CustomException(e, sys)


def load_object(file_path):
    """Load a Python object using pickle."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)

    except Exception as e:
        logging.error(f"❌ Error loading object from {file_path}: {e}")
        raise CustomException(e, sys)

def get_project_root() -> str:
    """Get the root directory of the project."""
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0])))

def get_data_dir() -> str:
    """Get the data directory path and create it if it doesn't exist."""
    data_dir = os.path.join(get_project_root(), "Data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir
