import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from src.big_mart_sales.exception import CustomException
from src.big_mart_sales.logger import logging

class DataTransformation:
    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

    def clean_data(self, df):
        """Handles missing values and corrects data inconsistencies."""
        try:
            logging.info("Performing data cleaning...")
            df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
            df['Outlet_Size'] = df['Outlet_Size'].fillna('Small')
            df.loc[df['Item_Visibility'] == 0, 'Item_Visibility'] = df['Item_Visibility'].mean()
            df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def create_features(self, df):
        """Creates new features to improve predictive power."""
        try:
            logging.info("Creating new features...")
            df['Outlet_Age'] = 2025 - df['Outlet_Establishment_Year']
            df['estimated_sales_per_year'] = (df['Item_MRP'] * (1 - df['Item_Visibility'])) / (df['Outlet_Age'] + 1)
            df['Price_Per_Weight'] = df['Item_MRP'] / (df['Item_Weight'] + 1)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def encode_and_standardize(self, df):
        """Encodes categorical variables and standardizes numerical features."""
        try:
            logging.info("Encoding and standardizing data...")
            categorical_features = df.select_dtypes(include='object').columns
            numerical_features = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age"]

            # Separate features for encoding
            one_hot_features, label_encode_features = [], []
            for col in categorical_features:
                (label_encode_features if df[col].nunique() < 5 else one_hot_features).append(col)

            # OneHotEncoding
            if one_hot_features:
                one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_array = one_hot_encoder.fit_transform(df[one_hot_features])
                encoded_df = pd.DataFrame(encoded_array, columns=one_hot_encoder.get_feature_names_out(one_hot_features), index=df.index)
                df = pd.concat([df, encoded_df], axis=1)
                df.drop(columns=one_hot_features, inplace=True)

            # LabelEncoding
            label_encoder = LabelEncoder()
            for col in label_encode_features:
                df[col] = label_encoder.fit_transform(df[col].fillna("Unknown"))

            # Standardization
            scaler = StandardScaler()
            df[numerical_features] = scaler.fit_transform(df[numerical_features])

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def transform_data(self, data):
        """
        Applies transformation pipeline on dataset.
        - If `data` is a file path (str), loads CSV.
        - If `data` is a DataFrame, processes directly.
        """
        try:
            logging.info("Loading dataset...")

            # 🔹 Check if input is a CSV file path or DataFrame
            if isinstance(data, str):
                df = pd.read_csv(data)  # Load CSV file
                is_training = True  # Assume training if reading from CSV
            elif isinstance(data, pd.DataFrame):
                df = data.copy()  # Use in-memory DataFrame
                is_training = "Item_Outlet_Sales" in df.columns  # Check if it's training data
            else:
                raise ValueError("Invalid data input: Must be a CSV file path or Pandas DataFrame.")

            target_column = "Item_Outlet_Sales"
            drop_columns = ['Item_Identifier']  # Always drop Item_Identifier

            # 🔹 Handle target variable dynamically
            y = None
            if is_training:
                drop_columns.append(target_column)
                y = df[target_column]  # Extract target variable

            # 🔹 Drop columns if they exist
            X = df.drop(columns=[col for col in drop_columns if col in df.columns])

            logging.info("Applying data transformations...")
            cleaned_df = self.clean_data(X)
            feature_df = self.create_features(cleaned_df)
            processed_df = self.encode_and_standardize(feature_df)

            # 🔹 Return processed DataFrame
            return pd.concat([processed_df, y], axis=1) if y is not None else processed_df

        except Exception as e:
            raise CustomException(e, sys)
