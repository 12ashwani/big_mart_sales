import os
import sys
import logging
import pandas as pd
import pymysql
import requests
from pymongo import MongoClient
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, source_type, source_details, ingestion_config):
        self.source_type = source_type
        self.source_details = source_details
        self.ingestion_config = ingestion_config
        
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    def fetch_data(self):
        """
        Fetch data from the specified source type with improved error handling.
        """
        try:
            if self.source_type == "csv":
                data_path = self.source_details.get("path")
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"CSV file not found: {data_path}")
                df = pd.read_csv(data_path)
            
            elif self.source_type == "mysql":
                with pymysql.connect(**self.source_details) as conn:
                    query = self.source_details.get("query")
                    df = pd.read_sql(query, conn)
            
            elif self.source_type == "nosql":
                client = MongoClient(self.source_details.get("uri"))
                db = client[self.source_details.get("database")]
                collection = db[self.source_details.get("collection")]
                df = pd.DataFrame(list(collection.find()))
                client.close()
            
            elif self.source_type == "api":
                url = self.source_details.get("url")
                if not url:
                    raise ValueError("API URL is required")
                response = requests.get(url)
                response.raise_for_status()
                df = pd.DataFrame(response.json())
            
            elif self.source_type == "web_scraping":
                url = self.source_details.get("url")
                parse_function = self.source_details.get("parse_function")
                if not url or not parse_function:
                    raise ValueError("URL and parse_function are required for web scraping")
                response = requests.get(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                data = parse_function(soup)
                df = pd.DataFrame(data)
            
            else:
                raise ValueError("Unsupported data source type")

            logging.info(f"✅ Data successfully fetched from {self.source_type}")
            return df
        
        except Exception as e:
            logging.error(f"❌ Error occurred while fetching data: {str(e)}")
            raise Exception(f"Data fetching failed: {e}")

    def initiate_data_ingestion(self):
        """
        Reads data, splits it into train-test sets, and saves them as CSV files.
        """
        try:
            logging.info("🚀 Starting data ingestion process...")
            df = self.fetch_data()
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("✅ Data ingestion process completed successfully.")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            logging.error(f"❌ Error occurred during data ingestion: {str(e)}")
            raise Exception(f"Data ingestion failed: {e}")
