import os
import pandas as pd
from pymongo import MongoClient
from src.configuration.mongo_db_config import MongoDBConfig

# Ensure this name matches exactly: IngestionPipeline
class IngestionPipeline:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.config = MongoDBConfig()

    def execute_ingestion(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Missing target file at {self.csv_path}")
            
        client = MongoClient(self.config.MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
        db = client[self.config.DATABASE_NAME]
        collection = db[self.config.COLLECTION_NAME]
        
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.strip()
        records = df.to_dict(orient="records")
        
        collection.delete_many({})
        collection.insert_many(records)
        print(f" [Ingestion] Cleanly synced {len(records)} records to Atlas Cloud.")