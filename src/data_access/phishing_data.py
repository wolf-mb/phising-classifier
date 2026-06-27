import pandas as pd
from pymongo import MongoClient
from src.configuration.mongo_db_config import MongoDBConfig

class PhishingData:
    def __init__(self):
        self.config = MongoDBConfig()

    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """Fetches the cloud collection dataset and returns it as a clean Pandas DataFrame."""
        try:
            # Connect to Atlas
            client = MongoClient(self.config.MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
            db = client[self.config.DATABASE_NAME]
            collection = db[self.config.COLLECTION_NAME]
            
            # Pull documents out of the cloud
            raw_documents = list(collection.find())
            df = pd.DataFrame(raw_documents)
            
            # Clean up the internal MongoDB '_id' column so it doesn't break the ML model
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
                
            return df
        except Exception as e:
            raise Exception(f"Failed to export data from MongoDB Atlas: {e}")