from src.pipeline.ingestion_pipeline import IngestionPipeline

if __name__ == "__main__":
    # This targets the dataset inside your upload subfolder
    pipeline = IngestionPipeline(csv_path="upload_data_to_db/raw_dataset.csv")
    pipeline.execute_ingestion()