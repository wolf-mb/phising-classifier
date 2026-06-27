import sys
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.pipeline.training_pipeline import ModelTrainingPipeline

def main():
    print("Select an enterprise workflow stage to execute:")
    print("1. Run Data Ingestion Pipeline (Local CSV -> MongoDB Atlas Cloud)")
    print("2. Run Model Training Pipeline (MongoDB Atlas Cloud -> Train ML Model)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        pipeline = IngestionPipeline(csv_path="upload_data_to_db/raw_dataset.csv")
        pipeline.execute_ingestion()
    elif choice == "2":
        pipeline = ModelTrainingPipeline(model_output_path="model.pkl")
        pipeline.run_training_workflow()
    else:
        print("Invalid choice. Exiting pipeline.")

if __name__ == "__main__":
    main()