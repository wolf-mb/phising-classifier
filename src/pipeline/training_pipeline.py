import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.data_access.phishing_data import PhishingData
from src.preprocessing.data_transformer import DataTransformer

class ModelTrainingPipeline:
    def __init__(self, model_output_path: str = "model.pkl"):
        self.model_output_path = model_output_path
        self.data_access = PhishingData()
        self.transformer = DataTransformer()

    def run_training_workflow(self):
        print("=== Starting ADVANCED ML Training Pipeline ===")
        df = self.data_access.export_collection_as_dataframe()
        df = self.transformer.clean_data(df)
        
        # THE MAXIMUM STRUCTURAL SELECTION
        # We are extracting 12 live features to maximize predictive probability
        # UCI Indices mapped:
        # 0(IP), 1(Len), 2(Shortener), 3(@), 4(//), 5(-), 6(Subdomains), 
        # 7(SSL), 10(Port), 11(HTTPS token), 17(Abnormal URL), 18(Redirect)
        selected_indices = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 17, 18]
        
        target_col = df.columns[-1]
        X = df.iloc[:, selected_indices]
        y = df[target_col]
        
        print(f"Selected {X.shape[1]} deep-lexical features for high-probability training.")
        
        # Train & Evaluate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Increased estimators to 150 for deeper probability calculation across the 12 features
        model = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42)
        model.fit(X_train, y_train)
        
        joblib.dump(model, self.model_output_path)
        print(f"🚀 Success! 12-Feature Model trained. Accuracy: {model.score(X_test, y_test)*100:.2f}%")