import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Configuration Constants
MONGO_URI = "mongodb://mahesh:mahesh2026@ac-y8mezgf-shard-00-00.gvuyc40.mongodb.net:27017,ac-y8mezgf-shard-00-01.gvuyc40.mongodb.net:27017,ac-y8mezgf-shard-00-02.gvuyc40.mongodb.net:27017/?ssl=true&replicaSet=atlas-y2jg8y-shard-0&authSource=admin"
DATABASE_NAME = "phishing"
COLLECTION_NAME = "urls_dataset"
MODEL_OUTPUT_PATH = "phishing_model.pkl"

def load_data_from_cloud():
    """Fetches records from MongoDB Atlas and converts them to a Pandas DataFrame."""
    print("=== Phase 3: Fetching Clean Dataset from Cloud ===")
    client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Read all documents out of MongoDB
    raw_data = list(collection.find())
    df = pd.DataFrame(raw_data)
    
    # Drop the internal MongoDB '_id' column so it doesn't interfere with training
    if '_id' in df.columns:
        df = df.drop(columns=['_id'])
        
    print(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} features from the cloud.")
    return df

def train_phishing_classifier():
    # 1. Load data
    df = load_data_from_cloud()
    
    # 2. Separate Features (X) and Label (y)
    # Assumes the target classification column is named 'Result' or the last column
    # If your CSV's label column has a different name, swap 'Result' out below
    target_column = 'Result' if 'Result' in df.columns else df.columns[-1]
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Target column detected: '{target_column}'")
    
    # 3. Train-Test Split (80% Training, 20% Testing for validation validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data Split Complete: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")
    
    # 4. Initialize and Train the Model
    print("\n=== Phase 4: Training Random Forest Classifier ===")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete!")
    
    # 5. Evaluate Performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*40)
    print(f"🎯 MODEL ACCURACY SCORE: {accuracy * 100:.2f}%")
    print("="*40)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 6. Serialize and Save Model Artifact for Deployment
    print(f"Saving trained model artifact to: {MODEL_OUTPUT_PATH}")
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print("🚀 Model successfully serialized and ready for deployment!")

if __name__ == "__main__":
    train_phishing_classifier()