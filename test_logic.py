from src.data_access.phishing_data import PhishingData
import joblib

# Load your model
model = joblib.load("model.pkl")

# Fetch 100 random rows from your cloud database
data = PhishingData().export_collection_as_dataframe()
sample = data.sample(100)

# Run batch prediction
X = sample.drop(columns=['Result'])
y = sample['Result']
predictions = model.predict(X)

# Calculate Accuracy
from sklearn.metrics import accuracy_score
print(f"Batch Accuracy on cloud data: {accuracy_score(y, predictions) * 100:.2f}%")