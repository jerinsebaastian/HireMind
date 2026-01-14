import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load training data
df = pd.read_csv("data/readiness_training_data.csv")

X = df[['gap_score']]
y = df['readiness']

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train model
model = LogisticRegression()
model.fit(X, y_encoded)

# Save model and encoder
joblib.dump(model, "ml/readiness_model.pkl")
joblib.dump(encoder, "ml/label_encoder.pkl")

print("Model and Label Encoder saved successfully.")
