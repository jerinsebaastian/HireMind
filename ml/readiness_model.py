import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("data/readiness_training_data.csv")

X = df[['gap_score']]
y = df['readiness']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = LogisticRegression()
model.fit(X, y_encoded)

joblib.dump(model, "ml/readiness_model.pkl")
joblib.dump(encoder, "ml/label_encoder.pkl")

print("ML model trained with large dataset.")
