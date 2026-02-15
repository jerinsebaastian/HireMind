import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load new multi-feature dataset
df = pd.read_csv("data/multifeature_training_data.csv")

# Features (must match Flask order exactly)
X = df[[
    "total_gap",
    "missing_count",
    "weak_count",
    "moderate_count",
    "strong_count",
    "avg_skill_level",
    "high_importance_gap"
]]

y = df["readiness"]

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and encoder
joblib.dump(model, "ml/readiness_model.pkl")
joblib.dump(encoder, "ml/label_encoder.pkl")

print("\nMulti-feature model saved successfully.")