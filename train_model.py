
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("app_usage_dataset.csv")

features = [
    "daily_screen_hours",
    "app_opens_per_day",
    "social_media_hours",
    "night_usage_hours",
    "sleep_hours",
    "control_difficulty"
]

X = df[features]
y = df["risk_level"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Evaluate on held-out data
predictions = model.predict(X_test)

print("Accuracy:", round(
    accuracy_score(y_test, predictions), 3
))

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(
    y_test, predictions, labels=model.classes_
))

# Save model and expected feature order
joblib.dump(
    {"model": model, "features": features},
    "risk_model.pkl"
)

print("\nModel saved as risk_model.pkl")