import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Paths
DATA_PATH = os.path.join("data", "winequality-red.csv")
MODEL_PATH = os.path.join("model", "wine_model.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH, sep=None, engine='python')
X = df.drop("quality", axis=1)
y = df["quality"]

# Optional: Convert to 3 categories for better performance
# y = y.apply(lambda q: 0 if q <= 4 else 1 if q <= 6 else 2)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")
