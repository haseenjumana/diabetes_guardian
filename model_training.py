import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

# Correct URL for the dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
data = pd.read_csv(url)

# Split features and label
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Build pipeline
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy='mean')),  # Handle missing values with mean imputation
    ("scaler", StandardScaler()),                  # Normalize data
    ("classifier", LogisticRegression(max_iter=1000))  # Model with increased max iterations
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate model
accuracy = pipeline.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Save to file
model_path = "diabetes_model.pkl"
joblib.dump(pipeline, model_path)

print(f"✅ Model saved as {model_path}")
