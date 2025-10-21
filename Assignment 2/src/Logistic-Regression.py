from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the breast cancer dataset
cancer_data = load_breast_cancer()

# Get features (X) and target (y)
# All features are used for prediction
X = cancer_data.data
y = cancer_data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(solver='liblinear', random_state=42)

# Fit the model to our training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("Breast Cancer Dataset - Logistic Regression Results")
print("=" * 50)
print(f"Dataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Target classes: {cancer_data.target_names}")
print(f"Feature names: {cancer_data.feature_names[:5]}... (showing first 5)")
print()

print("Model Performance:")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print()

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer_data.target_names))

# Show some sample predictions
print("Sample Predictions:")
print("-" * 30)
for i in range(10):
    actual = cancer_data.target_names[y_test[i]]
    predicted = cancer_data.target_names[y_pred[i]]
    confidence = model.predict_proba(X_test[i:i+1])[0]
    print(f"Sample {i+1}: Actual={actual}, Predicted={predicted}, "
          f"Confidence=[{confidence[0]:.3f}, {confidence[1]:.3f}]")
