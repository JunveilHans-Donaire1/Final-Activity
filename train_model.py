"""
Create a local model.pkl for Activity 3.

This is only needed because you do not have the model from Activity 1.
It trains a simple example classifier using synthetic study-hours examples.
Replace the sample training data with your instructor's actual dataset/model
if one is available. No internet is used.
"""
import os
import joblib
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# Synthetic demonstration data: [study_hours] -> 0 = FAIL, 1 = PASS.
# This simple rule-based example treats 5+ hours as PASS.
X = [[0], [1], [2], [3], [4], [4.5], [5], [5.5], [6], [7], [8], [10], [12], [16], [24]]
y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, MODEL_PATH)

print(f"Created model: {MODEL_PATH}")
print("Demo model rule: below 5 study hours -> FAIL; 5 or more -> PASS.")
print("This is a demonstration model, not a scientifically validated passing threshold.")
