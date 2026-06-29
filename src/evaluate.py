import joblib
import numpy as np
from sklearn.metrics import accuracy_score
model = joblib.load("models/model.pkl")
X_test = np.array([
    [25, 500, 12],
    [40, 1500, 3],
    [30, 700, 24],
    [50, 1800, 1]
])
y_test = np.array([0, 1, 0, 1])
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy * 100, "%")
if accuracy < 0.80:
    raise Exception("Model accuracy is below 80%")

print("Model evaluation successful.")
