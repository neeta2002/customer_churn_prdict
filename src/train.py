import numpy as np
from sklearn.tree import DecisionTreeClassifier 
import joblib
import os 
# [Age, Monthly Charges, Tenure]
X = np.array([
    [25, 500, 12],
    [30, 700, 24],
    [35, 1200, 6],
    [40, 1500, 3],
    [28, 600, 36],
    [45, 2000, 2],
    [50, 1800, 1],
    [32, 800, 30]
])

# Target:
# 0 = No Churn
# 1 = Churn
y = np.array([
    0,
    0,
    1,
    1,
    0,
    1,
    1,
    0
])
model=DecisionTreeClassifier()
model.fit(X,y)

os.makedirs("model",exist_ok=True)
joblib.dump(model,"model/model.pkl")
print("Model Trained Successfully")