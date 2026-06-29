import joblib
def test_customer_churn():
    model=joblib.load("model/model.pkl")
    prediction=model.predict([[50, 1800, 1]])
    assert prediction[0]==1