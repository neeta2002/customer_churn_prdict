import joblib 
model=joblib.load("model/model.pkl")

customer = [[40, 1500, 3]]

prediction = model.predict(customer)
if prediction[0] == 1:
    print("Customer will Churn")
else:
    print("Customer will Not Churn")