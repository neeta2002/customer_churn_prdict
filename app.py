from flask import Flask
import joblib 
app=Flask(__name__)
model=joblib.load("model/model.pkl")
@app.route("/")
def home():
    prediction=model.predict([[40, 1500, 3]])
    result="Customer Will Churn" if prediction[0]==1 else "Customer Will Not Churn"
    return f"customer predicition:{result}"

if __name__=="__main__":
    app.run(debug=True)