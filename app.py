from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

app = FastAPI()

# Input schema
class OrderData(BaseModel):
    Product_Category: str
    Product_Price: float
    Order_Quantity: int
    Discount_Applied: float
    User_Age: int
    User_Gender: str
    User_Location: str
    Payment_Method: str
    Shipping_Method: str

@app.post("/predict")
def predict(data: OrderData):
    
    input_df = pd.DataFrame([{
        "Product_Category": data.Product_Category,
        "Product_Price": data.Product_Price,
        "Order_Quantity": data.Order_Quantity,
        "Discount_Applied": data.Discount_Applied,
        "User_Age": data.User_Age,
        "User_Gender": data.User_Gender,
        "User_Location": data.User_Location,
        "Payment_Method": data.Payment_Method,
        "Shipping_Method": data.Shipping_Method
    }])

    X = preprocessor.transform(input_df)
    prediction = model.predict(X)[0]

    return {"prediction": int(prediction)}
