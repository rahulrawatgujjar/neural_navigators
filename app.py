from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------
# LOAD BEST MODEL & SCALER
# ---------------------------------------------------------
model = joblib.load("/home/rahulrawatr320/Desktop/hcl/return-risk-ui/best_model.pkl")
scaler = joblib.load("/home/rahulrawatr320/Desktop/hcl/return-risk-ui/scaler.pkl")

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# INPUT SCHEMA
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------------
@app.post("/predict")
def predict(data: OrderData):
    
    # Step 1: Create raw input DataFrame
    input_df = pd.DataFrame([{
        "Product_Category": data.Product_Category,
        "Product_Price": data.Product_Price,
        "Order_Quantity": data.Order_Quantity,
        "User_Age": data.User_Age,
        "Discount_Applied": data.Discount_Applied,
        "User_Gender": data.User_Gender,
        "User_Location": data.User_Location,
        "Payment_Method": data.Payment_Method,
        "Shipping_Method": data.Shipping_Method
    }])

    # -----------------------------------------------------
    # Step 2: Encode User_Location (same column treated as numeric in training)
    # Since LabelEncoder was not saved, we convert to category codes
    # -----------------------------------------------------
    input_df["User_Location"] = input_df["User_Location"].astype("category").cat.codes

    # -----------------------------------------------------
    # Step 3: One-Hot Encode Other Categorical Features
    # (Same as training: Product_Category, User_Gender, Payment_Method, Shipping_Method)
    # -----------------------------------------------------
    input_df = pd.get_dummies(
        input_df,
        columns=["Product_Category", "User_Gender", "Payment_Method", "Shipping_Method"],
        drop_first=True
    )

    # -----------------------------------------------------
    # Step 4: Align with Training Features
    # -----------------------------------------------------
    model_features = model.feature_names_in_
    input_df = input_df.reindex(columns=model_features, fill_value=0)

    # -----------------------------------------------------
    # Step 5: Scale Numerical Features
    # -----------------------------------------------------
    num_cols = ["Product_Price", "Order_Quantity", "User_Age", "Discount_Applied"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # -----------------------------------------------------
    # Step 6: Prediction
    # -----------------------------------------------------
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
