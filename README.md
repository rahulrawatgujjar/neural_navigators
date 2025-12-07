# Return Risk Prediction System

## Problem Statement

The objective of this project is to predict whether a customer is likely to **return a product after purchase** using machine learning.  
For example, if a customer buys a product worth ₹1000 (such as an RC car), the system predicts whether the customer will **return the product or keep it**.

The primary goals of this project are:
- Reduce product return rates  
- Improve customer satisfaction  
- Increase overall business profitability  

This is a **binary classification problem** where:
- `0` → Product will **not** be returned  
- `1` → Product **will** be returned  

---

## Dataset Description

The dataset consists of product, customer, and transaction-related features. The main features used for model training include:

- Product_Category  
- Product_Price  
- Order_Quantity  
- Discount_Applied  
- User_Age  
- User_Gender  
- User_Location  
- Payment_Method  
- Shipping_Method  
- Return_Status *(Target Variable)*  

### Data Preprocessing Includes:
- Target variable encoding (`Returned → 1`, `Not Returned → 0`)
- Removal of identifier and leakage columns (Order ID, User ID, Return Date, etc.)
- Categorical encoding using **Label Encoding & One-Hot Encoding**
- Numerical feature scaling using **StandardScaler**
- Feature alignment for real-time inference

---

## Label Definition

| Label | Meaning        |
|-------|----------------|
| 0     | Not Returned   |
| 1     | Returned       |

---

## Machine Learning Models Implemented

The following models were implemented and evaluated using **GridSearchCV with 5-Fold Cross Validation**:

1. Logistic Regression  
2. Support Vector Machine (SVM)  
3. Random Forest  
4. XGBoost  

After evaluation, the **best-performing model** was selected and saved as:

- `best_model.pkl`
- `scaler.pkl`

---

## Performance Evaluation Metrics

The models were evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

Additionally, the system uses **probability-based prediction (`predict_proba`)** to display the **confidence level of return risk**.

---

## System Architecture & Working

The system follows a **full-stack machine learning pipeline**:

### 1. Frontend (Next.js)
- User enters product and customer details.
- Displays:
  - Return prediction (Low Risk / High Risk)
  - Return probability
  - Visual probability bar

### 2. Backend (FastAPI)
- Receives input through a POST request.
- Applies the same preprocessing used during training.
- Loads:
  - `best_model.pkl`
  - `scaler.pkl`
- Generates:
  - Binary prediction (0 or 1)
  - Prediction probability

### 3. Machine Learning Model
- The trained model predicts whether a product will be returned or not based on input features.

---

## Real-Time Prediction Flow

1. User enters order details in the UI.  
2. Data is sent to the FastAPI backend using a **POST request**.  
3. Backend preprocesses the input features.  
4. Trained ML model predicts:
   - Return Status (0 or 1)
   - Return Probability  
5. Result is displayed on the UI in real time.

---

## Workflow Diagram

![Workflow](workflow.png)

---
