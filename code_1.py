import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("/home/rahulrawatr320/Desktop/hcl/ecommerce_returns_synthetic_data.csv")

# -----------------------
# Encode Target Variable
# -----------------------
df['Return_Status'] = df['Return_Status'].map({
    'Not Returned': 0,
    'Returned': 1
})

# -----------------------
# Select Final Features
# -----------------------
X = df[
    ['Product_Category',
     'Product_Price',
     'Order_Quantity',
     'Discount_Applied',
     'User_Age',
     'User_Gender',
     'User_Location',
     'Payment_Method',
     'Shipping_Method']
]

y = df['Return_Status']

# -----------------------
# Train Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Define Feature Types
# -----------------------
num_features = ['Product_Price', 'Order_Quantity', 'Discount_Applied', 'User_Age']
cat_features = ['Product_Category', 'User_Gender', 'User_Location',
                'Payment_Method', 'Shipping_Method']

# -----------------------
# Preprocessing
# -----------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),   # Needed for LR, SVM, ANN
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

print(X_train)
print(y_test)

# -----------------------
# Final Preprocessed Data
# -----------------------
X_train_final = preprocessor.fit_transform(X_train)
X_test_final = preprocessor.transform(X_test)


# print(X_test_final)