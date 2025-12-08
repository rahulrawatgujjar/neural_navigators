import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from joblib import dump

# -----------------------------
# LOAD DATA
# -----------------------------
file_path = "/home/rahulrawatr320/Desktop/hcl/main/ecommerce_returns_enhanced.csv"
df = pd.read_csv(file_path)

# -----------------------------
# TARGET ENCODING
# -----------------------------
df['Return_Status'] = df['Return_Status'].apply(lambda x: 1 if x == "Returned" else 0)

# -----------------------------
# DROP LEAKAGE COLUMNS
# -----------------------------
cols_to_drop = ['Order_ID', 'Product_ID', 'User_ID', 'Return_Date', 'Return_Reason', 'Days_to_Return']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# -----------------------------
# PROCESS DATE
# -----------------------------
df['Order_Date'] = pd.to_datetime(df['Order_Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Order_Date'])
df['Order_Month'] = df['Order_Date'].dt.month
df['Order_Weekday'] = df['Order_Date'].dt.weekday
df = df.drop(columns=['Order_Date'])

# -----------------------------
# ENCODE CATEGORICAL
# -----------------------------
le = LabelEncoder()
df['User_Location'] = le.fit_transform(df['User_Location'].astype(str))

cat_cols = ['Product_Category', 'User_Gender', 'Payment_Method', 'Shipping_Method']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# -----------------------------
# SCALE NUMERICAL
# -----------------------------
num_cols = ['Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X = df.drop(columns=['Return_Status'])
y = df['Return_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# FINAL RANDOM FOREST MODEL
# -----------------------------
rf_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    class_weight="balanced",
    random_state=42
)

rf_final.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = rf_final.predict(X_test)
y_proba = rf_final.predict_proba(X_test)[:, 1]

print("\n--- FINAL RANDOM FOREST PERFORMANCE ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_proba))

# -----------------------------
# SAVE MODEL & SCALER
# -----------------------------
dump(rf_final, "random_forest_model.pkl")
dump(scaler, "scaler.pkl")

print("\n✅ Model saved as: random_forest_model.pkl")
print("✅ Scaler saved as: scaler.pkl")
