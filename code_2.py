import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("/home/rahulrawatr320/Desktop/hcl/ecommerce_returns_synthetic_data.csv")

# -----------------------------
# TARGET ENCODING
# -----------------------------
df['Return_Status'] = df['Return_Status'].map({
    'Not Returned': 0,
    'Returned': 1
})

# -----------------------------
# SELECT FINAL FEATURES
# -----------------------------
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

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# FEATURE TYPES
# -----------------------------
num_features = ['Product_Price', 'User_Age']
cat_features = ['Product_Category', 'User_Location',
                'Payment_Method', 'Shipping_Method']

# -----------------------------
# PREPROCESSING PIPELINE
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ]
)

X_train_final = preprocessor.fit_transform(X_train)
X_test_final = preprocessor.transform(X_test)

# -----------------------------
# METRIC FUNCTION
# -----------------------------
def evaluate_model(name, y_test, y_pred, y_prob):
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall   :", recall_score(y_test, y_pred))
    print("F1-Score :", f1_score(y_test, y_pred))
    print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

# =====================================================
# 1. LOGISTIC REGRESSION
# =====================================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_final, y_train)

y_pred_lr = lr.predict(X_test_final)
y_prob_lr = lr.predict_proba(X_test_final)[:,1]

evaluate_model("Logistic Regression", y_test, y_pred_lr, y_prob_lr)

# =====================================================
# 2. SUPPORT VECTOR MACHINE
# =====================================================
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train_final, y_train)

y_pred_svm = svm.predict(X_test_final)
y_prob_svm = svm.predict_proba(X_test_final)[:,1]

evaluate_model("SVM", y_test, y_pred_svm, y_prob_svm)

# =====================================================
# 3. RANDOM FOREST
# =====================================================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_final, y_train)

y_pred_rf = rf.predict(X_test_final)
y_prob_rf = rf.predict_proba(X_test_final)[:,1]

evaluate_model("Random Forest", y_test, y_pred_rf, y_prob_rf)

# =====================================================
# 4. XGBOOST
# =====================================================
xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)
xgb.fit(X_train_final, y_train)

y_pred_xgb = xgb.predict(X_test_final)
y_prob_xgb = xgb.predict_proba(X_test_final)[:,1]

evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb)

# =====================================================
# 5. ARTIFICIAL NEURAL NETWORK (ANN)
# =====================================================

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_final.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    X_train_final.toarray(),
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test_final.toarray(), y_test),
    verbose=1
)

y_prob_ann = model.predict(X_test_final.toarray()).ravel()
y_pred_ann = (y_prob_ann > 0.5).astype(int)

evaluate_model("ANN", y_test, y_pred_ann, y_prob_ann)
