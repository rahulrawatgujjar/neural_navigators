import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ---------------------------------------------------------
# PART 1: LOAD AND PREPROCESS DATA
# ---------------------------------------------------------
print("--- Loading and Preprocessing Data ---")

# Update this path to your actual file location
file_path = r'/home/rahulrawatr320/Desktop/hcl/main/ecommerce_returns_enhanced.csv'

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please check the path.")
    exit()

# 1. Target Encoding (Returned=1, Not Returned=0)
if 'Return_Status' in df.columns:
    df['Return_Status'] = df['Return_Status'].apply(lambda x: 1 if x == 'Returned' else 0)
else:
    print("Error: 'Return_Status' column not found.")
    exit()

# 2. Drop identifiers and data leakage columns
cols_to_drop = ['Order_ID', 'Product_ID', 'User_ID', 'Return_Date', 'Return_Reason', 'Days_to_Return']
existing_to_drop = [c for c in cols_to_drop if c in df.columns]
df_processed = df.drop(columns=existing_to_drop)

# 3. Date Feature Engineering (FIXED SECTION)
if 'Order_Date' in df_processed.columns:
    # Use dayfirst=True to handle DD-MM-YYYY formats correctly
    # errors='coerce' turns unparseable dates into NaT (Not a Time) so we can handle them
    df_processed['Order_Date'] = pd.to_datetime(df_processed['Order_Date'], dayfirst=True, errors='coerce')
    
    # Drop rows where date conversion failed (if any)
    df_processed = df_processed.dropna(subset=['Order_Date'])
    
    df_processed['Order_Month'] = df_processed['Order_Date'].dt.month
    df_processed['Order_Weekday'] = df_processed['Order_Date'].dt.weekday
    df_processed = df_processed.drop(columns=['Order_Date'])

# 4. Categorical Encoding
# Handle User_Location
le = LabelEncoder()
if 'User_Location' in df_processed.columns:
    df_processed['User_Location'] = df_processed['User_Location'].fillna('Unknown')
    df_processed['User_Location'] = le.fit_transform(df_processed['User_Location'].astype(str))

# One-Hot Encode other categorical columns
categorical_cols = ['Product_Category', 'User_Gender', 'Payment_Method', 'Shipping_Method']
existing_cat_cols = [c for c in categorical_cols if c in df_processed.columns]
if existing_cat_cols:
    df_processed = pd.get_dummies(df_processed, columns=existing_cat_cols, drop_first=True)

# 5. Numerical Scaling
numerical_cols = ['Product_Price', 'Order_Quantity', 'User_Age', 'Discount_Applied']
existing_num_cols = [c for c in numerical_cols if c in df_processed.columns]

if existing_num_cols:
    # Fill missing values with median
    for c in existing_num_cols:
        if df_processed[c].isna().any():
            df_processed[c] = df_processed[c].fillna(df_processed[c].median())
    
    # Scale features
    scaler = StandardScaler()
    df_processed[existing_num_cols] = scaler.fit_transform(df_processed[existing_num_cols])

print("Preprocessing complete.")

# ---------------------------------------------------------
# PART 2: SETUP MODELS AND GRID SEARCH
# ---------------------------------------------------------

# Try to import XGBoost, otherwise fallback to GradientBoosting
try:
    import xgboost as xgb
    xgb_available = True
    print("XGBoost detected.")
except ImportError:
    xgb_available = False
    print("XGBoost not found. Using GradientBoostingClassifier.")

# Split Data
X = df_processed.drop(columns=['Return_Status'])
y = df_processed['Return_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calculate scale_pos_weight for XGBoost (Negatives / Positives)
pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

# Define Models and Hyperparameter Grids
model_configs = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ['liblinear', 'lbfgs']
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5]
        }
    },
    "Support Vector Machine": {
        "model": SVC(random_state=42, probability=True, class_weight='balanced'),
        "params": {
            "C": [1, 10],
            "kernel": ['rbf']
        }
    }
}

if xgb_available:
    model_configs["XGBoost"] = {
        "model": xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_weight,
            use_label_encoder=False
        ),
        "params": {
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [100, 200]
        }
    }
else:
    model_configs["Gradient Boosting"] = {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
            "n_estimators": [100, 200]
        }
    }

# ---------------------------------------------------------
# PART 3: TRAINING WITH GRID SEARCH (CV=5)
# ---------------------------------------------------------
print("\n--- Starting Grid Search (5-Fold Cross-Validation) ---")

results = []
best_auc = -1.0
best_model_obj = None
best_model_name = ""

for name, config in model_configs.items():
    print(f"Tuning {name}...")
    
    # GridSearchCV finds the best parameters automatically
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        cv=5,                 # 5-Fold Cross Validation
        scoring='roc_auc',    # Optimize for Area Under Curve
        n_jobs=-1,            # Use all processors
        verbose=1
    )
    
    # Train
    grid_search.fit(X_train, y_train)
    
    # Evaluate Best Version on Test Set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    
    # Track the global winner
    if auc > best_auc:
        best_auc = auc
        best_model_obj = best_model
        best_model_name = name
    
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "ROC-AUC": auc,
        "Best Params": str(grid_search.best_params_)
    })

# Display Results
results_df = pd.DataFrame(results).set_index("Model")
print("\n--- Final Model Performance ---")
print(results_df[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]])




# ---------------------------------------------------------
# PART 4: SAVE BEST MODEL AND SCALER AS PKL FILES
# ---------------------------------------------------------

from joblib import dump

print(f"\n--- Saving Best Model: {best_model_name} ---")

# Save the best performing model from GridSearch
dump(best_model_obj, "best_model.pkl")

# Save the scaler used for numerical features
dump(scaler, "scaler.pkl")

print("✅ Best model saved as: best_model.pkl")
print("✅ Scaler saved as: scaler.pkl")
