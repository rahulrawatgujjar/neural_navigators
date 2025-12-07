import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("/home/rahulrawatr320/Desktop/hcl/ecommerce_returns_synthetic_data.csv")

# Encode target
df['Return_Status'] = df['Return_Status'].map({
    'Not Returned': 0,
    'Returned': 1
})

# Categorical features
cat_features = [
    'Product_Category',
    'User_Gender',
    'User_Location',
    'Payment_Method',
    'Shipping_Method'
]

# Label encode categorical features
df_encoded = df.copy()
le = LabelEncoder()
for col in cat_features:
    df_encoded[col] = le.fit_transform(df[col])

X_cat = df_encoded[cat_features]
y = df_encoded['Return_Status']

# Apply Chi-Square Test
chi_scores, p_values = chi2(X_cat, y)

chi_df = pd.DataFrame({
    'Feature': cat_features,
    'Chi2_Score': chi_scores,
    'P_Value': p_values
}).sort_values(by='Chi2_Score', ascending=False)

print(chi_df)




# rahulrawatr320@realmebook:~/Desktop/hcl$ python -u "/home/rahulrawatr320/Desktop/hcl/chi_sq_test.py"
#             Feature  Chi2_Score   P_Value
# 2     User_Location    3.556155  0.059325
# 0  Product_Category    2.581906  0.108091
# 3    Payment_Method    0.838747  0.359755
# 4   Shipping_Method    0.144977  0.703382
# 1       User_Gender    0.006024  0.938135