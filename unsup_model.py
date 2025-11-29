import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from function_preprocessing_motorbike import *

# Load dữ liệu đã preprocess (đúng chuẩn như mô hình dự báo giá)
df = preprocess_motobike_data("data_motobikes.xlsx")

# Load model
with open('motobike_price_prediction_model.pkl', 'rb') as f:
    reg_model = pickle.load(f)

# Define cols
cat_cols = ['segment','bike_type','origin','engine_capacity']
num_cols = ['age','mileage_km','min_price','max_price','brand_meanprice']

# Build matrix
X = df[cat_cols + num_cols]

# Predict price
df['price_hat'] = np.expm1(reg_model.predict(X))

# Residual z-score by segment
df['resid'] = df['price'] - df['price_hat']

seg_stats = (
    df.groupby('segment')['resid']
      .agg(['median', 'std'])
      .reset_index()
      .rename(columns={'median':'resid_median', 'std':'resid_std'})
)

# Gộp trở lại với dữ liệu gốc
df = df.merge(seg_stats, on='segment', how='left')

df['resid_z'] = (df['resid'] - df['resid_median']) / (df['resid_std'] + 1e-9)

# Unsupervised features
unsup_feats = ['age','mileage_km','resid_z']

X_unsup = df[unsup_feats].fillna(0).values

# Scale
scaler = StandardScaler()
Xu = scaler.fit_transform(X_unsup)

# Save scaler
pickle.dump(scaler, open("unsup_scaler.pkl", "wb"))

# --- Train Isolation Forest ---
if_model = IsolationForest(n_estimators=300, contamination=0.2, random_state=42)
if_model.fit(Xu)
pickle.dump(if_model, open("if_model.pkl", "wb"))

# --- Train LOF ---
# lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.2)
# # LOF không có predict() sau fit, nên lưu bản fit_predict đặc biệt → dùng decision_function
# # Nhưng để đơn giản, ta chỉ lưu mô hình (scikit-learn cho phép)
# pickle.dump(lof_model, open("lof_model.pkl", "wb"))

lof_model = LocalOutlierFactor(n_neighbors=20,contamination=0.2, novelty=True)
lof_model.fit(Xu)  # X_train là dữ liệu training
pickle.dump(lof_model, open("lof_model.pkl", "wb"))

# --- Train KMeans ---
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(Xu)
pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))

print("Saved: unsup_scaler.pkl, if_model.pkl, lof_model.pkl, kmeans_model.pkl")
