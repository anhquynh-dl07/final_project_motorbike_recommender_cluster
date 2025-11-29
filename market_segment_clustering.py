import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')

k_best_kmeans = 3

df = pd.read_excel("data_motobikes.xlsx")
cols_to_drop = ['Tiêu đề', 'Địa chỉ', 'Mô tả chi tiết', 'Href']
# Drop the unnecessary columns
df = df.drop(cols_to_drop, axis=1)
# Rename columns to simplify
df = df.rename(columns={
    'Giá': 'price',
    'Khoảng giá min': 'min_price',
    'Khoảng giá max': 'max_price',
    'Thương hiệu': 'brand',
    'Dòng xe': 'model',
    'Năm đăng ký': 'registration_year',
    'Số Km đã đi': 'mileage_km',
    'Tình trạng': 'condition',
    'Loại xe': 'bike_type',
    'Dung tích xe': 'engine_capacity',
    'Xuất xứ': 'origin',
    'Chính sách bảo hành': 'warranty_policy',
    'Trọng lượng': 'weight'
})
df = df.dropna().reset_index(drop=True)
# drop column chỉ có 1 giá trị
df = df.drop(columns=['warranty_policy', 'weight']).reset_index(drop=True)
df['price'] = (
    df['price']
    .astype(str)
    .str.replace('[^0-9]', '', regex=True)   # chỉ giữ lại chữ số
    .replace('', np.nan)
    .astype(float)
)
def parse_minmax_price(s):
    if pd.isna(s):
        return np.nan
    s = str(s).lower().replace("tr", "").replace(" ", "")
    try:
        return float(s) * 1_000_000
    except:
        return np.nan

df['min_price'] = df['min_price'].apply(parse_minmax_price)
df['max_price'] = df['max_price'].apply(parse_minmax_price)
df = df[~(df['price'] == 0)]
# drop column condition vì có 1 giá trị
df = df.drop(columns=['condition'])
df = df[~df['engine_capacity'].astype(str).str.contains('Nhật Bản', case=False, na=False)]
# Thay thế các giá trị không rõ trong cột 'engine_capacity'
df['engine_capacity'] = df['engine_capacity'].replace(
    ['Không biết rõ', 'Đang cập nhật'],
    'Unknown'
)
df = df[~df['origin'].astype(str).str.contains('Bảo hành hãng', case=False, na=False)]
# Thay thế các giá trị không rõ trong cột 'origin'
df['origin'] = df['origin'].replace(
    ['Đang cập nhật', 'Nước khác'],
    'Nước khác'
)
# Chuẩn hóa registration_year
df['registration_year'] = (
    df['registration_year']
    .astype(str)
    .str.lower()
    .str.replace('trước năm', '1980', regex=False)  # thay cụm 'trước năm 1980' bằng '1980'
    .str.extract('(\d{4})')[0]                      # lấy 4 chữ số đầu tiên (nếu có)
)
# Chuyển sang numeric, những giá trị không chuyển được sẽ thành NaN
df['registration_year'] = pd.to_numeric(df['registration_year'], errors='coerce')
# Loại bỏ các giá trị không hợp lệ (nếu có, ví dụ >2025)
df.loc[(df['registration_year'] < 1980) | (df['registration_year'] > 2025), 'registration_year'] = np.nan
# Thêm biến age
df['age'] = 2025 - df['registration_year']  # update year if cần
# gom nhóm brand hiếm
brand_counts = df['brand'].value_counts()
rare_brands = brand_counts[brand_counts < 50].index
df['brand_grouped'] = df['brand'].replace(rare_brands, 'Hãng khác')
def group_model(x):
    counts = x.value_counts()
    rare_models = counts[counts < 100].index
    return x.replace(rare_models, 'Dòng khác')

df['model_grouped'] = df.groupby('brand_grouped')['model'].transform(group_model)
df['segment'] = df['brand_grouped'] + '_' + df['model_grouped']
num_cols = ['age', 'mileage_km', 'min_price', 'max_price', 'price']
for c in num_cols:
    df[f'log_{c}'] = np.log1p(df[c])

# 5️⃣ Chuẩn hóa dữ liệu numeric
num_cols = ['age', 'mileage_km',  'min_price', 'max_price', 'log_price'] # đây là version silhouette tốt nhất

# num_cols = ['age', 'mileage_km',  'log_min_price', 'log_max_price', 'log_price']
# num_cols = ['age', 'mileage_km',  'min_price', 'max_price', 'price']
X = df[num_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# lưu scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

features_used = ['age', 'mileage_km',  'min_price', 'max_price', 'log_price'] # đây là version tốt nhất với silhouette > 0.5

# Gán nhãn cụm
kmeans = KMeans(n_clusters=k_best_kmeans, random_state=42, n_init=50)
kmeans.fit(X_scaled)

# lưu model kmeans
with open("kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# lưu pca model
with open("pca.pkl", "wb") as f:
    pickle.dump(pca, f)

# kmeans_labels = KMeans(n_clusters=k_best_kmeans, random_state=42, n_init=50).fit_predict(X_scaled)
# plt.figure(figsize=(7,5))
# plt.scatter(X_pca[:,0], X_pca[:,1], c=kmeans_labels, cmap='tab10', s=10)
# plt.title('KMeans clusters (PCA 2D)')
# plt.show()

# predict & stats
df['cluster_label'] = kmeans.predict(X_scaled)

# Tính mean + count theo cụm
cluster_summary = (
    df.groupby('cluster_label')
      .agg(
          count=('cluster_label', 'size'),
          avg_price=('price', 'mean'),
          avg_age=('age', 'mean'),
          avg_mileage=('mileage_km', 'mean')
      )
      .sort_values('avg_price')   # sắp xếp theo giá tăng dần
)
