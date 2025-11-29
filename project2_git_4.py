import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import re
from underthesea import word_tokenize, pos_tag
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, hstack
from datetime import datetime
from text_resources import load_teen_dict, load_stopwords
import plotly.express as px
import textwrap
from function_preprocessing_motorbike import preprocess_motobike_data
from build_model_price_anomaly_detection import detect_outliers
import os
import tempfile
import pytz
from datetime import datetime
import plotly.graph_objects as go

# ==========================================================
# 1. CACHED LOADERS
# ==========================================================

@st.cache_resource
def get_resources():
    teen_dict = load_teen_dict()
    stop_words = load_stopwords()
    return teen_dict, stop_words

teen_dict, stop_words = get_resources()


def load_models():

    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    with open("kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    ohe = joblib.load("onehot_encoder.pkl")

    imputer = joblib.load("imputer.pkl")

    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)

    return vectorizer, tfidf_matrix, kmeans, scaler, ohe, imputer, pca


@st.cache_data
def compute_clusters(df_cluster):
    # models are accessed from global scope:
    global scaler, kmeans, pca

    num_cols = ['age', 'mileage_km', 'min_price', 'max_price', 'log_price']

    X_scaled = scaler.transform(df_cluster[num_cols])
    df_cluster['cluster_label'] = kmeans.predict(X_scaled)

    pca_points = pca.transform(X_scaled)
    df_cluster['x'] = pca_points[:, 0]
    df_cluster['y'] = pca_points[:, 1]

    return df_cluster, num_cols

def load_raw_data():
    data = pd.read_excel('data_motobikes.xlsx').rename(columns={
        'Tiêu đề': 'title',
        'Địa chỉ': 'address',
        'Mô tả chi tiết': 'description',
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
    return data

def clean_text(text): # tạo hàm xử lý text với text là chuỗi các từ

    text = str(text).lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s]', '', text)
    text = re.sub(r'\b\w\b', '', text)

    # Teen-code normalization
    words = text.split()
    words = [teen_dict.get(w, w) for w in words]
    text = ' '.join(words)

    # Tokenize & POS filter
    tokenized = word_tokenize(text)
    pos_tagged_text = pos_tag(" ".join(tokenized))
    filtered_words = [word for word, tag in pos_tagged_text if tag != 'T']

    # Stopword removal
    clean_words = [word for word in filtered_words if word not in stop_words]

    # Return string (not list), same as df['content_clean_cosine']
    return " ".join(clean_words)

def clean_df_for_recommender(df):
    ### For numeric part of vector

    # clean price
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

    # Xác định num/ non-num cols để fill NA
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    # Fill NA (num -> median, non-num -> mode)
    # 1. Numeric imputation
    num_imputer = joblib.load('imputer.pkl')
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # 2. Categorical imputation
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Thay thế các giá trị không rõ trong cột 'engine_capacity'
    df['engine_capacity'] = df['engine_capacity'].replace(
        ['Không biết rõ', 'Đang cập nhật', 'Nhật Bản'],
        'Unknown'
    )

    # Thay thế các giá trị không rõ trong cột 'origin', giữ nguyên nhóm "Bảo hành hãng" để xử lý text
    df['origin'] = df['origin'].replace(
        ['Đang cập nhật', 'Nước khác'],
        'Nước khác'
    )

    # Chuẩn hóa registration_year
    df['registration_year'] = (
        df['registration_year']
        .astype(str)
        .str.lower()
        .str.replace('trước năm', '1980', regex=False)
        .str.extract('(\d{4})')[0]
    )
    # Chuyển sang numeric, những giá trị không chuyển được sẽ thành NA
    df['registration_year'] = pd.to_numeric(df['registration_year'], errors='coerce')

    # Fill NA ban đầu
    df['registration_year'] = df['registration_year'].fillna(df['registration_year'].median())

    # Gắn giá trị bất hợp lệ thành NA
    df.loc[
        (df['registration_year'] < 1980) | (df['registration_year'] > 2025),
        'registration_year'
    ] = np.nan

    # Fill NA sau khi loại bất hợp lệ
    df['registration_year'] = df['registration_year'].fillna(df['registration_year'].median())

    # Thêm biến age
    current_year = datetime.now().year
    df['age'] = current_year - df['registration_year']

    # gom nhóm brand hiếm và tạo cột 'segment'
    brand_counts = df['brand'].value_counts()
    rare_brands = brand_counts[brand_counts < 50].index
    df['brand_grouped'] = df['brand'].replace(rare_brands, 'Hãng khác')

    def group_model(x):
        counts = x.value_counts()
        rare_models = counts[counts < 100].index
        return x.replace(rare_models, 'Dòng khác')

    df['model_grouped'] = df.groupby('brand_grouped')['model'].transform(group_model)
    df['segment'] = df['brand_grouped'] + '_' + df['model_grouped']

    # One hot encoding 'bike_type', 'engine_capacity'
    encoded = ohe.transform(df[['bike_type', 'engine_capacity']])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(['bike_type', 'engine_capacity']))
    # merge back to original dataframe
    df = pd.concat([df, encoded_df], axis=1)

    # numeric features
    num_features = ['price','mileage_km','min_price','max_price','age', 'registration_year']
    # log normalize numeric features
    normalized_features = []
    for col in num_features:
        new_col = col + "_log"
        df[new_col] = np.log1p(df[col])
        normalized_features.append(new_col)

    # tạo feature brand_meanprice
    brand_mean_log = df.groupby('brand')['price_log'].mean().rename('brand_meanprice')
    df = df.merge(brand_mean_log, on='brand', how='left')
    normalized_features.append('brand_meanprice')

    # features to turn to a vector: 
    onehot_features = ohe.get_feature_names_out(['bike_type', 'engine_capacity']).tolist()
    num_features = onehot_features + normalized_features

    # Xử lý NaN (nếu có) để tạo dense vector cho việc tính toán cosine similarity lúc sau
    X_num = df[num_features].copy()

    # 1️⃣ Impute missing values
    # imputer = SimpleImputer(strategy="median")
    X_num_imputed = imputer.fit_transform(X_num)

    # 2️⃣ Scaling for num features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num_imputed)

    ### For text part of vector
    # Ở đây đã load tfidf_matrix nên không xử lý phần text nữa

    ### Tạo vector đầu vào bằng cách kết hợp vector TF-IDF và array num col (X_num_scaled)
    # from scipy.sparse import csr_matrix, hstack
    # Chuyển array X_num_scaled thành matrix dạng sparse (ko store các giá trị 0)
    X_num_sparse = csr_matrix(X_num_scaled)

    # Ghép ma trận TF-IDF và ma trận X_num_sparse theo chiều ngang
    X_final = hstack([tfidf_matrix, X_num_sparse])

    return df, X_final

def clean_df_for_clustering(df_cluster):
    cols_drop = ['title', 'address', 'description', 'Href']
    df_cluster = df_cluster.drop(columns=[c for c in cols_drop if c in df_cluster.columns], errors='ignore')
    df_cluster = df_cluster.drop(columns=['warranty_policy', 'weight', 'condition'], errors='ignore')
    df_cluster = df_cluster.dropna()

    # Clean price
    df_cluster['price'] = (
        df_cluster['price'].astype(str)
        .str.replace('[^0-9]', '', regex=True)
        .replace('', np.nan).astype(float)
    )

    # Minimal cleaning df price for display
    if 'price' in df_cluster.columns:
        df_cluster['price'] = df_cluster['price'].astype(str).str.replace('[^0-9]', '', regex=True)
        df_cluster.loc[df_cluster['price'] == '', 'price'] = np.nan
        df_cluster['price'] = pd.to_numeric(df_cluster['price'], errors='coerce')

    # ensure registration_year numeric
    if 'registration_year' in df_cluster.columns:
        df_cluster['registration_year'] = (
            df_cluster['registration_year'].astype(str)
            .str.lower()
            .str.replace('trước năm', '1980', regex=False)
            .str.extract(r'(\d{4})')[0]
        )
        df_cluster['registration_year'] = pd.to_numeric(df_cluster['registration_year'], errors='coerce')
        df_cluster.loc[(df_cluster['registration_year'] < 1980) | (df_cluster['registration_year'] > 2025), 'registration_year'] = np.nan
    
    def parse_price(s):
        if pd.isna(s): return np.nan
        s = str(s).lower().replace("tr", "").replace(" ", "")
        try: return float(s) * 1_000_000
        except: return np.nan

    df_cluster['min_price'] = df_cluster['min_price'].apply(parse_price)
    df_cluster['max_price'] = df_cluster['max_price'].apply(parse_price)

    df_cluster = df_cluster[~(df_cluster['price'] == 0)]

    # Remove invalid engine_capacity
    df_cluster = df_cluster[~df_cluster['engine_capacity'].astype(str).str.contains("Nhật Bản", na=False)]

    # Clean origin
    df_cluster = df_cluster[~df_cluster['origin'].astype(str).str.contains('Bảo hành hãng', case=False, na=False)]
    df_cluster['origin'] = df_cluster['origin'].replace(['Đang cập nhật', 'Nước khác'], 'Nước khác')

    # Registration year
    df_cluster['registration_year'] = (
        df_cluster['registration_year'].astype(str)
        .str.lower()
        .str.replace('trước năm', '1980')
        .str.extract('(\d{4})')[0]
    ).astype(float)

    df_cluster.loc[(df_cluster['registration_year'] < 1980) | (df_cluster['registration_year'] > 2025),
            'registration_year'] = np.nan

    df_cluster["age"] = 2025 - df_cluster["registration_year"]

    # Log transforms
    numeric_cols = ['age', 'mileage_km', 'min_price', 'max_price', 'price']
    for c in numeric_cols:
        df_cluster[f"log_{c}"] = np.log1p(df_cluster[c])

    df_cluster = df_cluster.dropna(subset=numeric_cols)

    return df_cluster

# ==========================================================
# LOAD EVERYTHING (CACHED)
# ==========================================================

@st.cache_data
def get_clean_recommender_data():
    df_raw = load_raw_data()
    return clean_df_for_recommender(df_raw.copy())

@st.cache_data
def get_cluster_data():
    df_raw = load_raw_data()
    df_cluster = clean_df_for_clustering(df_raw.copy())
    df_cluster, num_cols = compute_clusters(df_cluster)
    return df_cluster, num_cols

# Load models (already cached)
vectorizer, tfidf_matrix, kmeans, scaler, ohe, imputer, pca = load_models()

# Load cleaned datasets
df_clean, X_final = get_clean_recommender_data()
df_cluster, num_cols = get_cluster_data()


# ==========================================================
# FUNCTIONS
# ==========================================================
def preprocess_user_input(price, min_price, max_price, mileage_km, registration_year):
    age = 2025 - registration_year
    log_price = np.log1p(price)
    X = np.array([[age, mileage_km, min_price, max_price, log_price]])
    return scaler.transform(X)

def get_top_n_similar_by_content(df, X_final, title, top_n=5):
    """
    Given a bike title, return top N most similar bikes based on
    combined TF-IDF + numeric features vector.

    Params:
        df (DataFrame): cleaned dataframe returned from clean_df_for_recommender
        X_final (sparse matrix): combined feature matrix
        title (str): the selected bike title
        top_n (int): number of similar bikes to return

    Returns:
        df_recommend (DataFrame): rows of top-N similar bikes
        scores (list): similarity scores
    """

    # Find the index of the selected bike
    matches = df.index[df['title'] == title]

    if len(matches) == 0:
        return None, []   # title not found

    idx = matches[0]

    # Compute cosine similarity for this single item
    sims = cosine_similarity(X_final[idx], X_final).flatten()

    # Sort by similarity (descending), ignore itself
    ranked_indices = np.argsort(sims)[::-1]

    # Remove itself
    ranked_indices = ranked_indices[ranked_indices != idx]

    # Take top-N
    top_indices = ranked_indices[:top_n]
    top_scores = sims[top_indices]

    # Return matching rows + scores
    df_recommend = df.iloc[top_indices].copy()
    df_recommend['similarity_score'] = top_scores

    return df_recommend, top_scores.tolist()

# helper: safe format number
def fmt_vnd(x):
    try:
        return f"{int(x):,} VNĐ"
    except:
        return '-'

MODEL_PATH = "motobike_price_prediction_model.pkl"
TRAINING_DATA = "data_motobikes.xlsx"  # optional, used to compute brand_meanprice & grouping to match train

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def build_training_helpers(path=TRAINING_DATA):
    """
    Load training data & build grouping rules + statistical thresholds
    (p10/p90, residual mean/std) for anomaly detection.
    """
    if not os.path.exists(path):
        return None

    try:
        df_train = preprocess_motobike_data(path)
        # =============== LOAD MODELS ===============
        with open("unsup_scaler.pkl", "rb") as f:
            scaler_nom = pickle.load(f)

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("kmeans_model.pkl", "rb") as f:
            kmeans_anom = pickle.load(f)

        with open("kmeans.pkl", "rb") as f:
            kmeans = pickle.load(f)

        # =============== 1) BRAND GROUPING ==================
        brand_counts = df_train['brand'].value_counts()
        rare_brands = set(brand_counts[brand_counts < 50].index)

        # model grouping by brand_grouped
        model_group_maps = {}
        for bg, g in df_train.groupby('brand_grouped'):
            counts = g['model'].value_counts()
            rare_models = set(counts[counts < 100].index)
            model_group_maps[bg] = rare_models

        # mean price for brand
        brand_mean_map = df_train.groupby('brand')['brand_meanprice'].first().to_dict()

        # =============== 2) PRICE P10/P90 BY SEGMENT ==================
        seg_price_stats = (
            df_train.groupby('segment')['price']
                    .quantile([0.10, 0.90])
                    .unstack(level=1)
                    .rename(columns={0.10:'p10', 0.90:'p90'})
        ).reset_index()

        seg_price_map = seg_price_stats.set_index('segment').to_dict('index')
        # format: seg_price_map[segment] = {'p10':..., 'p90':...}

        # =============== 3) RESIDUAL STATS BY SEGMENT ==================

        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        # Define cols
        cat_cols = ['segment','bike_type','origin','engine_capacity']
        num_cols = ['age','mileage_km','min_price','max_price','brand_meanprice']

        # Build matrix
        X = df_train[cat_cols + num_cols]
        # y = df['log_price']

        # Predict price
        df_train['price_hat'] = np.expm1(model.predict(X))
        df_train['resid'] = df_train['price'] - df_train['price_hat']  # price_hat từ preprocess

        seg_resid_stats = (
            df_train.groupby('segment')['resid']
                    .agg(['mean', 'std'])
                    .rename(columns={'mean': 'resid_mean', 'std': 'resid_std'})
        ).reset_index()

        seg_resid_map = seg_resid_stats.set_index('segment').to_dict('index')
        # format: seg_resid_map[seg] = {'resid_mean':..., 'resid_std':...}

        
        num_cols = ['age','mileage_km','min_price','max_price','log_price']

        X = df_train[num_cols].dropna()
        X_scaled = scaler.transform(X)

        df_train['cluster_label'] = kmeans.predict(X_scaled)

        cluster_summary = (
            df_train.groupby('cluster_label')
                    .agg(
                        avg_price=('price','mean'),
                        avg_age=('age','mean'),
                        avg_mileage=('mileage_km','mean'),
                        count=('cluster_label','size')
                    )
                    .to_dict('index')
        )

        return {
            'rare_brands': rare_brands,
            'model_group_maps': model_group_maps,
            'brand_mean_map': brand_mean_map,
            'seg_price_map': seg_price_map,
            'seg_resid_map': seg_resid_map,
            'cluster_summary': cluster_summary,
            'cluster_model': kmeans,
            'cluster_scaler': scaler
                }

    except Exception as e:
        print("Error building helpers:", e)
        return None


helpers = build_training_helpers(TRAINING_DATA)
model = load_model(MODEL_PATH)



# ==========================================================
# STREAMLIT PAGES
# ==========================================================
st.set_page_config(
    page_title="Hệ thống gợi ý xe máy tương tự và phân cụm xe máy",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Hệ thống gợi ý xe máy tương tự và phân cụm xe máy")
st.image("xe_may_cu2.jpg",  width=1500)

st.sidebar.markdown("""
## Hệ thống gợi ý xe máy tương tự và phân cụm xe máy
""")

st.sidebar.markdown("""
### Thành viên nhóm 6
1. Vũ Thị Ngọc Anh
2. Nguyễn Phạm Quỳnh Anh
""")

st.sidebar.markdown("### Menu")   
menu = ["Giới thiệu", "Bài toán nghiệp vụ", "Đánh giá mô hình và Báo cáo",
        "Gợi ý mẫu xe tương tự", "Xác định phân khúc xe máy"]

# page = st.sidebar.selectbox("Menu", menu, label_visibility="collapsed")
page = st.sidebar.selectbox(
    "Menu",
    menu,
    label_visibility="collapsed",
    key="menu_select",
    # enable full width
)


# ==========================================================
# STYLES
# ==========================================================

BASE_CSS = """
<style>
:root{
  --accent-1: #ffde37;       /* Your yellow */
  --accent-2: #e5c620;       /* Slightly darker yellow for gradients */
  --muted: #4a4a4a;
  --card-bg: #fff7c2;        /* Soft light yellow background */
  --glass: rgba(255,255,255,0.55);
}

/* Background */
html, body {
  background: linear-gradient(180deg, #fff5a0 0%, #ffef73 100%);
  color: #000000 !important;
}

/* Header / hero section */
.header-hero {
  background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
  padding: 22px;
  border-radius: 12px;
  color: #000000;
  font-weight: 600;
  margin-bottom: 18px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.12);
}

/* Small muted text */
.small-muted {
  color: var(--muted);
  font-size: 13px;
}

/* Cards */
.card {
  background: var(--card-bg);
  padding: 14px;
  border-radius: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.10);
  color: #000000;
}

/* Base typography */
h1, h2, h3, h4, h5, h6, p, span, div {
  color: #000000 !important;
}

/* Bike title / subtitles */
.bike-title{
  font-size:18px;
  font-weight:700;
  margin-bottom:4px;
}

.bike-sub{
  font-size:13px;
  color:var(--muted);
  margin-bottom:6px;
}

/* Cluster cards */
.cluster-card{
  padding:18px;
  border-radius:12px;
  color:#000000;
  margin-bottom:12px;
  font-weight:600;
}

/* Cluster variants using your yellow palette */
.cluster-0{
  background:linear-gradient(135deg, #ffeb7a, #ffde37);
}
.cluster-1{
  background:linear-gradient(135deg, #ffe45c, #e5c620);
}
.cluster-2{
  background:linear-gradient(135deg, #fff1a1, #ffde37);
}
</style>
"""

st.markdown(BASE_CSS, unsafe_allow_html=True)

@st.cache_data
def load_reference_data():
    return preprocess_motobike_data(TRAINING_DATA)

df_ref = load_reference_data()
brand_list = sorted(df_ref['brand_grouped'].dropna().unique())
model_list = sorted(df_ref['model_grouped'].dropna().unique())
bike_type_list = sorted(df_ref['bike_type'].dropna().unique())
origin_list = sorted(df_ref['origin'].dropna().unique())
engine_capacity_list = sorted(df_ref['engine_capacity'].dropna().unique())

# ==========================================================
# PAGE CONTENT
# ==========================================================

if page == 'Giới thiệu':
    # st.title("Hệ thống gợi ý xe máy tương tự và phân cụm xe máy")
    st.markdown("""
        <h1 style='font-size:35px; font-weight:800; margin-bottom:8px;'>
            Giới thiệu
        </h1>
        <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
    """, unsafe_allow_html=True)    
    # st.image("xe_may_cu2.jpg")
    st.subheader("[Trang chủ Chợ Tốt](https://www.chotot.com/)")

        # Function for light yellow pad header
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True)
    
    yellow_pad_header('Giới thiệu dự án')
    st.markdown('''Đây là dự án xây dựng hệ thống hỗ trợ **gợi ý mẫu xe máy tương tự** 
và **phân khúc xe máy bằng phương pháp phân cụm** trên nền tảng *Chợ Tốt* – 
trong khóa đồ án tốt nghiệp Data Science and Machine Learning 2024 lớp DL07_K308 của nhóm 6.

Thành viên nhóm gồm có:
1. Vũ Thị Ngọc Anh  
2. Nguyễn Phạm Quỳnh Anh
''')

    yellow_pad_header('Mục tiêu của dự án')
    st.markdown("""
    **1. Xây dựng mô hình đề xuất thông minh:**
    - Đề xuất các mẫu xe máy tương đồng cho một mẫu được chọn hoặc theo từ khóa tìm kiếm.
    - Kết hợp nhiều nguồn thông tin (thông số kỹ thuật, hình ảnh, mô tả, giá, đánh giá) để tăng độ chính xác.

    **2. Phân khúc thị trường xe máy:**
    - Phân loại sản phẩm theo nhóm theo tệp giá, tuổi xe, khoảng giá tối thiểu/tối đa.
    - Hỗ trợ định giá và xây dựng chiến lược marketing hiệu quả hơn.
    """)

    yellow_pad_header('Phân công công việc')
    st.write("""
    - **Xử lý dữ liệu:** Ngọc Anh và Quỳnh Anh  
    - **Gợi ý xe máy bằng Gensim:** Quỳnh Anh  
    - **Gợi ý xe máy bằng Cosine similarity:** Quỳnh Anh và Ngọc Anh 
    - **Phân khúc xe máy bằng phương pháp phân cụm:** Ngọc Anh  
    - **Làm slide:** Ngọc Anh và Quỳnh Anh  
    - **Giao diện Streamlit:** Quỳnh Anh và Ngọc Anh
    """)
    
elif page == 'Bài toán nghiệp vụ':
    st.markdown("""
    <h1 style='font-size:35px; font-weight:800; margin-bottom:8px;'>
        Bài toán nghiệp vụ
    </h1>
    <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
""", unsafe_allow_html=True)
    # Function for light yellow pad header
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True)

    yellow_pad_header('Vấn đề nghiệp vụ')
    st.markdown("""
        - Người dùng gặp khó khăn khi tìm xe phù hợp trong hàng trăm lựa chọn.
        - Chưa có hệ thống gợi ý xe tương tự khi người dùng chọn một mẫu cụ thể hoặc tìm kiếm theo từ khóa.
        - Thị trường xe máy rất đa dạng → khó nhận diện các phân khúc rõ ràng.
        - Cần hệ thống gợi ý & phân khúc tự động để hỗ trợ người dùng và đội ngũ phân tích.""")

    yellow_pad_header('Bài toán đặt ra')
    st.markdown("""
        1. Xây dựng mô hình **Gợi ý xe tương tự**
        - Sử dụng các đặc trưng từ mô tả xe và thông số kỹ thuật
        - Gợi ý các mẫu xe tương tự với xe được chọn hoặc theo từ khóa tìm kiếm.
        &nbsp;
        2. Xây dựng mô hình **Phân khúc thị trường xe bằng phương pháp phân cụm**
        - Phân cụm thị trường xe máy dựa các đặc trưng giá xe, tuổi xe, số km đã chạy, khoảng giá tối thiểu, tối đa.
        - Giúp nhận diện và phân loại xe theo các phân khúc khác nhau.
                """)
    
    yellow_pad_header('Phạm vi triển khai')
    st.markdown("""
        **1. Tiền xử lý dữ liệu và chuẩn hóa**:  
            - Chuẩn hóa các thông số của xe.  
            - Làm sạch dữ liệu và chuẩn hóa trường thông tin cho mô hình.  
                
        **2. Trích xuất đặc trưng văn bản và tính độ tương đồng**:  
            - Sử dụng **TF-IDF Vectorizer** để mã hóa mô tả và thông tin kỹ thuật.  
            - Tính độ tương đồng bằng **gensim similarity** và **cosine similarity**.  
            - Chọn phương pháp cho **điểm cao hơn** và **nghĩa đúng hơn** để đưa vào hệ thống gợi ý.  
                
        **3. Phân cụm thị trường (Clustering)**:  
            - Thử nghiệm trên các thuật toán: KMeans, Bisecting KMeans, Agglomerative Clustering  
            - Đánh giá bằng inertia, silhouette score, tính diễn giải.  
            - Chọn **KMeans** vì có hiệu suất ổn định, dễ diễn giải và ranh giới cụm phù hợp hơn với dữ liệu.

        **4. Xây dựng GUI trên Streamlit**:  
            - Cho phép người dùng **chọn xe trong danh sách** hoặc **nhập mô tả xe** → trả về **danh sách mẫu xe tương tự có trong sàn**.  
            - Cho phép **nhập tên xe** → hiển thị **xe thuộc cụm/phân khúc nào**.
                """)

    yellow_pad_header('Thu thập dữ liệu')
    st.markdown("""
        - Bộ dữ liệu gồm **7.208 tin đăng** với **18 thuộc tính** (thương hiệu, dòng xe, số km, năm đăng ký, giá niêm yết, mô tả, v.v…) được thu thập từ nền tảng **Chợ Tốt** (trước ngày 01/07/2025).
        - Bộ dữ liệu bao gồm các thông tin sau:
            - **id**: số thứ tự của sản phẩm trong bộ dữ liệu  
            - **Tiêu đề**: tựa đề bài đăng bán sản phẩm  
            - **Giá**: giá bán của xe máy  
            - **Khoảng giá min**: giá sàn ước tính của xe máy  
            - **Khoảng giá max**: giá trần ước tính của xe máy  
            - **Địa chỉ**: địa chỉ giao dịch (phường, quận, thành phố Hồ Chí Minh)  
            - **Mô tả chi tiết**: mô tả thêm về sản phẩm — đặc điểm nổi bật, tình trạng, thông tin khác  
            - **Thương hiệu**: hãng sản xuất (Honda, Yamaha, Piaggio, SYM…)  
            - **Dòng xe**: dòng xe cụ thể (Air Blade, Vespa, Exciter, LEAD, Vario, …)  
            - **Năm đăng ký**: năm đăng ký lần đầu của xe  
            - **Số km đã đi**: số kilomet xe đã vận hành  
            - **Tình trạng**: tình trạng hiện tại (ví dụ: đã sử dụng)  
            - **Loại xe**: Xe số, Tay ga, Tay côn/Moto  
            - **Dung tích xe**: dung tích xi-lanh (ví dụ: Dưới 50cc, 50–100cc, 100–175cc, …)  
            - **Xuất xứ**: quốc gia sản xuất (Việt Nam, Đài Loan, Nhật Bản, ...)  
            - **Chính sách bảo hành**: thông tin bảo hành nếu có  
            - **Trọng lượng**: trọng lượng ước tính của xe  
            - **Href**: đường dẫn tới bài đăng sản phẩm 
                """)

elif page == 'Đánh giá mô hình và Báo cáo':
    st.markdown("""
    <h1 style='font-size:35px; font-weight:800; margin-bottom:8px;'>
        Đánh giá mô hình và Báo cáo
    </h1>
    <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
""", unsafe_allow_html=True)
    
    # Function for light yellow pad header
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True) 

    yellow_pad_header('Thống kê mô tả sơ bộ')


    st.markdown("""        
    Bộ dữ liệu gồm **7.208 tin đăng** với **18 thuộc tính** (thương hiệu, dòng xe, số km, năm đăng ký, giá niêm yết, mô tả…) được thu thập từ nền tảng **Chợ Tốt** (trước ngày 01/07/2025).  
                """)
    
    image_width = 600
    # Hiển thị 4 biểu đồ dạng lưới 2x2
    col1, col2 = st.columns(2)
    with col1:
        st.image("brand_grouped_count.png", width=image_width) # Thêm width=500
        st.image("age_bin_stats.png", width=image_width)       # Thêm width=500

    with col2:
        st.image("price_bin_stats.png", width=image_width)     # Thêm width=500
        st.image("mileage_bin_stats.png", width=image_width)   # Thêm width=500

    yellow_pad_header('Mô hình gợi ý xe máy tương tự')

    # with open("data/data_motobikes.xlsx", "rb") as f:
    #     st.download_button(
    #         label="📥 Tải xuống dữ liệu xe máy (Excel)",
    #         data=f,
    #         file_name="data_motobikes.xlsx",
    #         mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    #     )

    st.markdown('#### 1. Hướng xử lý')
    st.write('''
             - Chuẩn hóa và làm sạch dữ liệu.
             - Chia khoảng một số đặc trưng kiểu số để tạo thêm các đặc trưng phân loại mới (khoảng giá, tình trạng dựa theo số km chạy, tuổi xe, dung tích xe)
             - Gom các đặc trưng phân loại thành biến text
             - Làm sạch text và tokenize, xây dựng ma trận tương đồng (sparse matrix) giữa các văn bản để đánh giá mức độ giống nhau
             - Tính độ tương đồng bằng gensim và cosine similarity
                 - Trường hợp 1: gợi ý xe theo id sản phẩm được chọn
                    - Người dùng chọn xe từ danh sách xe trong tập dữ liệu
                    - Dựa trên ma trận tương đồng, tìm các xe có similarity score cao nhất.
                    - Tính độ tương đồng trung bình giữa 5 mẫu gợi ý cho một mẫu, sau đó áp dụng cho 7000 mẫu trong tập dữ liệu và tính trung bình.

                 - Trường hợp 2: gợi ý xe theo cụm từ khóa tìm kiểm (vd: “honda vision xanh dưới 15 triệu”)
                    - Người dùng nhập từ khóa tìm kiếm. 
                    - Xử lý từ khóa và chuyển từ khóa thành vector số dựa trên từ điển và TF-IDF
                    - Tính độ tương đồng giữa từ khóa và tất cả xe trong dữ liệu. 
                    - Sắp xếp và lấy ra 5 xe gợi ý phù hợp nhất.
                    - Cho danh sách 10 cụm từ khóa tìm kiếm. Tính độ tương đồng trung bình giữa 5 mẫu gợi ý cho một mẫu, sau đó áp dụng cho 10 cụm từ trên và tính trung bình
             ''')
    
    st.markdown('#### 2. Kết quả')
    st.write('Giữa 02 mô hình Gensim và Cosine similarity, Cosine similarity, trong cả 2 trường hợp chọn xe có sẵn hoặc tìm bằng từ khóa, cho điểm tương đồng trung bình cao hơn so với Gensim và cho các gợi ý sát nghĩa hơn Gensim.\nMô hình dùng để dự đoán xe trong ứng dụng này là Cosine similarity.') 

    yellow_pad_header('Mô hình phân khúc xe máy')
    
    st.markdown('#### 1. Xử lý dữ liệu')
    st.write('Dữ liệu được làm sạch, các đặc trưng biến số liên tục như giá, khoảng giá thấp nhất, lớn nhất, tuổi xe, số km đã đi được chọn để tạo mô hình phân cụm')

    st.markdown('#### 2. Phân cụm bằng các phương pháp khác nhau')
    st.write('''
    Mô hình phân cụm được xây dựng trên 02 môi trường: máy học truyền thống (sci-kit learn) và PySpark.
    - Máy học truyền thống: KMeans, Bisect Kmeans, Agglomerative clustering
    - PySpark: Kmeans, Bisecting Kmeans, GMM.

    ''')

    st.markdown('#### 3. Kết quả')


    st.markdown('''
    Số cụm được tạo thành trên mô hình máy học truyền thống: **03 cụm**
    Số cụm được tạo thành trên PySpark: **02 cụm**''')
    st.image("silhoutte_sklearn.png",width=image_width)                

    st.markdown('''      
    KMeans trên môi trường máy học truyền thống cho kết quả silhoutte score cao nhất và kết quả phân cụm dễ diễn giải hơn.
    
    **Phân loại phân khúc xe**:                
    1/ Cụm 0: Phân khúc Xe Phổ Thông – Trung cấp (Mid-range Popular Motorcycles): Xe tuổi trung bình, giá vừa phải, phù hợp đại đa số người mua.   
    2/ Cụm 1: Phân khúc Xe Cao Cấp – Premium / High-end Motorcycles: Tiêu biểu là các dòng SH, Vespa cao cấp, phân khối lớn, xe mới chạy ít.          
    3/ Cụm 2: Phân khúc Xe Cũ – Tiết Kiệm (Budget Used Motorcycles): Giá rẻ nhất, xe tuổi cao, chạy nhiều — phù hợp khách cần xe rẻ để di chuyển cơ bản.
    ''')


    st.write('''Trong 3 mô hình phân cụm KMeans, Bisect KMeans và Agglomerate thì KMeans với k = 3 cho kết quả phân cụm tốt nhất.
            nên mô hình phân cụm xe được sử dụng trong ứng dụng này là KMeans với k = 3.''')

    st.markdown('#### 4. Thống kê theo từng cụm:')

    st.write('Trực quan hóa')
    st.image('pca_clusters.png')

    cluster_summary = (
        df_cluster.groupby('cluster_label')
        .agg(
            count=('cluster_label', 'size'),
            avg_price=('price', 'mean'),
            avg_age=('age', 'mean'),
            avg_mileage=('mileage_km', 'mean')
        )
        .sort_values('cluster_label')
    )


    # Rename the index (cluster_label → Nhãn cụm xe)
    cluster_summary = cluster_summary.rename_axis("Nhãn cụm xe")

    # Rename columns
    cluster_summary = cluster_summary.rename(columns={
        "count": "Số lượng (xe)",
        "avg_price": "Giá trung bình (VND)",
        "avg_age": "Tuổi trung bình (năm)",
        "avg_mileage": "Số km trung bình (km)"
    })

    # Format số nguyên và thêm dấu phẩy
    cluster_summary["Giá trung bình (VND)"] = (
        cluster_summary["Giá trung bình (VND)"]
            .round(0).astype(int)
            .map(lambda x: f"{x:,}")
    )

    cluster_summary["Số km trung bình (km)"] = (
        cluster_summary["Số km trung bình (km)"]
            .round(0).astype(int)
            .map(lambda x: f"{x:,}")
    )

    st.dataframe(cluster_summary, width='stretch')


elif page == "Gợi ý mẫu xe tương tự":
    # Main page header
    st.markdown("""
    <h1 style='font-size:35px; font-weight:800; margin-bottom:8px;'>
        Gợi ý mẫu xe tương tự
    </h1>
    <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
    """, unsafe_allow_html=True)

    # Prepare data + vector
    df_clean, X_final = df_clean, X_final

    # Styling and helpers
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <style>
        .card {
            border-radius: 10px;
            padding: 14px 16px;
            margin: 8px 0;
            border: 1px solid #eee;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            background-color: #ffffff;
        }
        .bike-title {
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .bike-sub {
            font-size: 13px;
            color: #666666;
        }
        .small-muted {
            font-size: 12px;
            color: #777777;
        }
        </style>
    """, unsafe_allow_html=True)

    def display_bike_card(row):
        title = row.get('title', 'N/A')
        price = fmt_vnd(row.get('price', None))
        brand = row.get('brand', '-')
        model = row.get('model', '-')
        km = row.get('mileage_km', '-')
        year = row.get('registration_year', '-')
        year_shown = int(year) if str(year).isdigit() else year
        origin = row.get('origin', '-')
        desc = row.get('description', '')

        card_html = f"""
        <div class='card'>
            <div style='display:flex; gap:14px; align-items:center'>
                <div style='flex:1'>
                    <div class='bike-title'>{title}</div>
                    <div class='bike-sub'>{brand} — {model} • {origin}</div>
                    <div style='margin-top:6px'>{textwrap.shorten(str(desc), width=220)}</div>
                </div>
                <div style='text-align:right; min-width:150px'>
                    <div style='font-weight:700; font-size:16px'>{price}</div>
                    <div class='small-muted' style='margin-top:8px'>
                        Số km: {km}<br/>Năm: {year_shown}
                    </div>
                </div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    # ✅ Main interaction
    yellow_pad_header("Gợi ý theo mẫu có sẵn")

    titles_list = df_clean['title'].unique().tolist()
    selected = st.selectbox("Chọn 1 mẫu trong danh sách", titles_list)

    if st.button("Gợi ý"):
        with st.spinner("🔎 Đang tìm mẫu tương tự..."):
            df_top, scores = get_top_n_similar_by_content(
                df_clean,
                X_final,
                title=selected,
                top_n=5
            )

        if df_top is None or len(df_top) == 0:
            st.warning("Không tìm thấy kết quả — kiểm tra lại dữ liệu.")
        else:
            st.success(f"Đã tìm {len(df_top)} mẫu tương tự ✅")

            # ✅ Show selected bike
            st.markdown("#### 🔶 Mẫu bạn đã chọn")
            selected_row = df_clean[df_clean["title"] == selected].iloc[0]
            display_bike_card(selected_row)

            # ✅ Show recommendations
            st.markdown("#### 🔶 Các mẫu tương tự")
            for _, row in df_top.iterrows():
                display_bike_card(row)
                st.caption(f"Similarity score: {row['similarity_score']:.3f}")

        
    # theo từ khóa
    yellow_pad_header("Tìm kiếm theo từ khóa")

    q = st.text_input('Nhập từ khóa tìm kiếm, ví dụ: "honda vision 2014 màu đỏ"')
    top_k = st.selectbox('Số kết quả trả về', [1, 3, 5, 10])

    if st.button('Tìm kiếm') and q.strip():
        with st.spinner('Đang xử lý từ khóa...'):

            # 1) Clean query like training data
            q_clean = clean_text(q)

            # 2) Vectorize cleaned query
            q_vec_tfidf = vectorizer.transform([q_clean])

            # 3) Pad numeric features with zeros
            num_dim = X_final.shape[1] - q_vec_tfidf.shape[1]
            q_num_zeros = np.zeros((1, num_dim))

            # 4) Combine TF-IDF + numeric zeros
            q_vec = hstack([q_vec_tfidf, q_num_zeros])

            # 5) Compute similarity
            sim_scores = cosine_similarity(q_vec, X_final).flatten()

            # 6) Select top results
            idxs = sim_scores.argsort()[::-1][:top_k]

            # 7) Select rows from cleaned DF
            res_df = df_clean.iloc[idxs].copy()
            res_df['similarity_score'] = sim_scores[idxs]

        st.success(f'Kết quả top {top_k} cho: "{q}"')

        # 8) Display
        for _, row in res_df.iterrows():
            display_bike_card(row)
            st.caption(f"Similarity score: {row['similarity_score']:.3f}")


elif page == "Xác định phân khúc xe máy":
    # Main page header
    st.markdown("""
    <h1 style='font-size:35px; font-weight:800; margin-bottom:8px;'>
        Phân cụm phân khúc xe máy
    </h1>
    <div style='width:90px; height:6px; background:#FF9A00; border-radius:3px; margin-bottom:24px;'></div>
    """, unsafe_allow_html=True)

    # Yellow pad header function (keep for consistent style)
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True)

    # ----- Card CSS -----
    st.markdown("""
        <style>
        .card {
            border-radius: 10px;
            padding: 14px 16px;
            margin: 8px 0;
            border: 1px solid #eee;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            background-color: #ffffff;
        }
        .bike-title {
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .bike-sub {
            font-size: 13px;
            color: #666666;
        }
        .small-muted {
            font-size: 12px;
            color: #777777;
        }
        </style>
    """, unsafe_allow_html=True)

        # ======================== HEADER + CSS ========================
    def yellow_pad_header(text):
        st.markdown(f"""
            <div style="
                background: #FFF4C2;
                border-left: 6px solid #FFDE37;
                padding: 12px 18px;
                border-radius: 6px;
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin: 15px 0 10px 0;
            ">
                {text}
            </div>
        """, unsafe_allow_html=True)


    st.markdown("""
    <style>
    .card {
        border-radius: 10px;
        padding: 14px 16px;
        margin: 8px 0;
        border: 1px solid #eee;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        background-color: #ffffff;
    }
    .cluster-card {
        border-radius: 10px;
        padding: 14px 18px;
        margin: 10px 0;
        border: 1px solid #E5C600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        color: #000000;
    }
    .cluster-title {
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 6px;
        color: #000000;
    }
    .cluster-desc {
        font-size: 14px;
        color: #000000;
        line-height: 1.4;
    }
    .cluster-0 { background: #FFF7A6; }
    .cluster-1 { background: #FFE970; }
    .cluster-2 { background: #FFDE37; }
    </style>
    """, unsafe_allow_html=True)


    # yellow_pad_header("Phân tích & Định giá theo cụm (Cluster)")


    # ======================== PREDICT CLUSTER ========================

    cluster_names = {
        0: "Phân khúc Xe Phổ Thông – Trung cấp",
        1: "Phân khúc Xe Cao Cấp – Premium",
        2: "Phân khúc Xe Cũ – Tiết Kiệm"
    }

    cluster_cards = {
        0: """
            <div class="cluster-card cluster-0">
                <div class="cluster-title">Phân khúc Xe Phổ Thông – Trung cấp</div>
                <div class="cluster-desc">
                    Giá thấp – tuổi xe trung bình – số km chạy vừa phải.<br>
                    Phân khúc xe phổ thông, phù hợp đa số người mua.
                </div>
            </div>
        """,
        1: """
            <div class="cluster-card cluster-1">
                <div class="cluster-title">Phân khúc Xe Cao Cấp – Premium</div>
                <div class="cluster-desc">
                    Xe mới – ít km – giá cao.<br>
                    Các dòng SH, Vespa, xe cao cấp, tình trạng tốt.
                </div>
            </div>
        """,
        2: """
            <div class="cluster-card cluster-2">
                <div class="cluster-title">Phân khúc Xe Cũ – Tiết Kiệm</div>
                <div class="cluster-desc">
                    Giá thấp nhất – km rất cao – tuổi xe lớn.<br>
                    Phân khúc xe đã cũ hoặc có dấu hiệu xuống cấp.
                </div>
            </div>
        """
    }

    
    # Tạo 2 TAB
    tab_user, tab_admin, tab_dash = st.tabs(["User nhập tin", "Admin duyệt", "Dashboard"])

    # ======================================
    # 1) TAB USER
    # ======================================
    with tab_user:


        # Hàm lưu request user vào file Excel
        def save_user_request(df_input, cluster_label):
            save_path = "user_submissions.xlsx"
            
            # Tạo bản sao để tránh thay đổi DataFrame gốc (df_in)
            df_save = df_input.copy() 

            # 1. Kiểm tra xem cột 'post_time' có tồn tại không
            if 'post_time' in df_save.columns:
                # 2. Nếu cột là timezone-aware (có múi giờ), chuyển nó thành timezone-unaware
                if df_save['post_time'].dt.tz is not None:
                    # .dt.tz_localize(None) sẽ loại bỏ thông tin múi giờ (GMT+7)
                    # Dữ liệu ngày giờ vẫn giữ nguyên giá trị theo giờ địa phương (GMT+7)
                    df_save['post_time'] = df_save['post_time'].dt.tz_localize(None)

            df_save["cluster_label"] = cluster_label
            # df_save["is_outlier"] = 0
            df_save["approved"] = False

            if os.path.exists(save_path):
                old = pd.read_excel(save_path)
                new = pd.concat([old, df_save], ignore_index=True)
            else:
                new = df_save.copy()

            # Đoạn này sẽ chạy trơn tru vì cột ngày giờ đã là timezone-unaware
            new.to_excel(save_path, index=False)

        # ============================
        # 1.1 Nhập tay
        # ============================
        st.subheader("Nhập thông tin xe cần rao bán")
        col1, col2 = st.columns(2)

        with col1:
            title = st.text_input("Tiêu đề", value='Bán xe')
            address = st.text_input("Địa chỉ", value= 'Quận 1, TP. HCM')
            brand = st.selectbox("Thương hiệu", brand_list)
            model_name = st.selectbox("Dòng xe", model_list)
            bike_type = st.selectbox("Loại xe", bike_type_list)
            origin = st.selectbox("Xuất xứ", origin_list)
            engine_capacity = st.selectbox("Dung tích", engine_capacity_list)

        with col2:
            description = st.text_input("Mô tả chi tiết", value='Bán xe giá rẻ')
            registration_year = st.number_input("Năm đăng ký", 1980, 2025, 2019)
            mileage_km = st.number_input("Số km đã đi", 0, value=10000)
            min_price = st.number_input("Khoảng giá min", 0)
            max_price = st.number_input("Khoảng giá max", 0)
            price = st.number_input("Giá niêm yết", 0, value=20000000)
        
        # Thêm ngày giờ đăng tin
        col_d, col_t = st.columns(2)

        with col_d:
            # Bạn có thể giữ nguyên giá trị mặc định là giờ hiện tại
            post_date = st.date_input("Ngày đăng tin", value=pd.Timestamp.now(tz=pytz.timezone('Asia/Ho_Chi_Minh')).date())

        with col_t:
            post_time = st.time_input("Giờ đăng tin", value=pd.Timestamp.now(tz=pytz.timezone('Asia/Ho_Chi_Minh')).time())

        # Gộp thành datetime và gán múi giờ:
        # 1. Tạo đối tượng datetime thô (naive datetime) từ date và time input
        naive_datetime = pd.to_datetime(str(post_date) + " " + str(post_time))

        # 2. Định nghĩa múi giờ Asia/Ho_Chi_Minh (GMT+7)
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')

        # 3. Gán múi giờ cho đối tượng datetime
        post_datetime = vietnam_tz.localize(naive_datetime)

        # chuẩn bị key cho session_state
        if "last_df_in" not in st.session_state:
            st.session_state["last_df_in"] = None
        if "last_anomaly" not in st.session_state:
            st.session_state["last_anomaly"] = None
        if "checked" not in st.session_state:
            st.session_state["checked"] = False

        if st.button("Kiểm tra"):
            df_in = pd.DataFrame([{
                "title": title,
                "description": description,
                "address": address,
                "brand": brand,
                "model": model_name,
                "bike_type": bike_type,
                "origin": origin,
                "engine_capacity": engine_capacity,
                "registration_year": registration_year,
                "mileage_km": mileage_km,
                "min_price": min_price,
                "max_price": max_price,
                "price": price
            }])

            df_in["age"] = 2025 - df_in["registration_year"]
            df_in["post_time"] = post_datetime

            # Mapping using helpers
            if helpers is not None:
                if df_in.at[0, 'brand'] in helpers['rare_brands']:
                    df_in['brand_grouped'] = 'Hãng khác'
                else:
                    df_in['brand_grouped'] = df_in['brand']

                rare_models = helpers['model_group_maps'].get(df_in.at[0, 'brand_grouped'], set())
                if df_in.at[0, 'model'] in rare_models:
                    df_in['model_grouped'] = 'Dòng khác'
                else:
                    df_in['model_grouped'] = df_in['model']

                df_in["segment"] = df_in["brand_grouped"] + "_" + df_in["model_grouped"]
                df_in["brand_meanprice"] = helpers["brand_mean_map"].get(df_in.at[0,"brand"], np.nan)
            else:
                df_in["brand_grouped"] = df_in["brand"]
                df_in["model_grouped"] = df_in["model"]
                df_in["segment"] = df_in["brand"] + "_" + df_in["model"]
                df_in["brand_meanprice"] = np.nan

            try:
                df_all, anomaly = detect_outliers(df_in, model_path=MODEL_PATH, input_is_df=True, helpers=helpers)

                # lưu tạm vào session để dùng sau khi user xác nhận
                st.session_state["last_df_in"] = df_in
                st.session_state["last_anomaly"] = anomaly
                st.session_state["checked"] = True

            except Exception as e:
                st.exception(e)

        # Nếu đã có kết quả kiểm tra trong session_state thì hiển thị
        if st.session_state.get("checked", False):
            df_in = st.session_state["last_df_in"]
            anomaly = st.session_state["last_anomaly"]

            if anomaly is None:
                st.info("Không có kết quả kiểm tra.")
            else:
                if len(anomaly) > 0:
                    # xác định reason dựa trên score như yêu cầu (model/business)
                    # note: detect_outliers đã tính score_model_based, score_business_based
                    r = []

                    price = anomaly["price"].iloc[0]
                    resid = anomaly["resid"].iloc[0]
                    p10 = anomaly["p10"].iloc[0]
                    p90 = anomaly["p90"].iloc[0]

                    # Tính giá mô hình dự đoán
                    predicted_price = price - resid
                    if predicted_price > 0:
                        diff_pct = resid / predicted_price * 100
                    else:
                        diff_pct = None


                    # ===================================================
                    # 1) LÝ DO DỰA TRÊN ĐIỂM MÔ HÌNH (score_model_based)
                    # ===================================================
                    # if anomaly["score_model_based"].iloc[0] >= 50:
                    #     r.append("Mô hình đánh giá xe có dấu hiệu bất thường")

                    # 1.1) Residual Z-score – giá lệch xa mô hình dự đoán
                    if anomaly["flag_resid"].iloc[0] == 1:
                        if diff_pct is not None:
                            if resid > 0:
                                r.append(
                                    f"Giá đang CAO hơn mức mô hình dự đoán khoảng {diff_pct:.1f}%"
                                )
                            else:
                                r.append(
                                    f"Giá đang THẤP hơn mức mô hình dự đoán khoảng {abs(diff_pct):.1f}%"
                                )
                        else:
                            r.append("Giá lệch quá xa mô hình dự đoán")

                    # 1.2) Giá nằm ngoài khoảng Min–Max
                    if anomaly["flag_minmax"].iloc[0] == 1:
                        r.append("Giá nằm ngoài khoảng giá hợp lý (Min–Max)")

                    # 1.3) Giá nằm ngoài phân vị P10–P90
                    if anomaly["flag_p10p90"].iloc[0] == 1:
                        if price < p10:
                            r.append("Giá thuộc nhóm 10% THẤP NHẤT của phân khúc (rẻ bất thường)")
                        elif price > p90:
                            r.append("Giá thuộc nhóm 10% CAO NHẤT của phân khúc (cao bất thường)")
                        else:
                            r.append("Giá nằm ngoài khoảng P10–P90 của phân khúc")

                    # 1.4) Bất thường từ mô hình không giám sát (Isolation Forest, LOF, KMeans)
                    if anomaly["flag_unsup"].iloc[0] == 1:
                        r.append("Mô hình học máy không giám sát phát hiện điểm bất thường")


                    # ===================================================
                    # 2) LÝ DO THEO LOGIC NGHIỆP VỤ (score_business_based)
                    # ===================================================
                    if anomaly["flag_mileage_low"].iloc[0] == 1:
                        r.append("Số km đã đi THẤP bất thường so với tuổi xe")

                    if anomaly["flag_mileage_high"].iloc[0] == 1:
                        r.append("Số km đã đi CAO bất thường so với tuổi xe")


                    # ===================================================
                    # 3) XỬ LÝ KẾT QUẢ CUỐI
                    # ===================================================
                    # reason_text = " + ".join(r) if r else "Không xác định nguyên nhân"

                    st.error("🚨 Hệ thống phát hiện bài đăng có dấu hiệu BẤT THƯỜNG")

                    if r:
                        st.markdown(
                            "**Nguyên nhân chi tiết:**\n"
                            + "\n".join([f"- {reason}" for reason in r])
                        )
                    else:
                        st.markdown("Không xác định được nguyên nhân.")
                    # st.dataframe(anomaly)

                     # ======================================
                    # CLUSTER + THÔNG BÁO CHI TIẾT
                    # ======================================

                    # ======================== TÍNH TOÁN ========================

                    try:
                        # Lấy đúng scaler và model cluster
                        scaler = helpers["cluster_scaler"]
                        kmeans = helpers["cluster_model"]
                        cluster_summary = helpers["cluster_summary"]

                        # Tạo log_price nếu chưa có
                        if "log_price" not in df_in.columns:
                            df_in["log_price"] = np.log1p(df_in["price"])

                        # Chuẩn hoá như lúc train
                        X_cluster = df_in[["age","mileage_km","min_price","max_price","log_price"]]
                        X_cluster_scaled = scaler.transform(X_cluster)

                        # Dự đoán cụm
                        cluster_label = int(kmeans.predict(X_cluster_scaled)[0])

                        # Lấy giá trung bình cụm
                        # Nếu summary là DataFrame
                        if hasattr(cluster_summary, "index"):
                            if cluster_label in cluster_summary.index:
                                cluster_mean_price = cluster_summary.loc[cluster_label, "avg_price"]
                            else:
                                cluster_mean_price = None
                        else:
                            # Nếu summary là dict
                            if cluster_label in cluster_summary:
                                cluster_mean_price = cluster_summary[cluster_label]["avg_price"]
                            else:
                                cluster_mean_price = None

                        price = df_in["price"].iloc[0]
                        if cluster_mean_price and cluster_mean_price > 0:
                            diff_pct = (price - cluster_mean_price) / cluster_mean_price * 100
                            diff_vnd = price - cluster_mean_price
                        else:
                            diff_pct = None
                            diff_vnd = None

                    except Exception as e:
                        st.error(f"Cluster error: {e}")
                        cluster_label = None
                        diff_pct = None
                        diff_vnd = None
                        cluster_mean_price = None



                    # ======================== HIỂN THỊ KẾT QUẢ ========================

                    # st.success("🎉 **Đăng tin thành công!**")
                    st.markdown("### 🔎 **Phân loại phân khúc**")

                    # Tên cụm dễ hiểu
                    if cluster_label is not None:
                        st.markdown(f"- **Xe của bạn** thuộc **{cluster_names[cluster_label]}**")

                    # Hiển thị card mô tả cụm
                    if cluster_label is not None:
                        st.markdown(cluster_cards[cluster_label], unsafe_allow_html=True)

                    # Chênh lệch giá so với trung bình cụm
                    if diff_pct is not None:
                        if diff_vnd >= 0:
                            st.markdown(
                                f"- **Giá cao hơn trung bình phân khúc** {diff_pct:.1f}% (**+{diff_vnd:,.0f} VND**)"
                            )
                        else:
                            st.markdown(
                                f"- **Giá thấp hơn trung bình phân khúc** {abs(diff_pct):.1f}% (**{diff_vnd:,.0f} VND**)"
                            )
                    else:
                        st.markdown("- Không có dữ liệu trung bình cụm để so sánh.")

                    # hỏi user: có muốn đăng không? + nút xác nhận lưu
                    choice = st.radio("Xe này bất thường, bạn vẫn muốn đăng tin không?", ["Không", "Có"], horizontal=True, key="confirm_post_radio")

                    if st.button("Xác nhận"):
                        if choice == "Có":

                            st.info("📌 Tin đã được lưu vào hệ thống.")

                            save_user_request(df_in, cluster_names[cluster_label])

                else:
                    st.success("Thông tin đăng hợp lệ.")

                    # ======================================
                    # CLUSTER + THÔNG BÁO CHI TIẾT
                    # ======================================

                    # ======================== TÍNH TOÁN ========================

                    try:
                        # Lấy đúng scaler và model cluster
                        scaler = helpers["cluster_scaler"]
                        kmeans = helpers["cluster_model"]
                        cluster_summary = helpers["cluster_summary"]

                        # Tạo log_price nếu chưa có
                        if "log_price" not in df_in.columns:
                            df_in["log_price"] = np.log1p(df_in["price"])

                        # Chuẩn hoá như lúc train
                        X_cluster = df_in[["age","mileage_km","min_price","max_price","log_price"]]
                        X_cluster_scaled = scaler.transform(X_cluster)

                        # Dự đoán cụm
                        cluster_label = int(kmeans.predict(X_cluster_scaled)[0])

                        # Lấy giá trung bình cụm
                        # Nếu summary là DataFrame
                        if hasattr(cluster_summary, "index"):
                            if cluster_label in cluster_summary.index:
                                cluster_mean_price = cluster_summary.loc[cluster_label, "avg_price"]
                            else:
                                cluster_mean_price = None
                        else:
                            # Nếu summary là dict
                            if cluster_label in cluster_summary:
                                cluster_mean_price = cluster_summary[cluster_label]["avg_price"]
                            else:
                                cluster_mean_price = None

                        price = df_in["price"].iloc[0]
                        if cluster_mean_price and cluster_mean_price > 0:
                            diff_pct = (price - cluster_mean_price) / cluster_mean_price * 100
                            diff_vnd = price - cluster_mean_price
                        else:
                            diff_pct = None
                            diff_vnd = None

                    except Exception as e:
                        st.error(f"Cluster error: {e}")
                        cluster_label = None
                        diff_pct = None
                        diff_vnd = None
                        cluster_mean_price = None



                    # ======================== HIỂN THỊ KẾT QUẢ ========================

                    # st.success("🎉 **Đăng tin thành công!**")
                    st.markdown("### 🔎 **Phân loại phân khúc**")

                    # Tên cụm dễ hiểu
                    if cluster_label is not None:
                        st.markdown(f"- **Xe của bạn** thuộc **{cluster_names[cluster_label]}**")

                    # Hiển thị card mô tả cụm
                    if cluster_label is not None:
                        st.markdown(cluster_cards[cluster_label], unsafe_allow_html=True)

                    # Chênh lệch giá so với trung bình cụm
                    if diff_pct is not None:
                        if diff_vnd >= 0:
                            st.markdown(
                                f"- **Giá cao hơn trung bình phân khúc** {diff_pct:.1f}% (**+{diff_vnd:,.0f} VND**)"
                            )
                        else:
                            st.markdown(
                                f"- **Giá thấp hơn trung bình phân khúc** {abs(diff_pct):.1f}% (**{diff_vnd:,.0f} VND**)"
                            )
                    else:
                        st.markdown("- Không có dữ liệu trung bình cụm để so sánh.")

                    # Show nút lưu nếu user muốn (optional) — tự lưu hoặc cho user bấm
                    if st.button("Đăng tin"):


                        st.info("📌 Tin đã được lưu vào hệ thống.")

                        save_user_request(df_in,cluster_names[cluster_label])


    # ======================================
    # 2) TAB ADMIN 
    # ======================================
    with tab_admin:

        st.subheader("Chế độ kiểm tra dành cho Admin")

        mode_admin = st.radio(
            "Chọn cách kiểm tra:",
            ["Dữ liệu user nhập hôm nay", "Upload file"],
            horizontal=True
        )
        # ============================================================
        # MODE 1: KIỂM TRA DỮ LIỆU USER NHẬP HÔM NAY
        # ============================================================
        # =========================
        # DUYỆT TIN USER SUBMISSIONS
        # =========================
        if mode_admin == "Dữ liệu user nhập hôm nay":

            # === DUYỆT TIN: DỮ LIỆU USER NHẬP HÔM NAY ===
            save_path = "user_submissions.xlsx"
            system_path = "data_motobikes_realtime.xlsx"
        
            # mapping từ header submissions (VN) -> hệ thống (EN)
            column_map = {
                'Tiêu đề': 'title',
                'Địa chỉ': 'address',
                'Mô tả chi tiết': 'description',
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
                'Trọng lượng': 'weight',
                'is_outlier' : 'is_outlier',
                'cluster_label' : 'cluster_label'
            }
            save_path = "user_submissions.xlsx"


            st.subheader("Danh sách tin user đã gửi")

            if os.path.exists(save_path):
                df_user = pd.read_excel(save_path)

                cols_to_hide = ["brand_grouped", "model_grouped", "segment", "brand_meanprice","tmp_id"]
                df_user_display = df_user.drop(columns=[c for c in cols_to_hide if c in df_user.columns])

                st.dataframe(df_user_display.sort_values(by='post_time', ascending=False))

                # --- nút: chạy kiểm tra bất thường ---
                # --- nút: chạy kiểm tra bất thường ---
                if st.button("Chạy kiểm tra bất thường (User submissions)"):
                    try:
                        # --- BƯỚC 0: đảm bảo df_user là bản copy và index ổn định ---
                        df_user = df_user.copy()  # tránh side-effect với biến gốc ngoài scope
                        # reset index để index = 0..N-1 (đảm bảo mapping theo vị trí)
                        df_user = df_user.reset_index(drop=True)

                        # --- BƯỚC 1: đảm bảo tmp_id tồn tại và không null (tmp_id = index) ---
                        # Lưu ý: tmp_id là "temporary id" dùng để mapping — form 0..N-1 (match index)
                        df_user["tmp_id"] = df_user.index.astype(int)

                        # --- BƯỚC 2: đảm bảo is_outlier mặc định = 0 ---
                        df_user["is_outlier"] = 0

                        # (tùy: lưu tạm tmp_id vào file để persist nếu muốn)
                        # df_user.to_excel(save_path, index=False)

                        # --- BƯỚC 3: gọi hàm detect_outliers (nó nên chấp nhận df có tmp_id) ---
                        df_all, anomaly = detect_outliers(
                            df_user,
                            model_path=MODEL_PATH,
                            input_is_df=True,
                            helpers=helpers
                        )

                        # bảo đảm anomaly là DataFrame (tránh None)
                        if anomaly is None:
                            anomaly = df_all.iloc[0:0].copy()

                        # --- BƯỚC 4: nếu anomaly không có tmp_id, sinh tmp_id từ index tương ứng ---
                        if "tmp_id" not in anomaly.columns:
                            # reset index of anomaly to ensure it's aligned with df_all/df_user positions
                            anomaly = anomaly.reset_index(drop=False)  # keep old index in column 'index' if needed
                            # nếu anomaly came from df_all where indices align with df_user after reset, then:
                            try:
                                # Try to take tmp_id from df_user using anomaly.index (prefer original index before reset)
                                # If anomaly.index aligns with df_user.index:
                                anomaly["tmp_id"] = df_user.loc[anomaly.index, "tmp_id"].values
                            except Exception:
                                # Fallback robust method: if anomaly has a column 'index' (from reset_index), use that
                                if 'index' in anomaly.columns:
                                    try:
                                        anomaly["tmp_id"] = df_user.loc[anomaly["index"].astype(int), "tmp_id"].values
                                    except Exception:
                                        # If still fails, create tmp_id from anomaly position (last-resort)
                                        anomaly = anomaly.reset_index(drop=True)
                                        anomaly["tmp_id"] = anomaly.index.astype(int)
                                        st.warning("⚠ Không thể map anomaly index trực tiếp tới df_user; đã gán tmp_id tạm theo vị trí anomaly (khả năng mapping sai).")
                                else:
                                    anomaly = anomaly.reset_index(drop=True)
                                    anomaly["tmp_id"] = anomaly.index.astype(int)
                                    st.warning("⚠ Không thể map anomaly index trực tiếp tới df_user; đã gán tmp_id tạm theo vị trí anomaly (khả năng mapping sai).")

                        # --- BƯỚC 5: map anomaly -> df_user để set is_outlier = 1 chính xác ---
                        try:
                            # preferred: map bằng tmp_id (anomaly["tmp_id"] chứa tmp_id tương ứng của df_user)
                            matched_tmp = set(anomaly["tmp_id"].astype(int).tolist())
                            df_user["is_outlier"] = df_user["tmp_id"].apply(lambda x: 1 if int(x) in matched_tmp else 0)
                        except Exception:
                            # fallback: thử map bằng index nếu tmp_id mapping fail
                            try:
                                df_user["is_outlier"] = 0
                                df_user.loc[anomaly.index, "is_outlier"] = 1
                                st.warning("⚠ Đã dùng fallback mapping theo index để đánh dấu is_outlier.")
                            except Exception:
                                df_user["is_outlier"] = 0
                                st.warning("⚠ Không thể map anomaly để gán is_outlier — tất cả giữ 0.")

                        # --- BƯỚC 6: lưu lại file submissions (persist tmp_id + is_outlier) ---
                        df_user.to_excel(save_path, index=False)

                        st.success(f"Phát hiện {len(anomaly)} tin bất thường")

                        # hiển thị anomaly rút gọn (bỏ các cột nội bộ nếu có)
                        cols_drop = [
                            'brand_grouped','model_grouped','segment','brand_meanprice',
                            'price_hat','resid','resid_median','resid_std','resid_z','flag_resid',
                            'p10','p90','log_price'
                        ]
                        st.dataframe(anomaly.drop(columns=[c for c in cols_drop if c in anomaly.columns], errors='ignore').head(20))

                    except Exception as e:
                        st.error("❌ Lỗi khi chạy kiểm tra bất thường")
                        st.exception(e)


                        # === BẮT ĐẦU THÊM NÚT TẢI XUỐNG ===
                        if len(anomaly) > 0:
                            # 1. Tạo tên file có ngày giờ
                            now = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_name = f"anomaly_detection_user_{now}.csv"
                            
                            # 2. Chuyển DataFrame sang CSV
                            # Loại bỏ múi giờ khỏi cột 'post_time' trước khi tải xuống nếu cần (đảm bảo không lỗi)
                            df_output = anomaly.copy()
                            if 'post_time' in df_output.columns and df_output['post_time'].dt.tz is not None:
                                df_output['post_time'] = df_output['post_time'].dt.tz_localize(None)

                            csv = df_output.to_csv(index=False).encode('utf-8')
                            
                            # 3. Tạo nút tải xuống
                            st.download_button(
                                label="Tải kết quả bất thường (CSV)",
                                data=csv,
                                file_name=file_name,
                                mime='text/csv'
                            )
                        # === KẾT THÚC THÊM NÚT TẢI XUỐNG ===

                    except Exception as e:
                        st.exception(e)

            else:
                st.info("⚠ Chưa có user nào gửi dữ liệu.")

            # DUYỆT

            # load submissions & hệ thống (an empty DF nếu chưa có)
            df_user = pd.read_excel(save_path) if os.path.exists(save_path) else pd.DataFrame()
            df_system = pd.read_excel(system_path) if os.path.exists(system_path) else pd.DataFrame(columns=list(column_map.values()) + ["id"])

            st.markdown("### 📝 Duyệt tin của user")

            if df_user.empty:
                st.info("Hiện chưa có tin từ user.")
            else:
                df_user = df_user.copy()

                # ---- tmp_id để tracking (tạm thời, persistent trong file cho mapping detect_outliers) ----
                if "tmp_id" not in df_user.columns:
                    df_user.insert(0, "tmp_id", range(1, len(df_user) + 1))
                else:
                    # đảm bảo tmp_id liên tục nếu cần
                    df_user["tmp_id"] = range(1, len(df_user) + 1)

                # init cột approved/is_outlier nếu chưa có
                if "approved" not in df_user.columns:
                    df_user["approved"] = False
                if "is_outlier" not in df_user.columns:
                    df_user["is_outlier"] = 0

                # --- thống kê chờ duyệt ---
                df_pending = df_user[df_user["approved"] == False].copy()
                num_total = len(df_pending)
                num_outlier = int(df_pending["is_outlier"].sum()) if "is_outlier" in df_pending.columns else 0
                num_normal = num_total - num_outlier
                st.info(f"📌 Tổng {num_total} tin chưa duyệt: **{num_normal} tin hợp lệ**, **{num_outlier} tin bất thường**.")

                # --- option hiển thị (chỉ 1 được chọn) ---
                option = st.radio(
                    "Chế độ hiển thị (chỉ chọn 1):",
                    ["Chỉ hiện tin hợp lệ", "Chỉ hiện tin bất thường", "Hiển thị tất cả"]
                )

                df_display = df_pending.copy()
                if option == "Chỉ hiện tin hợp lệ":
                    df_display = df_display[df_display["is_outlier"] == 0].copy()
                elif option == "Chỉ hiện tin bất thường":
                    df_display = df_display[df_display["is_outlier"] == 1].copy()
                # else: giữ nguyên (hiển thị tất cả)

                # Checkbox chọn duyệt tất cả trong bảng hiển thị
                select_all = st.checkbox("Chọn duyệt tất cả trong bảng hiển thị", value=False)
                df_display["duyet"] = select_all

                # Hiển thị data_editor (tmp_id được hiển thị để tracking)
                # Hiển thị bảng duyệt (ẩn approved)
                edited_df = st.data_editor(
                    df_display.drop(columns=["approved"], errors="ignore"),
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config={
                        "duyet": st.column_config.CheckboxColumn("Duyệt?", default=False),
                        "tmp_id": None  # ẩn nhưng giữ để map
                    },
                    hide_index=True
                )

                # Nếu chọn duyệt tất cả → override kết quả cuối cùng
                if select_all:
                    edited_df["duyet"] = True

                # --- Khi bấm DUYỆT ---
                # --- Khi bấm DUYỆT ---
                if st.button("✔ Duyệt và thêm vào hệ thống"):
                    try:
                        # edited_df là dataframe trả về từ st.data_editor (chứa cột 'duyet' & 'tmp_id')
                        df_selected = edited_df[edited_df["duyet"] == True].copy()

                        if df_selected.empty:
                            st.warning("⚠ Bạn chưa chọn tin nào để duyệt.")
                        else:
                            # Lấy list tmp_id đã approve (trước khi drop)
                            approved_tmp_ids = df_selected["tmp_id"].tolist()

                            # Tạo bản sao để xử lý map/append
                            df_approve_raw = df_selected.drop(columns=["duyet"], errors="ignore").copy()

                            # --- XÁC ĐỊNH MAPPING HEADER ---
                            # column_map: VN -> EN (nếu bạn có mapping khác thì cập nhật)
                            column_map_vn_to_en = {
                                'Tiêu đề': 'title',
                                'Địa chỉ': 'address',
                                'Mô tả chi tiết': 'description',
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
                                'Trọng lượng': 'weight',
                                'is_outlier': 'is_outlier',
                                'cluster_label' : 'cluster_label'
                            }
                            # reverse map (EN -> VN)
                            column_map_en_to_vn = {v:k for k,v in column_map_vn_to_en.items()}

                            # System columns (the final Excel has Vietnamese header order you specified)
                            system_cols = [
                                "id",
                                "Tiêu đề", "Giá", "Khoảng giá min", "Khoảng giá max",
                                "Địa chỉ", "Mô tả chi tiết", "Thương hiệu", "Dòng xe",
                                "Năm đăng ký", "Số Km đã đi", "Tình trạng", "Loại xe",
                                "Dung tích xe", "Xuất xứ", "Chính sách bảo hành",
                                "Trọng lượng", "Href","is_outlier","cluster_label"
                            ]

                            # Tạo target df với đúng số dòng (len(df_approve_raw))
                            df_target = pd.DataFrame(index=df_approve_raw.index, columns=system_cols)
                            # ban đầu rỗng -> sẽ gán từ df_approve_raw nếu có

                            # Có thể submissions dùng EN headers (title, price, ...) hoặc VN headers.
                            # Duyệt các cột trong df_approve_raw và map vào df_target tương ứng:
                            for col in df_approve_raw.columns:
                                # nếu cột là tiếng Anh và có map sang VN
                                if col in column_map_en_to_vn:
                                    vn_col = column_map_en_to_vn[col]
                                    if vn_col in df_target.columns:
                                        df_target[vn_col] = df_approve_raw[col].values
                                # nếu cột là tiếng Việt và cũng nằm trong system_cols
                                elif col in df_target.columns:
                                    df_target[col] = df_approve_raw[col].values
                                else:
                                    # cột khác (ví dụ: brand_grouped, segment, price_hat, tmp_id, approved, is_outlier)
                                    # nếu có 'title' hoặc 'Tiêu đề' tương đương trong tên, ưu tiên map
                                    # else: ignore / could log
                                    pass

                            # Nếu một số cột hệ thống chưa được gán, để chuỗi rỗng / NaN -> thay bằng chuỗi rỗng
                            df_target = df_target.fillna("")

                            # Tạo ID auto (theo df_system hiện tại)
                            next_id = int(df_system["id"].max()) + 1 if (not df_system.empty and "id" in df_system.columns and pd.notna(df_system["id"].max())) else 1
                            df_target["id"] = range(next_id, next_id + len(df_target))

                            # Reorder columns để 'id' là cột đầu
                            df_target = df_target[system_cols]

                            # Append vào hệ thống
                            if os.path.exists(system_path):
                                df_system_existing = pd.read_excel(system_path)
                                df_new_system = pd.concat([df_system_existing, df_target], ignore_index=True)
                            else:
                                df_new_system = df_target.copy()

                            df_new_system.to_excel(system_path, index=False)

                            # --- CẬP NHẬT TRẠNG THÁI ĐÃ DUYỆT TRONG user_submissions.xlsx ---
                            # Chuyển flag 'approved' cho những tmp_id đã duyệt
                            df_user.loc[df_user["tmp_id"].isin(approved_tmp_ids), "approved"] = True
                            # Lưu lại file submissions (persist)
                            df_user.to_excel(save_path, index=False)

                            st.success(f"🎉 Đã duyệt và thêm {len(df_target)} tin vào dữ liệu hệ thống!")

                    except Exception as e:
                        st.error("❌ Lỗi khi duyệt dữ liệu.")
                        st.exception(e)


        # ============================================================
        # MODE 2: ADMIN UPLOAD FILE KIỂM TRA
        # ============================================================
        else:
            st.subheader("Upload file để Admin kiểm tra")

            file_admin = st.file_uploader(
                "Chọn file dữ liệu cần kiểm tra (xlsx/csv)",
                type=["xlsx", "csv"],
                key="admin_upload_file"
            )

            if st.button("Chạy kiểm tra file Admin"):
                if file_admin is None:
                    st.error("Vui lòng upload file trước!")
                else:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(file_admin.name)[1]
                    ) as tmp:
                        tmp.write(file_admin.getvalue())
                        excel_path = tmp.name

                    try:
                        df_in = preprocess_motobike_data(excel_path)
                        df_all, anomaly = detect_outliers(
                            df_in, 
                            model_path=MODEL_PATH, 
                            input_is_df=True, 
                            helpers=helpers
                        )

                        st.success(
                            f"Hoàn tất kiểm tra. Tổng {len(df_in)} bản ghi — phát hiện {len(anomaly)} bất thường."
                        )
                        # st.dataframe(anomaly.head(20))
                        anomaly_print = anomaly.copy()
                        cols_to_drop = ['brand_grouped', 'model_grouped', 'segment', 'brand_meanprice','price_hat','resid','resid_median','resid_std','resid_z','flag_resid','p10','p90'
]
                        anomaly_print = anomaly_print.drop(columns=[c for c in cols_to_drop if c in anomaly_print.columns])
                        st.dataframe(anomaly_print.head(20))

                        # === BẮT ĐẦU THÊM NÚT TẢI XUỐNG ===
                        if len(anomaly) > 0:
                            # 1. Tạo tên file có ngày giờ
                            now = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_name = f"anomaly_detection_admin_{now}.csv"
                            
                            # 2. Chuyển DataFrame sang CSV
                            df_output = anomaly_print.copy()
                            # Nếu cột post_time có, hãy loại bỏ múi giờ (để tránh lỗi)
                            if 'post_time' in df_output.columns and df_output['post_time'].dt.tz is not None:
                                df_output['post_time'] = df_output['post_time'].dt.tz_localize(None)

                            csv = df_output.to_csv(index=False).encode('utf-8')
                            
                            # 3. Tạo nút tải xuống
                            st.download_button(
                                label="Tải kết quả bất thường (CSV)",
                                data=csv,
                                file_name=file_name,
                                mime='text/csv'
                            )
                        # === KẾT THÚC THÊM NÚT TẢI XUỐNG ===

                    except Exception as e:
                        st.exception(e)

    # ======================================
    # 2) TAB DASHBOARD
    # ======================================
    with tab_dash:
        st.title("📊 Dashboard Quản Lý – Motorbike Marketplace")
        st.markdown("Theo dõi trạng thái tin người dùng gửi, hiệu quả duyệt và phân khúc tin đăng real-time.")

        # =====================================
        # LOAD DATA
        # =====================================
        df_user = pd.read_excel("user_submissions.xlsx")
        df_real = pd.read_excel("data_motobikes_realtime.xlsx")

        df_user["post_time"] = pd.to_datetime(df_user["post_time"], errors="coerce")

        # Pending & approved
        df_pending = df_user[df_user["approved"] == False]
        df_approved = df_real

        # =====================================
        # KPI CARDS
        # =====================================
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📝 Tổng tin user gửi trong ngày", len(df_user))
        c2.metric("⏳ Tin chưa duyệt", len(df_pending))
        c3.metric("✅ Tin đã duyệt", len(df_approved))
        c4.metric("🌐 Tổng tin hệ thống", len(df_user) + len(df_approved))
        rate = round(len(df_approved) * 100 / (len(df_user) + len(df_approved)), 1) if (len(df_user) + len(df_approved)) > 0 else 0
        c5.metric("📈 Tỷ lệ duyệt (%)", f"{rate} %")

        st.markdown("---")

        # =====================================
        # CHART 1: Pie chart status
        # =====================================
        st.subheader("🔍 Trạng thái duyệt tin")
        fig1 = go.Figure(data=[go.Pie(
            labels=["Pending", "Approved"],
            values=[len(df_pending), len(df_approved)],
            hole=.45
        )])
        fig1.update_layout(height=300, template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

        # =====================================
        # CHART 2: Top 10 khu vực
        # =====================================
        if "Địa chỉ" in df_real.columns:
            st.subheader("📍 Top 10 khu vực có nhiều tin đăng nhất")
            df_area = df_real["Địa chỉ"].value_counts().head(10).reset_index()
            df_area.columns = ["Địa chỉ", "count"]
            fig2 = go.Figure([go.Bar(
                x=df_area["count"],
                y=df_area["Địa chỉ"],
                orientation="h",
                marker_color="skyblue"
            )])
            fig2.update_layout(height=350, template="plotly_white", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)

        # =====================================
        # CHART 3: Cluster distribution (Realtime)
        # =====================================
        if "cluster_label" in df_real.columns:
            st.subheader("🎯 Phân bố Phân khúc xe")
            cluster_count = df_real["cluster_label"].value_counts().reset_index()
            cluster_count.columns = ["cluster_label", "count"]
            fig3 = go.Figure([go.Bar(
                x=cluster_count["cluster_label"].astype(str),
                y=cluster_count["count"],
                marker_color="mediumpurple"
            )])
            fig3.update_layout(height=350, template="plotly_white", xaxis_title="Cluster", yaxis_title="Số tin")
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # =====================================
        # TABLE: Pending items
        # =====================================
        st.subheader("📌 Danh sách tin chưa duyệt")
        display_cols = ["title", "description","brand","model","price", "address", "post_time"]
        st.dataframe(df_pending[display_cols], use_container_width=True, height=350)


