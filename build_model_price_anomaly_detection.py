import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from function_preprocessing_motorbike import *
import pickle

def detect_outliers(
    data_input,
    model_path,
    input_is_df=False,
    helpers=None,
    scaler_path="unsup_scaler.pkl",
    if_path="if_model.pkl",
    lof_path="lof_model.pkl",
    km_path="kmeans_model.pkl",
    is_train=False
):
    """
    Detect anomalies cho xe máy.
    - Nếu input_is_df=False: data_input là đường dẫn file Excel/CSV.
    - Nếu input_is_df=True: data_input là DataFrame.
    - is_train=True nếu đang chạy trên dữ liệu train (dùng small cluster rule).
    """
    # 0. THRESHOLDS
    TH = {
        # Business rule (mileage)
        "min_km_per_year": 200,
        "max_km_per_year": 20000,

        # Residual threshold
        "resid_z_threshold": 3,
        "kmeans_small_ratio": 0.1,

        # Final scoring
        "score_resid": 0.4,
        "score_minmax": 0.2,
        "score_p10p90": 0.2,
        "score_unsup": 0.2,
        "score_threshold": 50
    }


    # 1. LOAD DỮ LIỆU
    if input_is_df:
        # Trường hợp nhập tay 1 xe → đã là DataFrame
        df = data_input.copy()

    else:
        # Trường hợp đọc file Excel hoặc CSV
        df = preprocess_motobike_data(data_input)

    # Load model
    with open(model_path, 'rb') as f:
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

    # seg_stats = (
    #     df.groupby('segment')['resid']
    #       .agg(['median', 'std'])
    #       .reset_index()
    #       .rename(columns={'median':'resid_median', 'std':'resid_std'})
    # )
    if helpers is not None and df['segment'].iloc[0] in helpers['seg_resid_map']:
        seg = df['segment'].iloc[0]
        df['resid_median'] = helpers['seg_resid_map'][seg]['resid_mean']
        df['resid_std']    = helpers['seg_resid_map'][seg]['resid_std']
    else:
        # fallback khi không có helpers
        seg_stats = (
                df.groupby('segment')['resid']
                .agg(['median', 'std'])
                .reset_index()
                .rename(columns={'median':'resid_median', 'std':'resid_std'})
            )
        df = df.merge(seg_stats.rename(columns={'median':'resid_median','std':'resid_std'}), on='segment')


    # df = df.merge(seg_stats, on='segment', how='left')

    df['resid_z'] = (df['resid'] - df['resid_median']) / (df['resid_std'] + 1e-9)
    df['flag_resid'] = (df['resid_z'].abs() > TH["resid_z_threshold"]).astype(int)

    # Min/Max & P10/P90 violations
    # seg_price_stats = (
    #     df.groupby('segment')['price']
    #       .quantile([0.10, 0.90])
    #       .unstack(level=1)
    #       .reset_index()
    #       .rename(columns={0.10:'p10', 0.90:'p90'})
    # )
    # df = df.merge(seg_price_stats, on='segment', how='left')

    if helpers is not None and df['segment'].iloc[0] in helpers['seg_price_map']:
        seg = df['segment'].iloc[0]
        df['p10'] = helpers['seg_price_map'][seg]['p10']
        df['p90'] = helpers['seg_price_map'][seg]['p90']
    else:
        # fallback
        seg_price_stats = (
            df.groupby('segment')['price']
              .quantile([0.10, 0.90])
              .unstack(level=1)
              .reset_index()
              .rename(columns={0.10:'p10', 0.90:'p90'})
        )
        df = df.merge(seg_price_stats, on='segment', how='left')

    df['flag_p10p90'] = ((df['price'] < df['p10']) |
                         (df['price'] > df['p90'])).astype(int)
    
    df['flag_minmax'] = ((df['price'] < df['min_price']) |
                         (df['price'] > df['max_price'])).astype(int)

    # business rule số km đã đi
    df['flag_mileage_low'] = (
        df['mileage_km'] < df['age'] * TH["min_km_per_year"]
    ).astype(int)

    df['flag_mileage_high'] = (
        df['mileage_km'] > df['age'] * TH["max_km_per_year"]
    ).astype(int)

    # Nếu 1 trong 2 rule vi phạm → flag_br = 1
    df['flag_br'] = (
        (df['flag_mileage_low'] == 1) | 
        (df['flag_mileage_high'] == 1)
    ).astype(int)

    # Unsupervised features
    unsup_feats = ['age','mileage_km','resid_z']
    X_unsup = df[unsup_feats].fillna(0).values
    scaler = pickle.load(open(scaler_path,"rb"))
    Xu = scaler.transform(X_unsup)

    # Isolation Forest
    if_model = pickle.load(open(if_path,"rb"))
    df['flag_if'] = (if_model.predict(Xu)==-1).astype(int)

    # # LOF
    # if is_train:
    #     # LOF bình thường cho train
    #     lof_model = pickle.load(open(lof_path,"rb"))  # đã train với novelty=False
    #     lof_pred = lof_model.fit_predict(Xu)
    #     df['flag_lof'] = (lof_pred==-1).astype(int)
    # else:
    #     # LOF cho mẫu mới (novelty=True)
    #     lof_model = pickle.load(open(lof_path,"rb"))  # đã train với novelty=True
    #     lof_pred = lof_model.predict(Xu)  # -1 = outlier, 1 = normal
    #     df['flag_lof'] = (lof_pred==-1).astype(int)
        
    # LOF
    lof_model = pickle.load(open(lof_path, "rb"))  # model đã train với novelty=True
    lof_pred = lof_model.predict(Xu)               # -1 = outlier, 1 = normal
    df['flag_lof'] = (lof_pred == -1).astype(int)
    
    # KMeans
    km_model = pickle.load(open(km_path,"rb"))
    cl = km_model.predict(Xu)
    centers = km_model.cluster_centers_
    dists = np.linalg.norm(Xu - centers[cl], axis=1)
    r95 = np.percentile(dists,95)
    df['flag_kmeans'] = (dists>r95).astype(int)

    # # Small cluster rule chỉ dùng khi là train
    # if is_train and helpers is not None and 'km_helper' in helpers:
    #     ratios = helpers['km_helper']['ratios']
    #     small_clusters = [k for k,v in ratios.items() if v<TH['kmeans_small_ratio']]
    #     df['flag_kmeans_small'] = [1 if k in small_clusters else 0 for k in cl]
    # else:
    #     df['flag_kmeans_small'] = 0
    

    # Combined unsupervised
    df['flag_unsup'] = (df[['flag_if','flag_lof','flag_kmeans']].sum(axis=1)>=2).astype(int)


    # Final scoring
    df['score_model_based'] = 100 * (
        TH["score_resid"]*df['flag_resid'] +
        TH["score_minmax"]*df['flag_minmax'] +
        TH["score_p10p90"]*df['flag_p10p90'] +
        TH["score_unsup"] *df['flag_unsup']
    )
    df['score_business_based'] = 100 * df['flag_br']
    df['final_score'] = df['score_model_based'] + df['score_business_based']


    df['is_outlier'] = (df['final_score'] >= TH["score_threshold"]).astype(int)

    # Xuất kết quả
    anomaly = df[df['is_outlier'] == 1].copy()
    anomaly.to_csv("outliers_detected.csv", index=False)

    print("=== Tổng thể ===")
    print(f"Tổng bản ghi: {len(df):,}")
    print(f"Số bản ghi bất thường: {len(anomaly):,} "
          f"({100*len(anomaly)/len(df):.2f}%)")
    print("Đã xuất 'outliers_detected_full.csv'.")

    return df, anomaly

# df, anomaly = detect_outliers(
#     "data_motobikes.xlsx",
#     "motobike_price_prediction_model.pkl"
# )
