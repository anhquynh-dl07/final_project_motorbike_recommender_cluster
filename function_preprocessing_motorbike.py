import pandas as pd
import numpy as np

def preprocess_motobike_data(path, current_year=2025, is_inference=False):
    """
    Load + làm sạch + tạo features cho dữ liệu xe máy
    
    Parameters
    ----------
    path : str
        Đường dẫn tới file excel
    current_year : int
        Năm hiện tại để tính tuổi xe (default = 2025)
    
    Returns
    -------
    df : pandas.DataFrame
        Dữ liệu đã làm sạch & tạo features
    """
    
    # 1. Load file
    df = pd.read_excel(path)

    # 2. Xóa cột không cần thiết + rename
    cols_to_drop = ['id', 'Tiêu đề', 'Địa chỉ', 'Mô tả chi tiết', 'Href']
    df = df.drop(cols_to_drop, axis=1)

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

    # 3. Loại các cột có 1 giá trị + dòng NaN
    # df = df.dropna().reset_index(drop=True)
    
    # Nếu KHÔNG phải inference -> được phép drop toàn bộ NA (bao gồm price)
    if not is_inference:
        df = df.dropna().reset_index(drop=True)

    # Nếu là inference -> chỉ drop NA ở các cột bắt buộc, ngoại trừ 'price'
    else:
        # danh sách cột phải có để model chạy
        required_cols = [c for c in df.columns if c != "price"]

        df = df.dropna(subset=required_cols).reset_index(drop=True)

    df = df.drop(columns=['warranty_policy', 'weight'], errors='ignore')

    # # 4. Clean price
    # df['price'] = (
    #     df['price']
    #     .astype(str)
    #     .str.replace('[^0-9]', '', regex=True)
    #     .replace('', np.nan)
    #     .astype(float)
    # )

    # 5. Clean min_price / max_price
    def parse_minmax_price(s):
        if pd.isna(s): return np.nan
        s = str(s).lower().replace("tr", "").replace(" ", "")
        try:
            return float(s) * 1_000_000
        except:
            return np.nan

    df['min_price'] = df['min_price'].apply(parse_minmax_price)
    df['max_price'] = df['max_price'].apply(parse_minmax_price)

    # # Loại price = 0
    # df = df[df['price'] != 0]

    # 6. Xóa cột condition (1 giá trị)
    df = df.drop(columns=['condition'], errors='ignore')

    # 7. Clean engine_capacity
    df = df[~df['engine_capacity'].astype(str).str.contains('Nhật Bản', case=False, na=False)]
    df['engine_capacity'] = df['engine_capacity'].replace(
        ['Không biết rõ', 'Đang cập nhật'], 'Unknown'
    )

    # 8. Clean origin
    df = df[~df['origin'].astype(str).str.contains('Bảo hành hãng', case=False, na=False)]
    df['origin'] = df['origin'].replace(
        ['Đang cập nhật', 'Nước khác'], 'Nước khác'
    )

    # 9. Chuẩn hóa registration_year
    df['registration_year'] = (
        df['registration_year'].astype(str)
        .str.lower()
        .str.replace('trước năm', '1980', regex=False)
        .str.extract('(\d{4})')[0]
    )
    df['registration_year'] = pd.to_numeric(df['registration_year'], errors='coerce')
    df.loc[(df['registration_year'] < 1980) | (df['registration_year'] > current_year), 
           'registration_year'] = np.nan

    # 10. Tính tuổi xe
    df['age'] = current_year - df['registration_year']

    # 11. Group rare brand
    brand_counts = df['brand'].value_counts()
    rare_brands = brand_counts[brand_counts < 50].index
    df['brand_grouped'] = df['brand'].replace(rare_brands, 'Hãng khác')

    # 12. Group rare model theo từng brand_grouped
    def group_model(x):
        counts = x.value_counts()
        rare = counts[counts < 100].index
        return x.replace(rare, 'Dòng khác')

    df['model_grouped'] = df.groupby('brand_grouped')['model'].transform(group_model)

    # 13. Segment feature
    df['segment'] = df['brand_grouped'] + '_' + df['model_grouped']

    # # 14. Log price
    # df['log_price'] = np.log1p(df['price'])
    
    # # 15. Brand-level mean price
    # brand_mean_log = df.groupby('brand')['log_price'].mean().rename('brand_meanprice')
    # df = df.merge(brand_mean_log, on='brand', how='left')

    # # Drop duplicates
    # df = df.drop_duplicates().reset_index(drop=True)

    # return df

    # -------------------------
    # ❗CHỈ TÍNH log_price + brand_meanprice KHI TRAIN
    # -------------------------
    if not is_inference:
        df['price'] = (
            df['price'].astype(str).str.replace('[^0-9]', '', regex=True).replace('', np.nan).astype(float)
        )
        df = df[df['price'] != 0]

        df['log_price'] = np.log1p(df['price'])
        brand_mean_log = df.groupby('brand')['log_price'].mean().rename('brand_meanprice')
        df = df.merge(brand_mean_log, on='brand', how='left')
    else:
        # ❗KHI DỰ ĐOÁN: brand_meanprice = trung bình của toàn tập train (loader từ helpers)
        df['brand_meanprice'] = np.nan  # sẽ được fill sau bởi helpers

    return df
