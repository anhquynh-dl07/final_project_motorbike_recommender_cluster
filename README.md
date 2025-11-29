# Motorbike Recommendation System and Market Segmentation by Clustering

## **Giới thiệu mục tiêu**

Dự án này xây dựng hệ thống gợi ý xe máy tương tự tại Việt Nam trên Chợ Tốt dựa trên id của xe và dựa trên từ khóa tìm kiếm của người dùng.
Dữ liệu về xe máy được thu thập trên nền tảng Chợ Tốt, trước ngày 01/07/2025. Dữ liệu được làm sạch, xử lý để huấn luyện các mô hình máy học nhằm mục tiêu:

* Xây dựng hệ thống gợi ý xe máy tương tự bằng thư viện Gensim và Cosine similarity
* Phân khúc thị trường bằng thuật toán phân cụm trên 2 môi trường (sklearn và pyspark).

## **Cấu trúc dự án**
├── cau_1_gensim_cosine.ipynb            # Hệ thống gợi ý xe máy  
├── cau_2_cluster_pyspark.ipynb          # Phân khúc bằng thuật toán phân cụm trên pyspark  
├── cau_2_cluster_python.ipynb           # Phân khúc bằng thuật toán phân cụm trên pyspark  
├── data                                 # dữ liệu được cung cấp  
│   ├── data_motobikes.xlsx  
│   ├── Mô tả bộ dữ liệu Chợ Tốt.docx  
│   └── Mô tả bộ dữ liệu Chợ Tốt.pdf  
├── files  
│   ├── emojicon.txt  
│   ├── english-vnmese.txt  
│   ├── teencode.txt  
│   ├── vietnamese-stopwords.txt  
│   └── wrong-word.txt  
├── README_project_2.md                 # hướng dẫn
├── requirements.txt                    # điều kiện cài đặt
├── slide  
|    └── Project 2_Bike Recommendation System and Market Segment.pptx
└── topic_RecommenderSystem_MarketSegmentation.pdf

## **Dữ liệu**

Dữ liệu được thu thập từ trang Chợ Tốt về xe máy đã qua sử dụng trước ngày 01/07/2025.
Dữ liệu được lấy từ file "data_motobikes.xlsx"
Dữ liệu thô gồm 7208 bản ghi và 18 thuộc tính.


## **Recommendation system (hệ thống gợi ý xe tương tự)**
### Import thư viện
Import/ tải các thư viện cần thiết

### Xử lý dữ liệu  
Các bước xử lý bao gồm:
- Chuẩn hóa các cột “price”, “mileage_km”, “engine_capacity”, “registration_year”
- Tạo các feature mới:
  - "price_range" = binned "price"
  - "km_range" = binned "mileage_km"
  - "weight_class" = chia kiểu "engine_capacity" (nặng, nhẹ)
  - "age" = 2025 - "registration_year"
- Nhóm các cột biến phân loại và các cột feature vừa tạo.
- Làm sạch văn bản tiếng Việt:
  - Ép kiểu string
  - lower case
  - lọc '\n'
  - lọc chữ chỉ gồm 1 kí tự chữ
  - lọc emoji và kí tự không phải chữ
  - chuyển teencode qua tiếng Việt
  - lọc bỏ động từ bằng undersea.pos_tag
  - lọc bỏ stop_words (stop words có thể bỏ thêm sau khi quan sát)

### Tạo ma trận tương đồng 
Chuyển dữ liệu văn bản đã làm sạch thành danh sách các từ (tokens).
Tạo từ điển (dictionary) từ tất cả các token để xác định tập từ vựng.
Chuyển văn bản thành corpus dựa trên dictionary (biểu diễn dạng số).
Áp dụng TF-IDF để tính trọng số cho từng từ trong corpus.
Tính ma trận tương đồng (sparse matrix) giữa các văn bản để đánh giá mức độ giống nhau.

### Trường hợp 1: gợi ý xe theo id sản phẩm được chọn
Người dùng chọn một xe (biết được “id”)
Dựa trên ma trận tương đồng, tìm các xe có similarity score cao nhất.
Lấy ra 5 xe gợi ý tương tự, bỏ chính xe đã chọn.

### Trường hợp 2: gợi ý xe theo cụm từ khóa tìm kiểm
(vd: “honda vision xanh dưới 15 triệu”)
Người dùng nhập từ khóa tìm kiếm. 
Xử lý từ khóa (clean_text()), chuyển từ khóa thành vector số dựa trên từ điển và TF-IDF
Tính độ tương đồng giữa từ khóa và tất cả xe trong dữ liệu. 
Sắp xếp và lấy ra 5 xe gợi ý phù hợp nhất.

### So sánh
**Trường hợp 1**:
- Tính độ tương đồng trung bình giữa 5 mẫu gợi ý cho một mẫu, sau đó áp dụng cho 7000 mẫu trong tập dữ liệu và tính trung bình.
- So sáng độ tương đồng trung bình giữa gensim và cosine similarity. Kết quả cho thấy cosine similarity cho độ tương đồng trung bình cao hơn gensim. Các kết quả gợi ý cũng sát hơn so với gensim về mặt semantic.

**Trường hợp 2**:
- Cho danh sách 10 cụm từ khóa tìm kiếm
- Tính độ tương đồng trung bình giữa 5 mẫu gợi ý cho một mẫu, sau đó áp dụng cho 10 cụm từ trên và tính trung bình.
- So sáng độ tương đồng trung bình giữa gensim và cosine similarity. Kết quả cho thấy cosine similarity cho độ tương đồng trung bình cao hơn gensim. Các kết quả gợi ý cũng sát hơn so với gensim về mặt semantic.

### Tạo ma trận đặc trưng tổng hợp (text + numeric)  
Hệ thống sử dụng kết hợp đặc trưng văn bản và đặc trưng số:
**Text features**:  
- Tokenize văn bản đã làm sạch  
- Tạo dictionary và corpus  
- TF-IDF để biểu diễn vector văn bản  

**Numeric features**:  
- Các cột số được chuẩn hóa (StandardScaler)  
- Ghép nối vector TF-IDF với vector numeric đã scale để tạo feature vector cuối cùng của mỗi xe  
- Tính độ tương đồng bằng Cosine Similarity  
- Từ feature vector tổng hợp (text + numeric), tính ma trận cosine similarity giữa tất cả các xe.

**Cosine similarity** được chọn vì:  
- Phù hợp cho dữ liệu sparse (TF-IDF)  
- Ổn định khi ghép thêm numeric features  
- kết quả sát hơn về mặt “tính tương tự” so với Gensim dựa trên word vectors.  

**Trường hợp 1**: gợi ý xe theo id sản phẩm  
- Người dùng chọn 1 xe (biết “id”)  
- Lấy vector của xe đó → tính similarity với toàn bộ xe  
- Sắp xếp theo điểm cosine similarity giảm dần  
- Trả về 5 xe tương tự nhất, loại bỏ chính xe đang chọn  

**Trường hợp 2**: gợi ý xe theo từ khóa tìm kiếm  
- Làm sạch text như pipeline chính  
- Chuyển từ khóa thành vector TF-IDF  
- Kết hợp với vector numeric mặc định (giá trị 0)  
- Tính cosine similarity giữa query và toàn bộ xe  
- Lấy ra 5 xe phù hợp nhất theo điểm tương đồng

## **Market segmentation by Clustering (phân khúc xe máy bằng thuật toán phân cụm)**
### 1. Môi trường machine learning truyền thống (sklearn)
#### Import thư viện
Import/ tải các thư viện cần thiết

#### Xử lý dữ liệu  
Các bước xử lý dữ liệu gồm (copy lai ở bài 1)
- Loại bỏ bỏ các cột: "id","Tiêu đề","Địa chỉ", "Href", "Tình trạng", "Chính sách bảo hành", "Trọng lượng"
- Đổi tên các cột để đơn giản hoá tên các đặc trưng:
  - 'Giá': 'price',
  - 'Khoảng giá min': 'min\_price',
  - 'Khoảng giá max': 'max\_price',
  - 'Thương hiệu': 'brand',
  - 'Dòng xe': 'model',
  - 'Năm đăng ký': 'registration\_year',
  - 'Số Km đã đi': 'mileage\_km',
  - 'Loại xe': 'bike\_type',
  - 'Dung tích xe': 'engine\_capacity',
  - 'Xuất xứ': 'origin'
    
- Xoá các dòng thiếu giá trị.
- Xóa giá trị không hợp lệ và thay thế các giá trị không rõ trong 'engine\_capacity' bằng 'Unknown'.
- Loại bỏ giá trị không hợp lệ trong "origin".
- Thay thế các giá trị không rõ trong "origin" bằng "Nước khác"
- Chuẩn hóa dữ liệu "registration\_year".
- Chuẩn hóa dữ liệu "price", "max\_price", "min\_price"
- Nhóm các thương hiệu ("brand") có số giá trị không trùng lặp dưới 50 giá trị thành nhóm có giá trị "Hãng khác".
- Nhóm các dòng xe ("model") có dưới 100 giá trị không trùng lặp thành nhóm có giá trị "Dòng khác"

#### Tạo các thuộc tính mới
- Tạo thuộc tính Tuổi xe ("age") từ "Năm đăng ký" và thời điểm thu thập dữ liệu.
- Tạo thuộc tính Phân khúc ("segment") bằng cách kết hợp thương hiệu đã được nhóm ("brand\_grouped") và dòng xe đã được nhóm ("model\_grouped").

#### Xử lý biến đầu ra
- Chuyển hoá log biến "price"

#### Phân loại dữ liệu
Trong mô hình này, nhóm chỉ chọn các biến numeric liên tục (continuous numeric) cho phân cụm.  Mục tiêu của clustering ở đây là tách các xe thành các nhóm dựa trên đặc tính định lượng: giá, tuổi xe, số km đã đi, vì:

Các thuật toán phân cụm phổ biến dựa trên khoảng cách
Các mô hình phân cụm KMeans, Gaussian Mixture, Bisecting KMeans… đều tính khoảng cách (Euclidean, squared Euclidean, Mahalanobis…) giữa các điểm dữ liệu. Khoảng cách này chỉ có ý nghĩa với các biến numeric liên tục, vì phép trừ, bình phương và căn bậc hai cần giá trị đo lường liên tục.
Tránh lỗi khi dùng biến rời rạc hoặc categorical: Biến rời rạc hoặc categorical nếu dùng trực tiếp sẽ làm khoảng cách Euclidean không hợp lý:
OneHot encoding => dữ liệu sparse, tăng chiều không cần thiết.
Label encoding => khoảng cách giữa nhãn không mang ý nghĩa (Honda = 0, Yamaha = 1 => khoảng cách 1 không đại diện cho sự khác biệt thực tế).
Chuẩn hóa và cân bằng scale
Biến numeric liên tục dễ scale / normalize (StandardScaler, MinMaxScaler), đảm bảo mỗi feature cùng trọng số khi tính khoảng cách.
Categorical hoặc rời rạc khó scale tự nhiên, có thể chi phối clustering nếu không xử lý đúng.
Giữ kết quả phân cụm đơn giản, trực quan
Sử dụng numeric liên tục giúp cluster có ý nghĩa về đặc tính định lượng:
Nếu dùng biến rời rạc nhiều => cluster sẽ bị ảnh hưởng bởi cách encode, khó giải thích kết quả.

=> Phân cụm (Clustering) và hồi quy (Regression) có mục tiêu khác nhau:
Clustering chỉ sử dụng các biến numeric liên tục (giá, tuổi xe, số km đã đi) để nhóm các xe dựa trên các đặc tính định lượng. Điều này giúp khoảng cách giữa các điểm có ý nghĩa, kết quả phân nhóm trực quan và dễ giải thích.
Price Prediction (Hồi quy giá) sử dụng cả biến numeric và categorical (ví dụ: phân khúc, thương hiệu, xuất xứ) để mô hình nắm bắt tất cả các yếu tố ảnh hưởng đến giá, từ đó đưa ra dự đoán chính xác.

Do đó, việc clustering chỉ dùng một số biến numeric nhưng hồi quy dùng đầy đủ thông tin là hợp lý và không mâu thuẫn.

Các cột thông tin sử dụng để phân cụm:

num\_cols = \['age','mileage\_km','min\_price','max\_price','log_price']

#### Chuẩn hóa dữ liệu
Chuẩn hóa dữ liệu bằng Standard Scaler

### Xây dựng mô hình phân cụm
Sử dụng Kmeans, GMM, AggomerativeClustering cho môi trường ML truyền thống và Kmeans, GMM, Bisecting Kmeans cho pyspark

### Kết quả
Phân loại phân khúc xe:
1. Cụm 2 – Phân khúc Xe Cũ – Tiết Kiệm (Budget Used Motorcycles): Giá rẻ nhất, xe tuổi cao, chạy nhiều — phù hợp khách cần xe rẻ để di chuyển cơ bản.
avg_price: ~ 14.3 triệu
avg_age: ~ 14.5 năm (rất cũ)
avg_mileage: ~ 649.000 km (cực kỳ nhiều)
số lượng: 283 xe
2. Cụm 0 – Phân khúc Xe Phổ Thông – Trung cấp (Mid-range Popular Motorcycles): Xe tuổi trung bình, giá vừa phải, phù hợp đại đa số người mua
avg_price: ~ 19.7 triệu
avg_age: ~ 11.6 năm
avg_mileage: ~ 40.500 km (mileage hợp lý, thấp bất ngờ do hành vi đăng tin)
số lượng: 5.866 xe (chiếm đa số)
3. Cụm 1 – Phân khúc Xe Cao Cấp – Premium / High-end Motorcycles: Rõ ràng là các dòng SH, Vespa cao cấp, phân khối lớn, xe mới chạy ít.
avg_price: ~ 266 triệu
avg_age: ~ 7 năm (tuổi thấp hơn)
avg_mileage: ~ 33.000 km (ít)
số lượng: 846 xe

### Price prediction after cluster

Coi kết quả phân cụm như một biến mới trong model, so sánh kết quả trước và sau phân cụm: Kết quả R2 tốt hơn 1 chút so với trước khi phân cụm nhưng MAE lại cao hơn 1 chút. Nhìn chung kết quả không quá khác biệt trước và sau khi phân cụm (coi cụm như một thuộc tính mới để dự đoán giá). Điều này chứng tỏ mô hình price prediction đã sử dụng học khá tốt với các biến đã xác định.


