import os

def load_teen_dict(filepath='teencode.txt'):
    teen_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                teen_dict[parts[0].lower()] = parts[1].lower()
    return teen_dict


def load_stopwords(filepath='vietnamese-stopwords.txt'):
    with open(filepath, 'r', encoding='utf-8') as f:
        stop_words = [line.strip() for line in f if line.strip()]

    stop_words = [w.replace('_', ' ') for w in stop_words]

    new_stop_words = [
        'bán','nốt','nhạc','zalo','inbox','giàlô','hình','ảnh','sản phẩm',
        'chay','di hoc','lái','ảnh','thích hợp','yên chí','yên tâm','thiện chí',
        'dép','đồ đạc','anhchị','cam on','cảm ơn','nhiệt tình','bà xã','vui lòng',
        'hậu tạ','nhắn','gọi','bảo hành','cập nhật','đăng kí','đăng ký',
    ]

    stop_words = stop_words + [w for w in new_stop_words if w not in stop_words]

    return stop_words
