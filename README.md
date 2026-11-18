# Phân tích đặc trưng âm thanh và đề xuất bài hát tương đồng

**Môn:** Tính toán Đa phương tiện (CS232) · Nhóm R11  
**Dataset:** MTG-Jamendo `autotagging_moodtheme/audio-low` (tập con 1.020 bài)

---

## Giới thiệu

Hệ thống nhận vào một file nhạc bất kỳ, trích đặc trưng âm thanh, rồi trả về **top-k bài "nghe giống nhất"** trong thư viện kèm lý do vì sao giống.

Trọng tâm là **xử lý tín hiệu âm thanh**: từ waveform → STFT → mel-spectrogram → vector đặc trưng 25 chiều → cosine similarity.

```python
# Ví dụ sử dụng (trong notebook 02)
results = recommend("path/to/song.mp3", k=5)
```

```
     track_id     mood      genre  similarity     nghe giống về
track_1227101     buồn       jazz       0.601  rhythm & spectral
track_0031209  bình yên  electronic    0.596  energy & spectral
track_1083509  bình yên       jazz       0.571  spectral & rhythm
```

---

## Kết quả chính

### Phân loại mood (4 nhóm: vui / bình yên / buồn / căng-tối)

| Mô hình | macro-F1 | macro-recall |
|---|---|---|
| Đoán mò | 0.249 | 0.255 |
| Chỉ dùng tag thể loại | 0.253 | 0.316 |
| **LogReg âm thanh** (theo artist) | **0.346** | **0.360** |
| **Random Forest âm thanh** (theo artist) | **0.352** | **0.387** |
| LogReg âm thanh (chia ngẫu nhiên — rò rỉ) | 0.417 | 0.427 |

> Chia theo artist để tránh rò rỉ dữ liệu: nếu chia ngẫu nhiên, điểm cao ảo thêm ~+0.065 macro-recall.

### Ablation nhóm đặc trưng

| Tổ hợp | Chiều | macro-F1 | macro-recall |
|---|---|---|---|
| Tất cả 93 chiều | 93 | 0.346 | 0.360 |
| Bỏ MFCC | 53 | 0.371 | 0.386 |
| **spectral + rhythm** *(tốt nhất)* | **25** | **0.393** | **0.407** |
| spectral + rhythm + energy | 29 | 0.392 | 0.407 |
| Chỉ spectral | 22 | 0.355 | 0.367 |

**Phát hiện:** MFCC (đường bao phổ / âm sắc) gây nhiễu cho task mood — bỏ đi điểm tăng. Spectral + rhythm (25 chiều) là tổ hợp tốt nhất.

### Phân tích hai trục mood

| Trục | Recall của mô hình | Đoán mò |
|---|---|---|
| Năng lượng (arousal: cao/thấp) | **0.655** | 0.5 |
| Cảm xúc (valence: dương/âm) | **0.560** | 0.5 |

---

## Cấu trúc thư mục

```
CS232.R11/
├── notebooks/
│   ├── 01_explore_data.ipynb        # Khám phá metadata + hiểu âm thanh số
│   └── 02_features_baseline.ipynb   # Trích đặc trưng, phân loại mood, recommend()
├── data_jamendo/
│   ├── audio-low/                   # File MP3 (tar 00–09, 1.991 bài)
│   ├── metadata/
│   │   ├── autotagging_moodtheme.tsv
│   │   ├── autotagging_genre.tsv
│   │   └── splits/split-0/          # Train/val/test split chính thức (chia theo artist)
│   ├── index_mood.csv               # 1.020 bài sạch mood, có audio
│   └── features_mood.csv            # Cache 93 đặc trưng (tạo bởi notebook 02)
├── src/                             # Hàm ổn định sẽ chuyển vào đây
├── requirements.txt
└── TOM_TAT.md                       # Nhật ký tiến độ chi tiết
```

---

## Cài đặt và chạy

### 1. Cài thư viện

```bash
pip install -r requirements.txt
```

### 2. Tải dataset

Tải metadata từ repo chính thức:
```
https://github.com/MTG/mtg-jamendo-dataset
```
Cần 4 file: `autotagging_moodtheme.tsv`, `autotagging_genre.tsv`, và splits trong `splits/split-0/`.

Tải audio (bản `audio-low`, mỗi tar ~1 GB):
```bash
# Thay NN = 00, 01, ..., 09
wget https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/audio-low/autotagging_moodtheme_audio-low-NN.tar
```
Giải nén vào `data_jamendo/audio-low/`.

### 3. Chạy notebook theo thứ tự

```
01_explore_data.ipynb     →  khám phá metadata, tạo index_mood.csv
02_features_baseline.ipynb →  trích đặc trưng, phân loại, recommend()
```

---

## Quyết định thiết kế chính

| Quyết định | Lý do |
|---|---|
| 4 nhóm mood theo valence × arousal | Có căn cứ lý thuyết, đủ bài mỗi nhóm, dễ giải thích |
| Chỉ giữ bài thuộc đúng 1 nhóm | Nhãn mơ hồ làm nhiễu mô hình |
| Chia train/test theo artist | Cùng artist → giọng/nhạc cụ giống → rò rỉ nếu chia ngẫu nhiên |
| 30 giây ở giữa bài | Phần đầu/cuối thường là intro/outro, không đại diện |
| Mono 22.050 Hz | Chuẩn cho audio ML; đủ dải tần 11 kHz |
| Tổ hợp đặc trưng: spectral + rhythm | Ablation cho thấy MFCC gây nhiễu cho task mood |
| Đánh giá bằng macro-F1 / macro-recall | Tránh bị nhóm đông (vui: 378 bài) kéo lên; nhóm nhỏ nhất 127 bài |

---

## Hạn chế

- **Dữ liệu nhỏ:** 1.020 bài, nhóm căng-tối chỉ 127 bài → kết quả dao động nhiều giữa các fold
- **Nhãn nhiễu:** mood tag do người dùng gán, không đo đạc khách quan
- **Mood ≈ thể loại:** nhóm bình yên ≈ ambient, nhóm vui ≈ pop/electronic → khó tách biệt hoàn toàn

---

## Tài liệu tham khảo

- MTG-Jamendo Dataset: [github.com/MTG/mtg-jamendo-dataset](https://github.com/MTG/mtg-jamendo-dataset)
- librosa: [librosa.org](https://librosa.org)
- Nhật ký tiến độ chi tiết: [TOM_TAT.md](TOM_TAT.md)
