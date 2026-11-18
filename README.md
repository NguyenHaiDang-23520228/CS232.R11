# Phân tích đặc trưng âm thanh và đề xuất bài hát tương đồng

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![librosa](https://img.shields.io/badge/librosa-0.10-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Dataset](https://img.shields.io/badge/Dataset-MTG--Jamendo-purple)

**Môn:** Tính toán Đa phương tiện (CS232) · Nhóm R11  
**Dataset:** MTG-Jamendo `autotagging_moodtheme/audio-low` · 1.020 bài

---

## Giới thiệu

Hệ thống nhận vào một file nhạc bất kỳ, trích đặc trưng âm thanh, rồi trả về **top-k bài "nghe giống nhất"** trong thư viện kèm lý do vì sao giống.

Trọng tâm là **xử lý tín hiệu âm thanh**:

```
waveform  →  STFT  →  mel-spectrogram  →  vector 25 chiều  →  cosine similarity
```

```python
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

### Phân loại mood — 4 nhóm (vui / bình yên / buồn / căng-tối)

| Mô hình | macro-F1 | macro-recall |
|---|:---:|:---:|
| Đoán mò | 0.249 | 0.255 |
| Chỉ dùng tag thể loại | 0.253 | 0.316 |
| LogReg âm thanh (theo artist) | 0.346 | 0.360 |
| **Random Forest âm thanh** (theo artist) | **0.352** | **0.387** |
| LogReg âm thanh (chia ngẫu nhiên — rò rỉ) | 0.417 | 0.427 |

> Chia theo artist để tránh rò rỉ: nếu chia ngẫu nhiên, điểm cao ảo thêm ~+0.065 macro-recall.

### Ablation nhóm đặc trưng

| Tổ hợp đặc trưng | Số chiều | macro-F1 | macro-recall |
|---|:---:|:---:|:---:|
| Tất cả 93 chiều | 93 | 0.346 | 0.360 |
| Bỏ MFCC | 53 | 0.371 | 0.386 |
| **spectral + rhythm** ✦ tốt nhất | **25** | **0.393** | **0.407** |
| spectral + rhythm + energy | 29 | 0.392 | 0.407 |
| Chỉ spectral | 22 | 0.355 | 0.367 |

**Phát hiện:** MFCC (đường bao phổ / âm sắc) gây nhiễu cho task mood — bỏ đi điểm tăng +0.025 F1.

### Phân tích hai trục mood

| Trục | Recall | Random |
|---|:---:|:---:|
| Năng lượng (arousal: cao / thấp) | **0.655** | 0.5 |
| Cảm xúc (valence: dương / âm) | **0.560** | 0.5 |

---

## Cấu trúc thư mục

```
CS232.R11/
├── notebooks/
│   ├── 01_explore_data.ipynb         # Khám phá metadata, phân nhóm mood
│   └── 02_features_baseline.ipynb    # Trích đặc trưng, phân loại, recommend()
├── data_jamendo/
│   ├── audio-low/                    # File MP3 (tar 00–09, 1.991 bài) — không commit
│   ├── metadata/
│   │   ├── autotagging_moodtheme.tsv
│   │   ├── autotagging_genre.tsv
│   │   └── splits/split-0/           # Official artist-split
│   └── index_mood.csv                # 1.020 bài sạch mood, có audio
├── src/
├── requirements.txt
└── README.md
```

---

## Cài đặt và chạy

```bash
pip install -r requirements.txt
```

Tải dataset từ [MTG-Jamendo](https://github.com/MTG/mtg-jamendo-dataset):

```bash
# metadata (4 file TSV + splits)
# audio — thay NN = 00..09, mỗi tar ~1 GB
wget https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/audio-low/autotagging_moodtheme_audio-low-NN.tar
```

Chạy notebook theo thứ tự:

```
01_explore_data.ipynb      →  tạo index_mood.csv
02_features_baseline.ipynb →  trích đặc trưng, train model, chạy recommend()
```

---

## Quyết định thiết kế

| Quyết định | Lý do |
|---|---|
| 4 nhóm mood theo valence × arousal | Căn cứ lý thuyết, đủ bài mỗi nhóm |
| Chỉ giữ bài thuộc đúng 1 nhóm | Nhãn mơ hồ làm nhiễu mô hình |
| Chia train/test theo artist | Cùng artist → rò rỉ nếu chia ngẫu nhiên |
| 30 giây ở giữa bài | Intro/outro không đại diện cho bài |
| Mono 22.050 Hz | Chuẩn audio ML, đủ dải tần 11 kHz |
| Đặc trưng: spectral + rhythm (25 chiều) | Ablation cho thấy MFCC gây nhiễu |
| Macro-F1 / macro-recall | Tránh bị nhóm đông (vui: 378 bài) kéo lên |

---

## Hạn chế

- **Dữ liệu nhỏ:** 1.020 bài, nhóm căng-tối chỉ 127 bài
- **Nhãn nhiễu:** mood tag do người dùng gán, không đo đạc khách quan
- **Mood ≈ thể loại:** bình yên ≈ ambient, vui ≈ pop/electronic

---

## Tham khảo

- [MTG-Jamendo Dataset](https://github.com/MTG/mtg-jamendo-dataset)
- [librosa](https://librosa.org)
