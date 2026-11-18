# Tóm tắt đồ án: Phân tích đặc trưng âm thanh và đề xuất bài hát tương đồng

Cập nhật: 2026-11-14 (lần 5) · Môn: Tính toán đa phương tiện

> **Cách dùng file này**
> - Mở phiên làm việc mới: đọc **mục 0** (đang ở đâu, làm gì tiếp) và **mục 10** (nhật ký) là đủ, không cần đọc lại từ đầu.
> - Làm xong một việc: thêm một dòng vào **mục 10** (mới nhất ở trên), rồi cập nhật mục 0 nếu trạng thái thay đổi.
> - Mục 1–9 là kiến thức nền và kết quả đã chốt; chỉ sửa khi có quyết định mới.

## 0. Trạng thái hiện tại (2026-11-14)

**Đang ở bước:** đã chạy xong nhánh A trên GTZAN (Colab). Dữ liệu Jamendo đã sẵn trên máy, notebook khám phá chạy được. **Việc tiếp theo:** trích đặc trưng và phân loại mood trên tập con Jamendo (mục 7, bước 2).

**Cấu trúc thư mục `project/`:**
```
project/
├── 01_explore-gtzan.ipynb     # chỉ GTZAN (mục 5): tải GTZAN lên Drive trên Colab, âm thanh số B1–B4
├── 01b_explore_jamendo_local.ipynb  # notebook khám phá bản chạy LOCAL trên Jamendo thật (mục 5c), trước đây tên Antigravity.ipynb
├── 02_genre_similarity.ipynb  # đặc trưng + phân loại + tìm bài giống trên GTZAN (mục 5b), chạy trên Colab
├── TOM_TAT.md                 # file này
├── Tính toán đa phương tiện.md  # đề cương / ý tưởng (đã cập nhật theo quyết định)
└── data_jamendo/
    ├── audio-low/             # tar 00–09 đã giải nén: 1.991 file .low.mp3, ~4,7 GB, đủ 100% so với checksum
    ├── metadata/
    │   ├── autotagging_moodtheme.tsv, autotagging_genre.tsv   # giống hệt bản trên GitHub
    │   ├── splits/split-0/autotagging_moodtheme-{train,validation,test}.tsv
    │   └── download/autotagging_moodtheme_audio-low_sha256_{tars,tracks}.txt
    └── index_mood.csv         # do 01b tạo: 1.020 bài sạch mood có audio, có cột split (train 551 / validation 212 / test 245 / trống 12)
```
- Thư mục cũ `data/`, `data_gtzan/`, `scripts/` **không còn**. GTZAN chỉ nằm trên Google Drive (`MyDrive/project/data/gtzan/...`).
- Môi trường local: Anaconda `base` (`/opt/anaconda3`, Python 3.13), đã cài `librosa 1.0.1`. Kernel notebook là `base`.
- Ổ đĩa máy còn khoảng **16 GB** trống (đầy 97%).

**Số liệu tập con Jamendo hiện có (10 tar):** 1.020 bài sạch mood, 299 nghệ sĩ.

| Nhóm mood | Số bài có audio |
|---|---|
| Vui, năng lượng cao | 378 |
| Bình yên, năng lượng thấp | 331 |
| Buồn, năng lượng thấp | 184 |
| Căng/tối, năng lượng cao | 127 |

**Đang bỏ ngỏ:**
- Có tải thêm tar 10–19 hay không. Tải từ máy rất chậm (đo 2026-11-14: 15–31 KB/s, mỗi tar 4–8 tiếng). Nếu cần thì tải bằng Colab rồi chép về.
- Nhóm căng/tối chỉ có 127 bài → test rất ít bài; ghi rõ trong báo cáo nếu không tải thêm.

## 1. Đề tài
Hệ thống nhận một bài hát, trích đặc trưng âm thanh và trả về các bài "nghe giống" trong thư viện nhỏ, kèm lý do vì sao giống. Giao diện phát nhạc (web mini) chỉ để demo; trọng tâm là xử lý tín hiệu âm thanh.

Hai nhánh:
- **A. Nghe giống:** đặc trưng → một vector mỗi bài → phân loại thể loại/mood → cosine top-k.
- **B. Nhận diện bài kiểu Shazam:** phổ → đỉnh cục bộ → hash cặp đỉnh → bỏ phiếu theo độ lệch thời gian.
- **Demo gộp:** đoạn thu → B nhận ra bài → A gợi ý bài giống.

Thứ tự làm: A chạy trọn vẹn trước, B làm bản tối thiểu, hướng C (nhạc Việt) để phát triển sau.

## 2. Phân loại hay tìm kiếm?
- **Phân loại** (thể loại, mood): công cụ kiểm tra chất lượng vector đặc trưng. Đo bằng macro-F1, confusion matrix.
- **Tìm bài giống** (retrieval): sản phẩm chính. Không cần huấn luyện, chỉ cần vector + cosine. Đo bằng precision@k theo nhãn, cộng khảo sát người nghe nhỏ.
- Nhãn chỉ dùng để chấm điểm, không dùng để lọc cứng.

## 3. Các quyết định đã chốt
- Bỏ lọc cứng theo thể loại/mood; dùng một không gian vector chung.
- Trọng số 40/25/20/15 trong ý tưởng ban đầu không có căn cứ → chỉnh trên tập validation hoặc bỏ.
- Mood gom thành 4 nhóm theo valence × arousal: vui–năng lượng cao, căng/tối–năng lượng cao, buồn–năng lượng thấp, bình yên–năng lượng thấp. Chỉ giữ bài có tag rơi vào đúng một nhóm.
- Đánh giá không vòng tròn: bản chỉ dùng âm thanh chấm bằng nhãn; bản có dùng nhãn chấm bằng khảo sát người nghe.
- Dataset: **MTG-Jamendo `autotagging_moodtheme/audio-low` (tập con) là chính** (thể loại + mood + phát nhạc); **GTZAN** để chạy thử pipeline và test ngoài miền. Không trộn hai tập.
- Không tải các bộ khác của Jamendo (`raw_30s`, bản full quality, `melspecs`): quá lớn, không cần cho đề tài (2026-11-14).
- Không dùng thể loại "lofi, ballad, EDM" trừ khi dataset có nhãn đó; lấy danh sách thể loại theo nhãn thật.
- Tần số lấy mẫu chung: **mono, 22.050 Hz**; mỗi bài lấy đoạn **30 giây ở giữa bài**.

## 4. Kết quả khám phá metadata Jamendo (chạy trên dữ liệu thật)
Bộ mood/theme: 18.486 bài, 1.533 nghệ sĩ, bài dài trung bình khoảng 219 giây.

Với từ điển 4 nhóm trong notebook, còn **9.049 bài** (49%) có đúng một nhóm mood:

| Nhóm mood | Số bài | Số nghệ sĩ |
|---|---|---|
| Vui, năng lượng cao | 3.365 | 495 |
| Bình yên, năng lượng thấp | 2.779 | 470 |
| Buồn, năng lượng thấp | 1.542 | 339 |
| Căng/tối, năng lượng cao | 1.363 | 316 |

- Khoảng 1.136 bài lẫn 2–3 nhóm; khoảng 8.300 bài không có tag nào trong từ điển.
- Nhóm lớn nhất gấp khoảng 2,5 lần nhóm nhỏ nhất → dùng macro-F1 hoặc lấy mẫu cân bằng.
- `split-0` có sẵn không trùng nghệ sĩ hay album giữa train/validation/test.
- **Mood bị lẫn với thể loại:** bình yên ≈ ambient/newage/easylistening, vui ≈ pop/electronic.
- **Baseline chỉ dùng thể loại** (mục A4c của `01b_explore_jamendo_local.ipynb`; logistic regression trên tag thể loại, train/test của split-0: 4.836 / 2.140 bài): accuracy 0,505, **macro-F1 0,501**. Đoán lớp đa số: accuracy 0,306; đoán bừa: macro-F1 ≈ 0,25. → **Mô hình mood dùng âm thanh phải vượt rõ 0,50** mới chứng minh học được mood chứ không chỉ thể loại. Đây là mốc đưa vào báo cáo.
- Cách đo đơn giản hơn trong `01b_explore_jamendo_local.ipynb` (mỗi thể loại lớn → đoán mood hay gặp nhất, học trên train 2.749 bài, chấm trên test 1.200 bài của split-0): accuracy 0,415, macro-recall 0,354; nhóm buồn và căng/tối recall 0. (Bản cũ chấm trên chính dữ liệu đã học cho 0,547 / 0,444, bị lạc quan.) Khác số với dòng trên vì khác cách đo; **mốc chính thức vẫn là 0,50**.
- **Từ điển mood mở rộng** (thêm epic, action, horror, positive, hopeful, inspiring, motivational, groovy, funny, deep, soundscape, ambiental): bài sạch tăng 9.049 → 11.284, nhóm căng/tối 1.363 → 2.066, nhưng nhóm buồn giảm 1.542 → 1.350 và bài lẫn nhóm tăng 1.136 → 1.865. Hiện vẫn dùng **từ điển gốc** (`USE_EXTENDED = False`); nếu đổi phải ghi lý do trong báo cáo.
- Split-0 theo nhóm mood (từ điển gốc): train 4.912, validation 1.938, test 2.170; 29 bài không có trong split.

## 5. Notebook `01_explore-gtzan.ipynb` (chỉ GTZAN, Colab)
Chạy trên Colab, GTZAN tải thẳng vào Google Drive (`USE_DRIVE = True`), không cần tải về máy. **Từ 2026-11-14 chỉ còn phần GTZAN**; mọi phần Jamendo (tải metadata/audio, phần A, A3b, A4b, A4c) đã bỏ, phần cần thiết chuyển sang `01b_explore_jamendo_local.ipynb`.
- 0.1: cài thư viện, mount Drive. 0.2: Kaggle key đọc từ **Colab Secrets** (`KAGGLE_API_TOKEN`, hoặc `KAGGLE_USERNAME` + `KAGGLE_KEY`) rồi tải GTZAN. Không dán key vào code.
- Cấu hình tự tìm `genres_original` của GTZAN, bỏ `jazz.00054.wav`, tự chọn bài GTZAN thật làm `SAMPLE_FILE`; không có GTZAN thì dùng tín hiệu giả (DEMO).
- Nội dung: lấy mẫu → waveform → STFT → mel-spectrogram (B1–B4), kèm phần tự thử và câu hỏi vấn đáp.
- Mel dùng `HTK = True` cho khớp công thức; notebook sau **phải giữ cùng lựa chọn**.
- **Đã chạy trọn trên Colab (2026-11-11)** (bản cũ còn phần Jamendo): GTZAN trên Drive tại `MyDrive/project/data/gtzan/Data/genres_original`; phần âm thanh chạy trên `blues.00000.wav` (30 s, 661.794 mẫu). Bản đã bỏ Jamendo chạy local không lỗi (dùng DEMO vì máy không có GTZAN).
- Chưa kiểm tra: từ điển mood mở rộng (A3b) và kiểm tra trùng nghệ sĩ/album (A4b) không còn trong notebook nào; kết quả cũ vẫn ghi ở mục 4.
- Secret Colab phải đặt tên đúng `KAGGLE_API_TOKEN` (tên token bên Kaggle thì tùy ý).

## 5b. Notebook `02_genre_similarity.ipynb` (đã chạy trên GTZAN, 2026-11-11)
- Đặc trưng 93 chiều, 4 nhóm: `mfcc` 40 (20 hệ số × mean/std), `chroma` 24, `spec` 26 (centroid, bandwidth, rolloff, flatness, contrast 7 băng, zcr, rms), `tempo` 3 (BPM, onset mean/std). Cùng tham số với notebook 01 (`HTK = True`). Cache CSV trên Drive: `data/features/`.
- Phân loại: 5-fold stratified CV, `StandardScaler` trong `Pipeline`; so KNN, Logistic Regression, SVM RBF, Random Forest với 2 baseline (lớp đa số, ngẫu nhiên). Đo macro-F1 + confusion matrix.
- Tìm bài giống: chuẩn hoá → cân bằng nhóm (nhân 1/√số chiều) → cosine → precision@1/5/10 leave-one-out; baseline ngẫu nhiên ≈ 0,10.
- Ablation 9 tổ hợp nhóm cho cả SVM macro-F1 và P@10. Kết quả lưu `results/gtzan_*.csv` (trên Drive).
- Kiểm tra bài trùng: liệt kê cặp có cosine gần 1.
- **Quyết định thiết kế:** giải thích "giống nhờ / khác về" dùng **thứ hạng phần trăm theo từng nhóm**, không dùng thẳng cosine của nhóm, vì nhóm ít chiều (tempo 3 chiều) cho cosine cao một cách dễ dàng.
- **Đã chạy trên GTZAN thật (2026-11-11):** 999 bài, trích đặc trưng 8,4 phút trên Colab, 0 lỗi.

**Kết quả GTZAN** (chia ngẫu nhiên 5-fold, chỉ để kiểm tra pipeline):

| | macro-F1 | P@10 |
|---|---|---|
| Ngẫu nhiên | 0,094 | ≈ 0,099 |
| KNN / LogReg / RF | 0,704 / 0,752 / 0,765 | |
| **SVM (RBF), đủ 4 nhóm** | **0,792 ± 0,019** | 0,564 (không cân bằng) / 0,554 (cân bằng) |
| mfcc + spec | 0,788 | **0,584** (tốt nhất) |
| chỉ spec / chỉ mfcc | 0,726 / 0,716 | 0,534 / 0,504 |
| chỉ chroma / chỉ tempo | 0,468 / 0,345 | 0,319 / 0,232 |

- **Phát hiện chính:** âm sắc (MFCC) và hình dạng phổ (spec) quyết định thể loại; chroma **làm giảm** P@10 khi thêm vào (chroma đo giọng/key, khác nhau giữa các bài cùng thể loại); tempo yếu (ước lượng BPM hay sai bội số).
- **Cân bằng nhóm không giúp** → tạm thời không dùng; ứng viên tốt cho tìm bài giống là **mfcc + spec**.
- Thể loại khó nhất: rock (F1 0,58, P@10 ≈ 0,23), disco, reggae; dễ nhất: classical, jazz, metal. Nhầm nhiều: disco → rock 11%, country → rock 8%.
- **Bài trùng:** 31 cặp cosine > 0,98, nhiều cặp = 1,000 (vd metal.00040–45 ↔ metal.00061–66); **metal.00058 = rock.00016** là lỗi nhãn. Điểm trên bị thổi phồng một phần.
- Demo blues.00000: top-5 là jazz/rock/disco, không có blues; giống về âm sắc, khác về nhịp → "nghe giống" ≠ "cùng thể loại" (ví dụ hay cho báo cáo).
- Việc nên làm thêm: chạy lại sau khi bỏ bài trùng (cosine > 0,999) để đo mức thổi phồng.

## 5c. Notebook `01b_explore_jamendo_local.ipynb` (bản local, chạy trên Jamendo thật, 2026-11-14)
- Notebook khám phá riêng cho **Jamendo**, chạy local: metadata ở `data_jamendo/metadata/`, audio ở `data_jamendo/audio-low/`. Không liên quan đến `01_explore-gtzan.ipynb` (notebook đó của bạn làm cho dataset GTZAN).
- Phần A: đọc TSV, ghép genre theo `TRACK_ID`, gán split-0, gom 4 nhóm mood, gộp 8 thể loại lớn, kiểm tra rò rỉ mood ↔ thể loại (học trên train, chấm trên test), lưu `data_jamendo/index_mood.csv` (cột: `track_id, artist_id, album_id, path, duration, quadrant, genre_coarse, split`).
- A4c: baseline chính thức LogReg chỉ-thể-loại, macro-F1 **0,501** (chuyển từ `01_explore-gtzan.ipynb` sang, chạy lại ra đúng số cũ).
- Phần B: lấy mẫu → waveform → STFT (so `n_fft` 512 / 4096) → mel-spectrogram 128 băng (`HTK = True`, giống notebook 01 và 02) → mel của 4 bài thật, mỗi nhóm một bài.
- `SAMPLE_FILE = None` thì tự lấy bài thật đầu tiên trong `audio-low` (`00/1009600.low.mp3`), đọc 30 s.
- **Đã chạy trọn, không lỗi (2026-11-14)** bằng `jupyter nbconvert --execute`.

## 6. Dữ liệu
- **Jamendo `autotagging_moodtheme/audio-low`** (46 GB, 100 tar, 18.486 bài): đã có tar **00–09** (1.991 bài, khớp checksum). Tar 10–99 chưa tải (~41 GB). 20 tar sẽ cho ≈1.999 bài sạch; với 10 tar, nhóm căng/tối chỉ có 127 bài.
- Link tải tar: `https://cdn.freesound.org/mtg-jamendo/autotagging_moodtheme/audio-low/autotagging_moodtheme_audio-low-NN.tar` (mirror khác: `https://essentia.upf.edu/documentation/datasets/mtg-jamendo/...`). Cả hai đều tải được nhưng **rất chậm từ máy** (15–31 KB/s, đo 2026-11-14).
- Script chính thức `scripts/download/download.py` trong repo `MTG/mtg-jamendo-dataset` tải **cả 100 tar**, không chọn khoảng được, và tải lại tar nào đã xoá → không dùng; nếu tải thì tự lặp theo khoảng tar + kiểm checksum.
- Metadata lấy từ thư mục `data/` của repo `https://github.com/MTG/mtg-jamendo-dataset`: chỉ cần `autotagging_moodtheme.tsv`, `autotagging_genre.tsv`, `splits/split-0/autotagging_moodtheme-*.tsv`, `download/*audio-low*.txt`.
- **GTZAN** (≈1,2 GB, Kaggle): chỉ dùng `genres_original`; bỏ file `jazz.00054.wav` (hỏng). GTZAN có lỗi nhãn và bài trùng → chỉ để chạy thử và tham khảo. Hiện chỉ có trên Google Drive.
- Tên file trong tar có dạng `00/1009600.low.mp3`, khác cột `PATH` trong metadata (`48/948.mp3`): khi ghép đổi đuôi `.mp3` thành `.low.mp3`.

## 7. Thứ tự làm tiếp
1. ~~Chạy pipeline trên GTZAN: trích đặc trưng, phân loại thể loại, precision@k~~ → **xong 2026-11-11** (mục 5b).
2. **[Tiếp theo]** Chạy lại trên tập con Jamendo (`data_jamendo/index_mood.csv`, 1.020 bài): trích cùng 93 đặc trưng, phân loại mood theo split-0 (chia theo nghệ sĩ), so với mốc chỉ-thể-loại 0,50; precision@k theo mood.
3. Test chéo GTZAN ↔ Jamendo trên các thể loại chung (đối chiếu nhãn hai bên trước); điểm thấp hơn là bình thường vì khác miền dữ liệu (cắt Jamendo thành đoạn 30 giây cho cùng cỡ).
4. Ablation trên Jamendo (MFCC → thêm chroma, tempo, spectral) và giải thích kết quả.
5. Nhánh B tối thiểu, rồi khảo sát người nghe, demo gộp, giao diện web mini, báo cáo.

## 8. Rủi ro chính
- **Bảo mật:** Kaggle token cũ đã lộ trong notebook → thu hồi trên Kaggle, chỉ dùng Colab Secrets.
- Phạm vi A + B quá rộng → làm A trọn vẹn trước, đặt mốc cắt B.
- Nhãn mood nhiễu, gián tiếp (lấy từ tag) → báo cáo mood như kết quả thăm dò.
- Rò dữ liệu theo nghệ sĩ/album → dùng split có sẵn.
- Precision@k theo nhãn chỉ là đại diện cho "nghe giống" → bù bằng khảo sát người nghe.
- Tập con nhỏ (1.020 bài, nhóm nhỏ nhất 127) → kết quả mood dao động; báo cáo kèm số bài từng nhóm.
- Ổ đĩa gần đầy (16 GB trống) và mạng tải Jamendo chậm → khó mở rộng dữ liệu trên máy.
- Chưa kiểm chứng: độ trùng nhãn thể loại GTZAN–Jamendo.

## 9. Tài liệu liên quan
- Đề cương đầy đủ (doc trên Claude): https://claude.ai/artifact/JDhMREzVzGiGTNvXFq6Am6
- Ý tưởng và quyết định: `Tính toán đa phương tiện.md`
- Repo dataset: https://github.com/MTG/mtg-jamendo-dataset

## 10. Nhật ký (mới nhất ở trên)
Mỗi dòng: ngày · việc đã làm · kết quả / file thay đổi.

- **2026-11-14** · Bỏ toàn bộ phần Jamendo khỏi `01_explore-gtzan.ipynb` (ô tải 0.3/0.4, phần A, A3b, A4b, A4c, câu hỏi về mood/split); đổi tiêu đề thành notebook GTZAN, `SAMPLE_FILE` chỉ chọn bài GTZAN. 31 → 14 ô. Chạy local không lỗi (DEMO).

- **2026-11-14** · Chuyển baseline LogReg chỉ-thể-loại (A4c) sang `01b_explore_jamendo_local.ipynb`. Chạy lại khớp số cũ: train 4.836 / test 2.140, accuracy 0,505, macro-F1 0,501. `01_explore-gtzan.ipynb` chưa sửa (người dùng tự bỏ phần Jamendo nếu muốn).

- **2026-11-14** · Sửa phần giới thiệu `01b_explore_jamendo_local.ipynb`: bỏ câu "bản local của `01_explore-gtzan.ipynb`" vì người dùng xác nhận `01_explore-gtzan` là notebook cho dataset GTZAN, không phải bản Colab của 01b.

- **2026-11-14** · Sửa 4 điểm của notebook khám phá local: (1) thêm cột `split` (split-0) vào `index_mood.csv`; (2) mel dùng `HTK = True`; (3) kiểm tra rò rỉ học trên train, chấm trên test (macro-recall 0,354 thay vì 0,444); (4) đổi tên `Antigravity.ipynb` → `01b_explore_jamendo_local.ipynb`, sửa tiêu đề. Chạy trọn, không lỗi.

- **2026-11-14** · Đánh giá `01b_explore_jamendo_local.ipynb`: là bản local của notebook khám phá, thêm A6 (tạo `index_mood.csv`) và B5 (mel của bài Jamendo thật); thiếu A3b/A4b/A4c và `HTK = True`. 1.020 bài theo split-0: train 551, validation 212, test 245, 12 bài không có trong split (căng/tối test chỉ 25 bài). Chưa sửa notebook.

- **2026-11-14** · Viết lại `TOM_TAT.md` thành file theo dõi tiến độ (thêm mục 0, 5c, 10) và cập nhật `Tính toán đa phương tiện.md` theo các quyết định đã chốt.
- **2026-11-14** · Sắp xếp `data_jamendo/`: dời 2 file TSV vào `metadata/`, thêm `splits/split-0` (mood) và checksum audio-low từ repo MTG. Sửa đường dẫn trong `01b_explore_jamendo_local.ipynb` sang `data_jamendo/...`; chạy lại trọn, kết quả không đổi (1.991 audio, 1.020 bài sạch mood).
- **2026-11-14** · Thử tải tar 10–19: cả hai mirror chỉ 15–31 KB/s → dừng, đã dọn file tạm. Viết notebook tải bằng Colab nhưng người dùng chọn không dùng (đã xoá). Chưa tải thêm audio.
- **2026-11-14** · Kiểm tra audio có sẵn: đúng bộ `autotagging_moodtheme/audio-low`, tar 00–09 đủ 1.991/1.991 file so với checksum.
- **2026-11-14** · Sửa `01b_explore_jamendo_local.ipynb` chạy local: cài `librosa` vào Anaconda `base`; sửa đường dẫn metadata và audio; Phần B tự dùng bài thật thay vì tín hiệu DEMO.
- **2026-11-11** · Chạy `02_genre_similarity.ipynb` trên GTZAN (999 bài): SVM macro-F1 0,792; P@10 tốt nhất 0,584 với mfcc + spec; phát hiện 31 cặp bài trùng (mục 5b).
- **2026-11-11** · Chạy trọn notebook khám phá trên Colab: tải GTZAN + metadata Jamendo lên Drive; baseline chỉ-thể-loại macro-F1 0,501 (mục 4, 5).
- **2026-11-09** · Chốt hướng A + B, dùng MTG-Jamendo làm dataset chính, GTZAN để chạy thử (`Tính toán đa phương tiện.md`).
