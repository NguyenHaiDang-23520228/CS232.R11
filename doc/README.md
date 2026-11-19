# Tài liệu giải thích đồ án CS232 (Tính toán đa phương tiện)

Folder này giải thích từng notebook trong `notebooks/` bằng lời dễ hiểu: notebook **làm gì**, **vì sao làm vậy**, **kết quả thật là gì**, và **hạn chế nào cần nói thật**. Mục đích là để người đọc (Đăng) hiểu được đồ án đủ sâu để tự chạy lại và tự trình bày, không phải để học thuộc.

## Bức tranh tổng thể

Dữ liệu: bộ nhạc **MTG-Jamendo** (bản `audio-low`), dùng 1.020 bài có nhãn mood sạch và có file audio trên máy (299 artist).

Đồ án có **hai bài toán** cùng xuất phát từ spectrogram, nhưng bản chất khác nhau:

| | Bài toán A: đoán mood | Bài toán B: nhận diện bài (kiểu Shazam) |
|---|---|---|
| Câu hỏi | chỉ nghe âm thanh, đoán bài thuộc nhóm mood nào (vui, căng/tối, buồn, bình yên) | cho đoạn thu ngắn, tìm ra bài nào trong kho |
| Kiểu bài toán | phân loại (học máy) | tra cứu (không học máy) |
| Notebook | 01, 02 | 03 |

Điểm chung: cả hai đều cần hiểu âm thanh dưới dạng spectrogram (notebook 01).

## Thứ tự đọc

| File | Notebook | Nội dung chính |
|---|---|---|
| `01_notebook01_du_lieu_va_am_thanh.md` | `01_explore_data.ipynb` | Dữ liệu Jamendo, vì sao gom mood thành 4 nhóm, mood gắn chặt với thể loại; âm thanh thành số: lấy mẫu, STFT, mel |
| `02_notebook02_dac_trung_va_baseline.md` | `02_features_baseline.ipynb` | 93 đặc trưng âm thanh, chia train/test theo artist (tránh rò rỉ), các mốc so sánh, kết quả, ablation |
| `03_notebook03_fingerprint_shazam.md` | `03_fingerprint_shazam.ipynb` | Dấu vân tay âm thanh: đỉnh phổ, hash cặp đỉnh, bỏ phiếu theo độ lệch thời gian; đo độ chính xác theo nhiễu, độ dài, tham số, kích thước kho |

Mỗi file có phần **"Tóm tắt 30 giây"** ở cuối để ôn nhanh.

## Kết quả chính trong một bảng

| Notebook | Kết quả đáng nhớ |
|---|---|
| 02 | Mô hình âm thanh hơn đoán mò rõ (macro-recall 0,362 so với 0,257, thắng 20/20 fold chia theo artist), hơn mốc "chỉ dùng thể loại" một chút (0,362 so với 0,319), và phần lớn lợi thế đến từ những bài không rõ thể loại. Chia ngẫu nhiên theo bài làm điểm tăng ảo (0,356 lên 0,417). Mô hình đoán trục năng lượng (0,647) tốt hơn trục cảm xúc (0,569). |
| 03 | Đoạn 5 s sạch, nén MP3 hoặc qua micro điện thoại nhận đúng 100%; ồn 10 dB: 91%; ồn 0 dB: 50%. Đổi tốc độ 1% là sụp. Một truy vấn mất khoảng 4 ms; chỉ mục 1.020 bài nặng 128 MB. |

## Cách chạy lại

1. Cài thư viện: tạo môi trường ảo rồi `pip install -r requirements.txt` (cần thêm `joblib`; `soundfile` cần hỗ trợ MP3, tức libsndfile từ 1.1 trở lên, để notebook 03 mô phỏng nén MP3).
2. Đặt dữ liệu: `data/autotagging_moodtheme.tsv`, `data/autotagging_genre.tsv`, và thư mục audio (`audio-low`). Nếu audio nằm ngoài repo, đặt biến môi trường `JAMENDO_AUDIO_DIR`.
3. Chạy **theo thứ tự 01, 02, 03**. Notebook 01 tạo `data/index_mood.csv`, notebook 02 tạo bảng đặc trưng, notebook 03 dùng lại danh sách bài.
4. Thời gian tham khảo: 01 khoảng vài phút; 02 khoảng vài phút trích đặc trưng (có lưu cache); 03 khoảng 10 đến 20 phút lần đầu (có cache từng bước).

Ghi chú về số liệu: kết quả ở notebook 02 phụ thuộc phiên bản scikit-learn (cách chia fold theo artist có thể khác), notebook 03 phụ thuộc phiên bản thư viện âm thanh. Chạy trên máy khác có thể lệch vài điểm phần trăm; **kết luận không đổi**. Nhớ kết luận và mức chắc chắn của nó, đừng học thuộc từng con số.

## Cách dùng tài liệu này khi chuẩn bị vấn đáp

- Với mỗi notebook, tự trả lời được ba câu: *bài toán là gì*, *vì sao chọn cách làm này*, *kết quả nói được gì và chưa nói được gì*.
- Phần **"Hạn chế cần nói thật"** ở từng file quan trọng ngang phần kết quả: nói thẳng điểm yếu của mình thường được đánh giá tốt hơn là nói quá.
- Nên **tự chạy lại ít nhất một lần** và thử các mục "Tự thử" trong notebook. Hiểu bằng cách thay đổi tham số và nhìn kết quả đổi theo là cách nhớ lâu nhất.

## Nguồn

- Dataset: MTG-Jamendo (Bogdanov et al., 2019), dùng cho nghiên cứu phi thương mại.
- Ý tưởng dấu vân tay âm thanh: Avery Wang (2003), "An Industrial-Strength Audio Search Algorithm", ISMIR.
