# Notebook 02 · Đặc trưng âm thanh và baseline phân loại mood

> File này giải thích notebook `notebooks/02_features_baseline.ipynb`. Nên đọc `01_notebook01_du_lieu_va_am_thanh.md` trước. Cuối file có phần "Tóm tắt 30 giây" để ôn nhanh.

## Mục tiêu và bức tranh tổng thể

Notebook 02 trả lời một câu hỏi: chỉ nghe âm thanh, không nhìn tag thể loại, mô hình đoán đúng nhóm mood tới đâu, và có hơn mốc "chỉ dùng thể loại" hay không.

Luồng xử lý, mỗi bước là một phần của notebook:
1. Đọc data_jamendo/index_mood.csv từ notebook 01: các bài sạch mood và có audio.
2. Với mỗi bài, cắt 30 giây ở giữa, tính mel-spectrogram rồi trích đặc trưng. Mỗi bài thành một vector có số chiều cố định.
3. Lưu bảng đặc trưng ra file để không phải tính lại.
4. Chia train/test theo artist, không theo bài.
5. Chuẩn hóa (fit chỉ trên train) và huấn luyện Logistic Regression.
6. Đo macro-F1 và confusion matrix, so với 3 mốc, rồi làm ablation.

Vì sao phải gom về một vector cố định chiều: bài dài ngắn khác nhau nên spectrogram mỗi bài có số khung khác nhau, trong khi bộ phân loại cần đầu vào cùng kích thước. Ta tóm tắt mỗi đặc trưng theo thời gian bằng trung bình và độ lệch chuẩn.

Ba mốc để so sánh. Một mô hình chỉ có nghĩa khi vượt được chúng:
- Đoán mò: macro-recall 0.25 với 4 nhóm.
- Luôn đoán nhóm đông nhất: đúng khoảng bằng tỉ lệ nhóm đó, nhưng macro-recall vẫn chỉ 0.25.
- Chỉ dùng thể loại để đoán mood: notebook 01 ước khoảng 0.44, nhưng số đó tính trên chính dữ liệu đó, không chia train/test, nên **không dùng được** để so. Notebook 02 tính lại đúng cách (học trên train, đo trên test): macro-recall 0.319, macro-F1 0.263.

Các con số trong tài liệu này lấy từ lần chạy notebook mới nhất (Python 3.14, scikit-learn 1.9.1), không ước lượng trước. Nếu bạn chạy với phiên bản thư viện khác, cách chia fold theo artist có thể khác nên điểm lệch vài phần trăm (ví dụ macro-F1 của LogReg ra 0.346 thay vì 0.356). Kết luận không đổi, vì vậy hãy nhớ kết luận và mức chắc chắn của nó, đừng học thuộc từng con số.

## Trích đặc trưng: từ mel đến vector 93 chiều

Mỗi bài trong 1,020 bài sạch mood (299 artist) được đổi thành một vector 93 số. Chạy trên 4 nhân CPU mất 38 giây, không bài nào lỗi, không có ô thiếu giá trị.

### Việc đầu tiên: cắt 30 giây giữa bài

Bài dài trung vị khoảng 203 giây, dài nhất gần 44 phút. Ta chỉ lấy 30 giây ở giữa vì đầu và cuối bài thường là đoạn dạo và outro, không đại diện cho cả bài. Đây là một giả định: nếu mood thay đổi giữa bài thì 30 giây đó chỉ phản ánh một phần.

### MFCC: log-mel rồi DCT

Mel-spectrogram cho 128 số mỗi khung thời gian, nhưng các băng kề nhau tương quan mạnh vì bộ lọc mel chồng lấn. MFCC xử lý hai bước: lấy log năng lượng mel, rồi áp DCT dọc theo trục tần số.

$$c_k = \sum_{m=0}^{M-1} \log E_m \cdot \cos\Big[\tfrac{\pi k}{M}\big(m+\tfrac12\big)\Big], \qquad k = 0, 1, \dots, 19$$

Trong đó $E_m$ là năng lượng băng mel thứ $m$ và $M = 128$. DCT làm hai việc: khử tương quan giữa các băng, và dồn thông tin vào vài hệ số đầu. Hệ số nhỏ (k = 0, 1, 2) mô tả dáng chung của phổ, gồm mức năng lượng và độ nghiêng giữa tần thấp và cao. Hệ số lớn dần mô tả chi tiết mịn hơn, như hài âm. Ta giữ 20 hệ số đầu, tức đường bao phổ, xấp xỉ âm sắc.

Ta tự kiểm bằng cách tự tính DCT từ log-mel rồi so với librosa.feature.mfcc: hai kết quả trùng nhau, độ lệch tối đa bằng 0. Vậy MFCC không phải hộp đen, nó chỉ là DCT của log-mel.

### Năm nhóm đặc trưng

Mood thể hiện qua năng lượng, âm sắc, hòa âm và nhịp. Mỗi nhóm dưới đây đo một mặt.

| Nhóm     | Số chiều | Đặc trưng                                        | Đo cái gì                                                     |
|----------|----------|--------------------------------------------------|---------------------------------------------------------------|
| mfcc     | 40       | 20 hệ số MFCC                                    | Âm sắc (đường bao phổ)                                        |
| chroma   | 24       | 12 lớp cao độ (C, C#, ... B)                     | Hòa âm: nốt nào đang vang, trưởng hay thứ                     |
| spectral | 22       | centroid, rolloff, bandwidth, flatness, contrast | Độ sáng, độ rộng phổ, mức giống tiếng ồn, tương phản đỉnh/đáy |
| energy   | 4        | RMS, zero-crossing rate                          | Độ to, độ gắt                                                 |
| rhythm   | 3        | tempo, trung bình và độ lệch chuẩn của onset     | Tốc độ, độ dồn dập của nhịp                                   |

Mỗi đặc trưng theo thời gian (một giá trị mỗi khung) được tóm tắt bằng trung bình và độ lệch chuẩn theo thời gian. Trung bình cho biết mức chung, độ lệch chuẩn cho biết bài biến động nhiều hay đều đều. Nhờ đó bài dài hay ngắn đều cho ra đúng 93 số.

### Thang số lệch nhau hàng chục ngàn lần

Trong bảng đặc trưng, khoảng giá trị nhỏ nhất là 0.12 (độ lệch chuẩn của RMS), lớn nhất là khoảng 8,263 (rolloff trung bình), tức chênh nhau khoảng 69,000 lần. Nếu đưa thẳng vào Logistic Regression hoặc tính khoảng cách, các chiều có số lớn sẽ lấn át các chiều nhỏ. Vì vậy bước chuẩn hóa là bắt buộc, không phải tùy chọn.

## Chia theo artist: rò rỉ làm điểm ảo bao nhiêu

Chia ngẫu nhiên theo bài làm macro-F1 của cùng một mô hình tăng từ 0.356 lên 0.417, tức cao hơn khoảng 17%, chỉ vì bài của cùng một artist nằm ở cả train lẫn test.

### Vì sao có rò rỉ

Cùng một nghệ sĩ thường có cùng giọng hát, nhạc cụ, cách thu âm và phối khí, và hay gán tag mood giống nhau cho các bài của mình. Nếu một phần bài của họ ở train và phần còn lại ở test, mô hình có thể học "nghe giống artist X thì nhãn là Y" thay vì học mood. Điểm test khi đó đo khả năng nhận ra artist, không đo khả năng nhận ra mood.

### Cách làm

Ta dùng StratifiedGroupKFold chia 4 fold: mỗi artist nằm trọn trong một fold, và tỉ lệ 4 nhóm mood mỗi fold xấp xỉ nhau. Vì dữ liệu nhỏ (127 bài ở nhóm nhỏ nhất) nên điểm rất rung theo cách chia. Ta lặp 5 lần với 5 cách xáo khác nhau, tổng 20 fold, rồi lấy trung bình và độ lệch chuẩn.

### Số liệu

Cùng một mô hình (Logistic Regression), cùng 93 đặc trưng, chỉ khác cách chia:

| Cách chia           | Artist có ở cả train và test (mỗi fold) | macro-F1      | macro-recall  |
|---------------------|-----------------------------------------|---------------|---------------|
| Theo artist         | 0                                       | 0.356 ± 0.041 | 0.362 ± 0.042 |
| Ngẫu nhiên theo bài | 100 đến 108 (trong 299 artist)          | 0.417 ± 0.029 | 0.427 ± 0.031 |

Ta kiểm tra trực tiếp: với cách chia theo artist, số artist trùng giữa train và test bằng 0 ở mọi fold. Với cách chia ngẫu nhiên, hơn một phần ba số artist bị trùng.

Mức ảo ở đây vừa phải. Có thể một phần vì mỗi artist trong tập này trung bình chỉ có 3.4 bài; ta chưa kiểm chứng điều này. Điều chắc chắn là điểm theo artist mới là con số trung thực, và là con số nên báo cáo.

## Chuẩn hóa và baseline: điểm bao nhiêu thì gọi là tốt?

Một con số macro-F1 0.356 tự nó không nói gì. Phải có mốc để so, và phải xử lý thang số trước khi cho mô hình học.

### Chuẩn hóa nằm trong pipeline

Thấy ở phần trước: rolloff_mean_00 có khoảng giá trị rộng hơn 8000, còn rms_std_00 chỉ khoảng 0.12. Logistic Regression có phạt độ lớn hệ số, nên đặc trưng nào số to sẽ bị đối xử khác đặc trưng số nhỏ. Ta dùng StandardScaler (trừ trung bình, chia độ lệch chuẩn) nhưng đặt **bên trong pipeline**, như vậy mỗi fold chỉ tính trung bình và độ lệch chuẩn trên phần train. Nếu chuẩn hóa cả bảng trước rồi mới chia fold, thông tin của phần test đã lọt vào train, một dạng rò rỉ nhỏ nhưng có thật.

Random Forest không cần chuẩn hóa vì nó chỉ so sánh đặc trưng với ngưỡng.

### Ba baseline
- **Đoán mò theo tỉ lệ nhóm:** đoán ngẫu nhiên theo tỉ lệ nhóm trong train. Đây là sàn thấp nhất.
- **Luôn đoán nhóm đông nhất:** luôn trả 'vui'. Đạt accuracy 0.371 mà chẳng học điều gì, đây là lý do ta không nhìn accuracy một mình.
- **Chỉ dùng thể loại:** với mỗi thể loại lớn, đoán nhóm mood phổ biến nhất của thể loại đó trong train; bài không rõ thể loại thì đoán nhóm đông nhất. Đây là mốc quan trọng nhất, vì mood và thể loại gắn chặt. Nếu mô hình âm thanh không hơn mốc này thì nó chưa chứng minh được 'nghe nhạc' hữu ích.

### Dùng macro, không dùng accuracy

Nhóm nhỏ nhất (căng-tối, 127 bài) chỉ bằng khoảng một phần ba nhóm lớn nhất (vui, 378 bài). **Macro-recall** lấy recall của từng nhóm rồi trung bình, nên mỗi nhóm nặng bằng nhau, và 'luôn đoán vui' chỉ được đúng 0.25 (recall 1 cho nhóm vui, 0 cho ba nhóm còn lại). **Macro-F1** tương tự nhưng phạt thêm việc đoán nhầm vào một nhóm (precision thấp), nên 'luôn đoán vui' chỉ còn 0.135. Với 4 nhóm, đoán mò theo tỉ lệ cho cả hai chỉ số quanh 0.25.

### Kết quả (chia theo artist, 20 fold, trung bình ± độ lệch chuẩn)

| Mô hình                                   | macro-F1      | macro-recall  | accuracy      |
|-------------------------------------------|---------------|---------------|---------------|
| Đoán mò theo tỉ lệ nhóm                   | 0.256 ± 0.034 | 0.257 ± 0.033 | 0.310 ± 0.036 |
| Luôn đoán nhóm đông nhất                  | 0.135 ± 0.000 | 0.250 ± 0.000 | 0.371 ± 0.002 |
| Chỉ dùng thể loại                         | 0.263 ± 0.038 | 0.319 ± 0.034 | 0.445 ± 0.046 |
| Âm thanh, Logistic Regression             | 0.356 ± 0.041 | 0.362 ± 0.042 | 0.394 ± 0.041 |
| Âm thanh, Random Forest                   | 0.354 ± 0.038 | 0.382 ± 0.029 | 0.505 ± 0.033 |
| (Tham khảo, rò rỉ) LogReg chia ngẫu nhiên | 0.417 ± 0.029 | 0.427 ± 0.031 | 0.452 ± 0.030 |

Một điều đáng chú ý: mốc 'chỉ dùng thể loại' có macro-F1 0.263, gần như đoán mò, nhưng macro-recall 0.319 và accuracy 0.445 thì cao hơn rõ. Tức là thể loại có cho một chút thông tin, nhưng đoán nhầm nhiều nên F1 không tăng. Khi so với mô hình âm thanh, nên nói rõ ta so trên chỉ số nào.

## Đọc kết quả: mô hình học được gì, và đừng nói quá điều gì

Tóm tắt trung thực: mô hình âm thanh hơn đoán mò rõ ràng, nhưng chỉ hơn mốc 'chỉ dùng thể loại' một chút, và phần lớn lợi thế đó nằm ở những bài không rõ thể loại.

### 1. Hơn mốc thể loại đến đâu?

So sánh ghép cặp trên cùng 20 fold (hiệu macro-recall):

| So sánh                        | Hiệu trung bình | Độ lệch chuẩn của hiệu | Tỉ lệ fold thắng |
|--------------------------------|-----------------|------------------------|------------------|
| LogReg trừ đoán mò             | +0.105          | 0.036                  | 100%             |
| LogReg trừ chỉ thể loại        | +0.043          | 0.037                  | 90%              |
| Random Forest trừ chỉ thể loại | +0.063          | 0.026                  | 100%             |
| Random Forest trừ LogReg       | +0.020          | 0.029                  | 75%              |
- Hơn đoán mò thì chắc: thắng ở cả 20 fold.
- Hơn mốc thể loại thì khiêm tốn hơn: với LogReg, hiệu 0.043 chỉ nhỉnh hơn độ lệch chuẩn 0.037 và thắng 90% số fold. Random Forest hơn 0.063 và thắng 100% số fold, nhưng cũng chỉ là mức hơn vừa phải.
- Random Forest hơn LogReg chưa chắc: hiệu 0.020 nhỏ hơn độ lệch chuẩn 0.029 và chỉ thắng 75% số fold.
- 20 fold này dùng chung dữ liệu nên không độc lập hoàn toàn. Các con số cho thấy xu hướng, không phải kiểm định thống kê chặt.

### 2. Mô hình học mood hay học thể loại?

Chỉ có 641 trong 1020 bài thuộc đúng một thể loại lớn; 379 bài còn lại là 'unknown'. Ta tính điểm riêng cho hai tập (macro-recall):

| Tập                   | Chỉ thể loại | LogReg | Random Forest |
|-----------------------|--------------|--------|---------------|
| Thể loại rõ (641 bài) | 0.364        | 0.376  | 0.400         |
| Unknown (379 bài)     | 0.246        | 0.336  | 0.352         |

Ở bài thể loại rõ, LogReg chỉ hơn mốc thể loại 0.012, gần như hòa. Ở bài unknown, mốc thể loại chỉ còn cách đoán nhóm đông nhất nên rớt xuống 0.246, còn mô hình âm thanh vẫn nghe được nên giữ 0.336. Vậy phần lớn khoảng cách 0.043 đến từ chỗ mốc thể loại 'mù', không phải từ việc âm thanh chứa thêm nhiều thông tin về mood. (Theo macro-F1, khoảng cách ở bài thể loại rõ lớn hơn một chút: 0.368 so với 0.329. Bạn nên nói cả hai chỉ số thay vì chọn cái có lợi.)

Tập con nhỏ nên các con số này còn nhiễu. Đây là bằng chứng gợi ý, chưa kết luận.

### 3. Đoán đúng nhóm nào, nhầm sang nhóm nào?

Recall từng nhóm của LogReg: vui 0.523, bình yên 0.341, buồn 0.301, căng-tối 0.283. Nhóm 'vui' dễ nhất và cũng đông nhất trong dữ liệu. Hai cặp nhầm lẫn nổi bật trong ma trận nhầm lẫn:
- Thật là 'buồn' thì 34% bị đoán thành 'bình yên'; thật là 'bình yên' thì 28% bị đoán thành 'buồn'. Hai nhóm này cùng năng lượng thấp, khác nhau ở cảm xúc.
- Thật là 'căng-tối' thì 31% bị đoán thành 'vui'. Hai nhóm này cùng năng lượng cao.

Cả hai đều là nhầm trên trục cảm xúc (dương/âm), không phải trục năng lượng. Số liệu theo trục xác nhận điều này: mô hình đoán đúng phía năng lượng cao hay thấp với 0.647 (đoán mò = 0.5), nhưng đoán đúng phía cảm xúc dương hay âm chỉ 0.569.

Giải thích hợp lý, nhưng chưa kiểm chứng: năng lượng là thứ đo được trực tiếp từ nhịp, độ to và độ sáng của âm thanh, còn cảm xúc vốn khó hơn, phụ thuộc hòa âm, lời và ngữ cảnh văn hóa mà đặc trưng 30 giây này khó nắm hết.

### 4. Ablation: nhóm đặc trưng nào đáng giá?

Chạy lại LogReg (cùng cách chia theo artist) với từng nhóm riêng, và với bỏ từng nhóm:

| Cấu hình     | Số chiều | macro-F1      |
|--------------|----------|---------------|
| Tất cả       | 93       | 0.356 ± 0.041 |
| Chỉ MFCC     | 40       | 0.340 ± 0.039 |
| Chỉ chroma   | 24       | 0.310 ± 0.027 |
| Chỉ spectral | 22       | 0.365 ± 0.041 |
| Chỉ energy   | 4        | 0.340 ± 0.038 |
| Chỉ rhythm   | 3        | 0.337 ± 0.031 |
| Bỏ MFCC      | 53       | 0.385 ± 0.035 |
| Bỏ chroma    | 69       | 0.374 ± 0.049 |
| Bỏ spectral  | 71       | 0.347 ± 0.031 |
| Bỏ energy    | 89       | 0.357 ± 0.038 |
| Bỏ rhythm    | 90       | 0.332 ± 0.037 |

Cách đọc: chênh lệch giữa hai dòng bất kỳ (lớn nhất khoảng 0.075, giữa 'chỉ chroma' 0.310 và 'bỏ MFCC' 0.385) chỉ cỡ hai lần độ lệch chuẩn (0.03 đến 0.05), và 20 fold không độc lập hoàn toàn, nên **không xếp hạng chắc chắn được** các nhóm. Những điều vẫn nói được:
- Một nhóm đơn lẻ đã đạt 0.31 đến 0.37, gần bằng dùng cả 93 đặc trưng. Ngay cả nhóm rhythm chỉ có 3 đặc trưng cũng đạt 0.337. Các nhóm chứa thông tin chồng lấn nhau, hoặc tín hiệu trong dữ liệu này nông.
- Bỏ MFCC không làm điểm tụt (0.385 so với 0.356). Với hơn 1000 bài mà 93 chiều, có thể bớt chiều giúp ít nhiễu hơn, nhưng đây chỉ là giả thuyết, chênh lệch còn nằm trong nhiễu.
- Bỏ rhythm làm điểm giảm nhiều nhất (0.332), phù hợp với việc mô hình bắt được trục năng lượng tốt hơn trục cảm xúc. Cũng chỉ là gợi ý.

### 5. Mô hình dựa vào đặc trưng nào?

Nhìn 5 đặc trưng có hệ số dương lớn nhất của LogReg cho mỗi nhóm (huấn luyện trên toàn bộ dữ liệu chỉ để xem hệ số):
- **Vui:** onset_mean (+0.93), mfcc_mean_00 (+0.77), contrast_mean_04 (+0.65)
- **Căng-tối:** centroid_std_00 (+0.88), zcr_mean_00 (+0.74), contrast_std_04 (+0.72)
- **Buồn:** contrast_mean_02 (+0.87), chroma_mean_11 (+0.57), zcr_std_00 (+0.52)
- **Bình yên:** bandwidth_mean_00 (+0.55), zcr_std_00 (+0.51), mfcc_mean_01 (+0.47)

Vài dòng hợp trực giác: onset_mean cao (nhiều nốt khởi phát, nhịp dày) thiên về 'vui'; centroid_std và zcr_mean cao (âm sáng, nhiễu, thay đổi nhiều) thiên về 'căng-tối'. Nhưng nhiều đặc trưng tương quan với nhau, trọng số bị chia sẻ giữa chúng, nên hệ số chỉ gợi ý giả thuyết, không phải bằng chứng nhân quả.

### 6. Hạn chế nên nói thẳng khi vấn đáp
- Dữ liệu nhỏ: 1020 bài, 299 artist, nhóm nhỏ nhất 127 bài. Độ lệch chuẩn giữa fold lớn.
- Nhãn nhiễu: mood là tag do người dùng gán, và ta bỏ các bài có tag thuộc nhiều nhóm nên tập đang xét 'rõ ràng' hơn thực tế.
- Cách gom tag thành 4 nhóm là quyết định thiết kế, chưa kiểm chứng với người nghe.
- Chỉ dùng 30 giây giữa bài, giả định mood không đổi trong bài.
- Chưa tinh chỉnh siêu tham số. Nếu tinh chỉnh thì phải dùng nested cross-validation, không điểm sẽ lạc quan.


## Tóm tắt 30 giây (để nói khi vấn đáp)

- **Làm gì:** từ 1.020 bài Jamendo (299 artist), mỗi bài lấy 30 s ở giữa, tính 93 đặc trưng âm thanh (âm sắc, hòa âm, năng lượng, nhịp), rồi dùng Logistic Regression và Random Forest đoán 1 trong 4 nhóm mood.
- **Điểm đáng nói nhất:** chia train/test **theo artist**, không theo bài. Chia ngẫu nhiên làm điểm tăng ảo từ 0,356 lên 0,417 vì cùng artist nằm ở cả hai phía.
- **Kết quả trung thực:** hơn đoán mò rõ (macro-recall 0,362 so với 0,257, thắng 20/20 fold), hơn mốc "chỉ dùng thể loại" một chút (0,362 so với 0,319), và phần lớn lợi thế đó đến từ những bài không rõ thể loại.
- **Phát hiện phụ:** mô hình đoán trục năng lượng (0,647) tốt hơn trục cảm xúc (0,569).
- **Đừng nói quá:** dữ liệu nhỏ, nhãn do người dùng gán nên nhiễu, các nhóm đặc trưng không xếp hạng được chắc chắn.
