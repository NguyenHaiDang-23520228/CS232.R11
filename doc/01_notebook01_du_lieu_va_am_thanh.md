# Notebook 01 · Khám phá dữ liệu và hiểu âm thanh dưới dạng số

> File này giải thích notebook `notebooks/01_explore_data.ipynb`. Đây là notebook nền: hai notebook sau (02 và 03) đều dùng lại những thứ ở đây. Cuối file có phần "Tóm tắt 30 giây".

## Notebook này làm gì và vì sao phải làm

Trước khi làm bất cứ mô hình nào, ta cần trả lời hai câu hỏi rất cơ bản:

1. **Dữ liệu có gì, có đủ dùng không?** (Phần A: metadata, nhãn mood, nhãn thể loại.)
2. **Âm thanh thực chất là gì khi nằm trong máy tính?** (Phần B: lấy mẫu, waveform, spectrogram, mel.)

Bỏ qua bước này thì mọi con số ở các notebook sau đều khó giải thích. Ví dụ không hiểu vì sao phải chia dữ liệu theo artist, hoặc spectrogram từ đâu ra, thì không bảo vệ được kết quả khi vấn đáp.

## Dữ liệu: MTG-Jamendo

- **Nguồn:** bộ dữ liệu nhạc MTG-Jamendo, hàng chục nghìn bài nhạc từ nền tảng Jamendo, mỗi bài có các **tag** do người đăng gắn (thể loại, mood/theme, nhạc cụ).
- **Hai file metadata** dùng ở đây: `autotagging_moodtheme.tsv` (tag mood/theme) và `autotagging_genre.tsv` (tag thể loại). Mỗi dòng là một bài: 5 cột cố định (mã bài, mã artist, mã album, đường dẫn file, độ dài) rồi một số tag thay đổi, mỗi tag dạng `loại---tên` như `mood/theme---happy`.
- **Audio:** dùng bản `audio-low` (mono, nén MP3 thấp), chia thành 100 file `.tar`. Máy hiện có 10 file, tức 1.991 bài. Thư mục của bài là hai chữ số cuối của mã bài.
- **Giấy phép:** dữ liệu dùng cho nghiên cứu phi thương mại.

Vì số tag mỗi dòng khác nhau nên không đọc thẳng bằng `pd.read_csv` được; notebook có hàm tự tách 5 cột đầu rồi gom phần còn lại thành danh sách tag.

## Phần A: từ tag lộn xộn đến một bài toán rõ ràng

### Con số cơ bản

- 18.486 bài có tag mood/theme; trong đó 18.255 bài cũng có tag thể loại. Độ dài trung vị khoảng 203 giây, dài nhất 66 phút.
- Tag mood phổ biến nhất: `happy`, `film`, `energetic`, `relaxing`, `emotional`. Tag thể loại phổ biến: `soundtrack`, `electronic`, `ambient`, `pop`, `classical`.

### Gom mood thành 4 nhóm theo valence × arousal

Có hàng chục tag mood khác nhau, quá nhiều để học với dữ liệu nhỏ. Ta gom lại theo mô hình hai trục quen thuộc trong tâm lý học âm nhạc:

- **Valence**: cảm xúc tích cực hay tiêu cực.
- **Arousal**: năng lượng cao hay thấp.

Hai trục cho ra 4 nhóm: **vui** (tích cực, năng lượng cao), **căng/tối** (tiêu cực, năng lượng cao), **buồn** (tiêu cực, năng lượng thấp), **bình yên** (tích cực, năng lượng thấp). Gom như vậy có hai lợi: ít lớp hơn nên dễ học hơn, và có căn cứ lý thuyết để giải thích khi vấn đáp. Việc tag nào vào nhóm nào là **quyết định thiết kế của mình**, không phải sự thật có sẵn.

### Hai quyết định lọc dữ liệu (và lý do)

1. **Chỉ giữ bài có tag mood rơi vào đúng một nhóm.** Bài có tag lẫn nhiều nhóm có nhãn mơ hồ, sẽ làm nhiễu mô hình. Kết quả: trong số 10.185 bài có ít nhất một tag trong từ điển, 9.049 bài chỉ thuộc một nhóm, 1.068 bài thuộc hai nhóm, 68 bài thuộc ba nhóm.
2. **Bỏ bài chỉ mang tag ngữ cảnh.** Nhóm `mood/theme` của Jamendo trộn cảm xúc (`happy`, `sad`, `dark`) với **ngữ cảnh sử dụng** (`film`, `advertising`, `corporate`, `trailer`, `game`...). Loại thứ hai không nằm trên trục valence × arousal nên không xếp nhóm được. 8.301 bài không có tag nào xếp được. Đây là một **hạn chế của dữ liệu** cần nói thật.

Sau lọc còn 9.049 bài "sạch mood": vui 3.365, bình yên 2.779, buồn 1.542, căng/tối 1.363.

### Tập con thực sự dùng

Chỉ 1.020 trong 9.049 bài sạch mood có file audio trên máy. Đó là tập dùng ở cả notebook 02 và 03:

| Nhóm | Số bài |
|---|---|
| vui | 378 |
| bình yên | 331 |
| buồn | 184 |
| căng/tối | 127 |

299 artist khác nhau, trung bình 3,4 bài mỗi artist. Con số **299 artist** này quan trọng: nó là lý do phải chia dữ liệu theo artist ở notebook 02.

### Thể loại và vì sao phải để ý đến nó

Mood và thể loại gắn chặt: ví dụ nhóm "bình yên" phần lớn là ambient, nhóm "vui" nhiều electronic, nhóm "căng/tối" nhiều rock. Nếu mô hình âm thanh chỉ học ra được "đây là ambient" rồi suy ra "chắc là bình yên" thì nó **không hề học mood**.

Để kiểm soát, ta gộp 95 tag thể loại thành **8 thể loại lớn** (electronic, rock, pop, ambient, classical, jazz, hiphop, folk), và chỉ gán thể loại khi bài thuộc đúng một thể loại lớn; bài còn lại ghi `unknown`. Từ đó có một **mốc so sánh**: nếu chỉ biết thể loại, đoán mood đúng được bao nhiêu? Mô hình âm thanh phải hơn mốc này mới chứng minh được nó nghe ra thứ gì đó ngoài thể loại.

Một cảnh báo: notebook 01 tính mốc này ra macro-recall khoảng 0,44, nhưng đó là tính **trên chính dữ liệu dùng để học** (không chia train/test) nên bị lạc quan. Notebook 02 tính lại đúng cách ra **0,319**. Con số 0,44 ở notebook 01 chỉ để thấy "mood và thể loại gắn chặt", không dùng để so sánh.

## Phần B: âm thanh dưới dạng số

### Lấy mẫu (sampling)

Âm thanh là sóng liên tục. Máy tính đo biên độ $f_s$ lần mỗi giây, ra một dãy số $x[0], x[1], \dots$.

- Với $f_s = 22\,050$ Hz, mỗi giây có 22.050 số.
- **Nyquist:** chỉ biểu diễn đúng được tần số tới $f_s / 2$ (ở đây 11.025 Hz). Cao hơn thì bị méo (aliasing).
- Cả dự án thống nhất **mono, 22.050 Hz** để các bài so sánh với nhau được. (Notebook 03 hạ xuống 11.025 Hz cho nhẹ vì chỉ cần phần tần số chính của nhạc.)

### Waveform

Waveform là biên độ theo thời gian. Cho biết to/nhỏ theo thời gian, nhưng **khó thấy nốt nào, tần số nào đang vang**. Vì vậy cần bước tiếp theo.

### STFT: nhìn âm thanh theo (thời gian, tần số)

Cắt tín hiệu thành các **khung** ngắn dài $N$ mẫu, nhân với cửa sổ $w[n]$, biến đổi Fourier từng khung; hai khung liên tiếp cách nhau $H$ mẫu (`hop_length`):

$$X(m, k) = \sum_{n=0}^{N-1} x[n + mH]\, w[n]\, e^{-j 2\pi k n / N}$$

$|X(m,k)|$ cho biết **tại khung thứ $m$, có bao nhiêu năng lượng ở tần số thứ $k$**. Kết quả là một ma trận (tần số × thời gian), gọi là **spectrogram**.

**Đánh đổi quan trọng nhất (phải nắm chắc):**

- Độ phân giải tần số $\Delta f = f_s / N$. $N$ lớn thì phân biệt được hai tần số sát nhau.
- Độ dài khung là $N / f_s$ giây. $N$ lớn thì khung dài, **mờ về thời gian**.
- Không thể tốt cả hai cùng lúc. Với $f_s = 22\,050$ và $N = 2048$: $\Delta f \approx 10{,}8$ Hz, khung khoảng 93 ms, bước nhảy $H = 512$ khoảng 23 ms.

Notebook so sánh $N = 512$ và $N = 4096$ trên cùng tín hiệu: $N$ nhỏ thấy rõ **thời điểm** nốt chuyển nhưng các họa âm dính nhau; $N$ lớn tách họa âm rõ nhưng ranh giới thời gian nhòe.

Đơn vị **dB** ($20 \log_{10}$ biên độ) được dùng vì tai người nghe theo thang log; nếu để thang tuyến tính thì gần như chỉ thấy vài điểm sáng nhất.

### Mel-spectrogram

Tai người nhạy với chênh lệch ở tần số thấp hơn nhiều so với tần số cao: 100 đến 200 Hz nghe khác rõ, còn 8.000 đến 8.100 Hz gần như không. Thang **mel** mô phỏng điều này:

$$m = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)$$

Ta dùng bộ lọc tam giác dày ở tần số thấp, thưa ở tần số cao, để gom hơn một nghìn ô tần số của STFT thành chỉ **128 băng mel**. Ma trận nhỏ đi nhiều lần mà vẫn giữ phần thông tin quan trọng với việc nghe. Mel-spectrogram là đầu vào chuẩn của nhiều mô hình âm thanh và là bước đệm để tính MFCC ở notebook 02.

Notebook cũng vẽ mel-spectrogram của một bài thật cho mỗi nhóm mood, **chỉ để làm quen bằng mắt**. Một bài mỗi nhóm không chứng minh được gì; đừng kết luận kiểu "nhóm buồn thì tối màu".

## Những hạn chế cần nói thật

- **Nhãn mood do người đăng gắn** nên chủ quan, và Jamendo trộn cảm xúc với ngữ cảnh sử dụng.
- **Cách gom tag thành 4 nhóm** là quyết định thiết kế, chưa kiểm chứng với người nghe thật.
- **Chỉ 1.020 bài có audio**, nhóm nhỏ nhất chỉ 127 bài.
- **Mood và thể loại gắn chặt**, nên mọi đánh giá mô hình đều phải so với mốc "chỉ dùng thể loại".

## Tóm tắt 30 giây (để nói khi vấn đáp)

- **Dữ liệu:** MTG-Jamendo, tag mood và thể loại do người đăng gắn. Gom mood thành 4 nhóm theo valence × arousal; giữ bài chỉ thuộc một nhóm; còn 9.049 bài sạch mood, trong đó 1.020 bài có audio trên máy (299 artist).
- **Rủi ro chính của dữ liệu:** mood gắn chặt với thể loại, và nhiều bài cùng artist. Hai điều này quyết định cách đánh giá ở notebook 02.
- **Âm thanh thành số:** lấy mẫu, STFT ra spectrogram (đánh đổi tần số/thời gian $\Delta f = f_s / N$), rồi mel (128 băng theo tai người).
