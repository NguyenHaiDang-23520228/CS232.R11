# Notebook 03 · Nhận diện bài hát bằng "dấu vân tay" âm thanh (kiểu Shazam)

> File này giải thích notebook `notebooks/03_fingerprint_shazam.ipynb`. Nên đọc `01_...md` và `02_...md` trước. Cuối file có phần "Tóm tắt 30 giây". Mọi con số trong file này lấy từ lần chạy đã lưu trong notebook, không ước lượng trước.

## Bài toán

> Cho một đoạn thu ngắn (2 đến 10 giây), có thể bị ồn, bị nén MP3, hoặc lẫn nhạc khác. Tìm xem đó là **bài nào trong kho nhạc**, và chính xác tới đâu.

Đây là bài toán mà ứng dụng Shazam giải. Notebook này tự cài một **bản đơn giản hóa** của thuật toán trong bài báo của Avery Wang (2003), "An Industrial-Strength Audio Search Algorithm", rồi đo xem nó chạy tốt tới đâu trên dữ liệu thật.

## Vì sao làm bài này (nói thật)

1. **Đáp án đúng/sai rõ ràng.** Ở notebook 02, nhãn mood do người gắn nên mơ hồ, phải rất cẩn thận mới đánh giá được. Ở đây, bài nào là bài nào thì không có gì để cãi, nên kết quả đo thẳng.
2. **Cùng dữ liệu, ý tưởng ngược hẳn.** Notebook 02 *nén cả đoạn thành vài chục con số* để mô hình học. Notebook 03 *giữ lại vài điểm nổi bật nhất* và dùng như chữ ký để tra bảng. So sánh hai cách giúp hiểu khi nào cần học máy, khi nào không cần.
3. **Nhiều thứ để tự đặt thí nghiệm và giải thích:** nhiễu gì làm hỏng, đoạn ngắn tới đâu thì chịu được, tham số nào quan trọng, kho lớn lên thì sao. Đúng kiểu "tạo bài toán, giải quyết, giải thích".

| | Notebook 02 (đoán mood) | Notebook 03 (nhận diện bài) |
|---|---|---|
| Bài toán | phân loại: đoán nhãn cho bài chưa từng thấy | tra cứu: tìm bài đã có trong kho |
| Học từ dữ liệu | có | **không** (không có mô hình, không huấn luyện) |
| Đặc trưng | 93 số tóm tắt cả đoạn | hàng chục nghìn mã băm từ đỉnh của phổ |
| Đáp án | mơ hồ | duy nhất |
| Cách đo | macro-recall so với đoán mò, chia theo artist | top-1 theo độ dài đoạn và loại nhiễu, ngưỡng từ chối bài lạ |

## Ý tưởng chính

**Vì sao không so sánh thẳng âm thanh?** Kho có hàng triệu bài, mỗi bài hàng triệu mẫu: so từng mẫu là bất khả thi. Hơn nữa nhiễu, nén MP3, âm lượng khác làm mọi con số đổi, dù tai người vẫn nhận ra bài.

**Ý tưởng của Shazam:** chỉ giữ lại **những điểm nổi bật nhất** của phổ (ít bị nhiễu làm mất), biến chúng thành **mã số**, rồi tra bảng như tra từ điển. Gồm 4 bước.

### Bước 1. Spectrogram

Giống notebook 01, nhưng dùng $f_s = 11\,025$ Hz, $N = 1024$, bước nhảy 256 mẫu (mỗi khung nhảy 23 ms, mỗi ô tần số rộng 10,8 Hz). Trước khi tính phổ, tín hiệu được **chuẩn hóa theo RMS** để thang dB không phụ thuộc đoạn thu to hay nhỏ.

### Bước 2. Đỉnh của phổ (constellation map)

Một **đỉnh** là điểm có năng lượng **lớn nhất trong vùng lân cận** (mặc định 21 ô tần số × 41 khung) và cao hơn ngưỡng. Dùng `maximum_filter` của SciPy để tìm.

Với một đoạn nhạc 10 s, chỉ khoảng 229 điểm được giữ lại: **0,10% số ô của phổ**, tức khoảng 23 đỉnh mỗi giây. Vẽ ra trông như bản đồ sao, nên gọi là *constellation map*. Toàn bộ thông tin nhận diện nằm trong mấy chấm này.

**Vì sao đỉnh bền với nhiễu?** Nhiễu thêm một lớp năng lượng đều và nhỏ lên toàn phổ; những điểm nổi hẳn lên (một nốt, một tiếng trống) vẫn nổi. Điểm lưng chừng thì có thể đổi chỗ, nên ta chấp nhận mất một phần đỉnh.

### Bước 3. Ghép cặp đỉnh thành hash

Một đỉnh đơn chỉ cho một con số tần số (khoảng 512 giá trị), xuất hiện ở mọi bài, nên tra bảng sẽ ra hàng nghìn kết quả vô nghĩa. Ta ghép **cặp** hai đỉnh gần nhau và mã hóa:

$$\text{hash} = (f_1, f_2, \Delta t), \qquad \Delta t = t_2 - t_1$$

- $f_1, f_2$ là ô tần số của hai đỉnh (9 bit mỗi cái).
- $\Delta t$ là khoảng cách thời gian giữa hai đỉnh, tính theo khung (6 bit).
- Gộp thành số nguyên **24 bit**: khoảng 16,7 triệu giá trị có thể, đặc trưng hơn đỉnh đơn khoảng 33.000 lần.

Hash **không chứa thời điểm tuyệt đối**, chỉ chứa khoảng cách tương đối giữa hai đỉnh. Nhờ vậy đoạn thu bắt đầu ở giây nào cũng cho **cùng hash** với đoạn tương ứng trong bài gốc. Thời điểm tuyệt đối $t_1$ được cất riêng để dùng ở bước bỏ phiếu.

Cách ghép trong bản này (đơn giản hóa so với bài báo): mỗi đỉnh ghép với `K = 5` đỉnh kế tiếp theo thời gian, miễn là cách không quá 63 khung. Kết quả: khoảng **48 hash mỗi giây nhạc**.

### Bước 4. Xây chỉ mục và bỏ phiếu theo độ lệch thời gian

**Chỉ mục (bảng tra).** Với mỗi bài trong kho: tính hash cho cả bài, lưu mọi bộ ba (hash, thời điểm $t_1$ trong bài, mã bài), rồi **sắp xếp theo hash**. Giống chỉ mục của công cụ tìm kiếm: từ hash tra ra "bài nào, ở thời điểm nào, có hash này". Cài bằng ba mảng numpy song song, tra bằng `np.searchsorted`.

**Truy vấn.** Tính hash của đoạn thu (kèm thời điểm $t_q$ của từng hash trong đoạn). Tra từng hash vào chỉ mục, mỗi lần khớp là một **hit** (bài $b$, thời điểm $t_b$ trong bài). Từ hit tính

$$\Delta = t_b - t_q$$

Nếu đoạn thu thật sự cắt từ bài $b$, bắt đầu ở giây $s$, thì **mọi hit đúng đều có cùng $\Delta = s$**. Hit sai (hash trùng hợp, hash phổ biến) có $\Delta$ rải ngẫu nhiên. Vậy với mỗi bài ta đếm số hit rơi vào cùng một $\Delta$: **điểm của bài = số phiếu cao nhất ở một $\Delta$**. Bài điểm cao nhất thắng.

Bonus: $\Delta$ của bài thắng cho biết **đoạn thu bắt đầu ở giây nào của bài**.

**Ví dụ trong notebook:** đoạn 5 s cắt từ bài #0 bắt đầu ở giây 60, thêm ồn 10 dB. Có 275 hash, tra ra 2.516 hit trên 702 bài khác nhau, nhưng bài #0 được **40 phiếu** tại đúng $\Delta = 60{,}00$ s, còn bài đứng thứ hai chỉ **3 phiếu**.

## Kho nhạc và chỉ mục

- **Kho:** 1.020 bài (cùng tập với notebook 02), tổng 62,2 giờ. Không dùng nhãn mood.
- **Bài ngoài kho:** 971 bài còn lại trong thư mục audio, để thử khi hệ thống nghe bài lạ.
- **Chỉ mục:** 10,7 triệu hash (khoảng 10.500 hash mỗi bài, 48 hash mỗi giây nhạc), chiếm **128 MB**, dựng mất khoảng 80 giây trên 4 nhân CPU.
- Hash không đều: 41% giá trị hash chỉ xuất hiện đúng một lần, nhưng có hash xuất hiện tới 900 lần (thường là cặp tần số thấp sát nhau, kiểu tiếng bass đều). Hash phổ biến một mình không nói lên gì; bước bỏ phiếu theo $\Delta$ chính là thứ lọc chúng.

## Thí nghiệm chính

### Thiết kế

- 200 đoạn truy vấn: mỗi đoạn cắt từ một bài ngẫu nhiên, **bắt đầu ở thời điểm ngẫu nhiên** (không canh theo khung nào).
- Độ dài 2, 5, 10 giây. Đoạn ngắn là phần đầu của đoạn dài, nên so sánh ghép cặp được.
- 8 điều kiện: sạch, ồn trắng (SNR 10, 0, -5 dB), nén MP3 (khoảng 40 kbps), micro điện thoại (lọc 300 đến 3400 Hz), nhạc nền (10 và 3 dB).
- Độ đo: **top-1** = bài điểm cao nhất đúng là bài đã cắt.
- Lưu ý: đoạn "sạch" cắt từ chính bản giải mã đã lập chỉ mục, nên là **mốc trên**, dễ nhất, không phải kết quả khi thu qua loa và micro.

SNR (tỷ lệ tín hiệu trên nhiễu) tính bằng $\text{SNR} = 10\log_{10}(P_{\text{tín hiệu}} / P_{\text{nhiễu}})$. SNR 0 dB nghĩa là nhiễu to bằng nhạc.

### Kết quả: top-1 (%)

| Điều kiện | 2 s | 5 s | 10 s |
|---|---|---|---|
| sạch | 99,0 | 100 | 100 |
| nén MP3 (~30 đến 40 kbps) | 98,5 | 100 | 100 |
| micro điện thoại | 98,0 | 100 | 100 |
| ồn 10 dB | 72,0 | 91,0 | 96,5 |
| ồn 0 dB | 24,0 | 50,0 | 66,5 |
| ồn -5 dB | 3,5 | 17,5 | 35,0 |
| nhạc nền 10 dB | 80,0 | 82,0 | 82,5 |
| nhạc nền 3 dB | 61,0 | 63,0 | 63,0 |

Với 200 truy vấn, sai số lấy mẫu từ vài phần trăm (khi sát 100%) đến khoảng ±7 điểm phần trăm (khi quanh 50%).

### Cách đọc

- **Đoạn dài hơn thì chính xác hơn:** có nhiều hash hơn để bỏ phiếu.
- **MP3 và micro điện thoại gần như không làm giảm.** Đo trực tiếp trên một đoạn 5 s thử trong notebook thì chỉ còn khoảng một nửa số hash của đoạn trùng với bài gốc (49% cho MP3, 56% cho lọc micro), nhưng như vậy đã quá đủ để bỏ phiếu: không cần giữ hết hash, chỉ cần đủ nhiều hash đúng cùng chỉ về một vị trí.
- **Nhiễu trắng mạnh làm hỏng nhiều nhất.** Nó biến những ô vốn yên lặng thành đỉnh giả, nên nhiều hash không còn khớp. Ở 0 dB chỉ còn khoảng 4% hash đúng (nhưng vẫn cao hơn hẳn mức trùng hợp dưới 1%), và đoạn 5 s vẫn nhận đúng 50%.
- **Nhạc nền: top-1 thấp hơn top-2 nhiều.** Với nhạc nền 10 dB, đoạn 5 s: top-1 là 82% nhưng bài đúng nằm trong top-2 tới **97%**. Lý do không phải hệ thống "lỡ", mà là **cả hai bài đều có mặt trong âm thanh**, hệ thống nhận ra cả hai và chọn bài có nhiều đỉnh nổi hơn. Phép thử trộn hai bài ở cuối notebook cho thấy rõ: cả hai bài đều nằm trong top-2.

Thông số vận hành: mỗi truy vấn (tính dấu vân tay và tra cứu) mất khoảng **4 ms** trên một nhân CPU. Khi đoán đúng bài, sai số vị trí bắt đầu đoạn có trung vị **6 ms**.

### Một chi tiết thú vị: đúng bài nhưng sai vị trí

Khoảng **10%** lần nhận đúng bài nhưng chỉ ra sai vị trí (lệch hơn 0,1 s). Nguyên nhân: nhạc điện tử hay lặp **loop** giống hệt, nên nội dung đoạn thu xuất hiện ở nhiều chỗ trong bài và nhận phiếu ở nhiều $\Delta$. Hình phiếu bầu của bài #0 trong notebook cho thấy rõ: cột cao nhất đúng ở 60 s, nhưng có thêm các cột phụ cao ở khoảng 27, 93 và 127 s.

## Bài lạ: khi bài không có trong kho

Ngoài đời hệ thống hay nghe bài chưa có trong kho. Nó vẫn trả về "bài điểm cao nhất" nhưng điểm sẽ thấp. Cần một **ngưỡng $\theta$**: điểm $\ge \theta$ thì tin, còn lại trả lời "không biết".

Thí nghiệm: 200 đoạn 5 s từ bài ngoài kho, so điểm của chúng với điểm của truy vấn trong kho cùng điều kiện.

| Ngưỡng $\theta$ | Sạch: trả lời đúng (%) | Sạch: nhận nhầm bài lạ (%) | Ồn 10 dB: trả lời đúng (%) | Ồn 10 dB: nhận nhầm bài lạ (%) |
|---|---|---|---|---|
| 3 | 100 | 85,5 | 91,0 | 87,0 |
| 5 | 100 | 9,5 | 85,0 | 8,0 |
| 8 | 100 | 1,5 | 77,5 | 0,5 |
| 10 | 100 | 1,0 | 72,0 | 0,5 |
| 20 | 99,0 | 1,0 | 43,0 | 0,5 |
| 50 | 76,5 | 0,5 | 6,0 | 0,0 |

**Cách đọc:** điểm của bài lạ dồn về mức rất thấp (vài phiếu, ngang mức trùng hợp). Ngưỡng khoảng 8 đến 10 phiếu làm nhận nhầm bài lạ chỉ còn 0,5 đến 1,5% mà vẫn giữ toàn bộ câu trả lời đúng khi sạch. Giá phải trả là **bỏ sót** những truy vấn đúng nhưng điểm thấp (ồn nhiều, đoạn ngắn). Chọn $\theta$ là chọn điểm cân bằng: ứng dụng nào sợ trả lời sai thì đặt cao. Ngưỡng quá thấp (3 phiếu) thì vô nghĩa vì gần như mọi bài lạ đều lọt qua.

## Giới hạn: đổi tốc độ

Hash chứa **tần số tuyệt đối** và **khoảng cách thời gian tuyệt đối**. Nếu bài bị phát nhanh hơn $x\%$, mọi tần số nhân lên và mọi khoảng cách co lại, nên hash đổi hết.

| Đổi tốc độ | -5% | -2% | -1% | -0,5% | 0% | +0,5% | +1% | +2% | +5% |
|---|---|---|---|---|---|---|---|---|---|
| top-1 (%) | 0,5 | 1,0 | 9,5 | 72,5 | 100 | 77,5 | 12,5 | 0,0 | 0,0 |

Chỉ cần lệch **1%** là độ chính xác sụp. Đây là điểm yếu có thật của hướng hash tuyệt đối, nên nói rõ khi vấn đáp. (Shazam bản chính xử lý tốt hơn nhờ thêm các mẹo khác; đoạn thu thực tế cũng hiếm khi bị đổi tốc độ.)

## Tham số nào quan trọng? (ablation)

Vặn **từng tham số một**, giữ các tham số khác ở mặc định. Để chạy nhanh: kho 300 bài, 100 truy vấn 5 s, hai điều kiện (ồn 10 dB và nhạc nền 10 dB). Với 100 truy vấn, sai số khoảng ±6 đến 10 điểm phần trăm, nên **chỉ tin những chênh lệch lớn**.

| Cấu hình | Hash/giây nhạc | Chỉ mục (MB) | Top-1 ồn 10 dB (%) | Top-1 nhạc nền 10 dB (%) |
|---|---|---|---|---|
| mặc định (21×41, floor 0, K=5, dt_max=63) | 47,6 | 38,4 | 95 | 83 |
| lân cận nhỏ 15×31 | 77,8 | 62,8 | 97 | 85 |
| lân cận lớn 31×61 | 24,1 | 19,5 | 98 | 84 |
| floor 10 dB | 45,7 | 36,8 | 98 | 86 |
| floor 20 dB | 41,1 | 33,2 | 96 | 87 |
| K=2 | 16,0 | 12,9 | **82** | 86 |
| K=10 | 103,3 | 83,4 | 98 | 83 |
| dt_max=15 | 39,6 | 31,9 | 94 | 83 |
| dt_max=31 | 46,6 | 37,6 | 96 | 83 |

Điều rút ra được:

- **K (số cặp mỗi đỉnh)** là núm rõ nhất: K=2 tụt từ 95% xuống 82% ở ồn 10 dB; K=10 lên 98% nhưng chỉ mục to gấp 2,2 lần.
- **Mật độ đỉnh** (vùng lân cận) chủ yếu đổi **kích thước chỉ mục** (63, 38, 20 MB) mà độ chính xác không đổi rõ (97%, 95%, 98%). Đỉnh thưa nghĩa là rẻ hơn mà không mất nhiều.
- `floor` và `dt_max`: không thấy khác biệt vượt sai số.
- Nhạc nền 10 dB hầu như **không phụ thuộc tham số** (83 đến 87% ở mọi cấu hình): giới hạn nằm ở chỗ "có hai bài trong âm thanh", không phải ở cách chỉnh.

## Kho lớn dần thì sao?

Kho càng nhiều bài thì càng nhiều hash trùng hợp, nên điểm bài sai tăng dần, còn điểm bài đúng giữ nguyên. Đo trực tiếp bằng cách cắt chỉ mục chỉ giữ `n` bài đầu (ồn 10 dB, đoạn 3 s, 100 truy vấn):

| Số bài trong kho | 50 | 100 | 250 | 500 | 750 | 1.020 |
|---|---|---|---|---|---|---|
| top-1 (%) | 88 | 88 | 86 | 85 | 85 | 84 |
| điểm trung vị bài đúng | 11 | 11 | 11,5 | 12 | 12 | 12 |
| điểm trung vị bài sai | 1 | 1,5 | 2 | 2 | 3 | 3 |
| ms mỗi truy vấn | 2,7 | 2,5 | 2,5 | 2,5 | 2,6 | 2,7 |

Top-1 giảm nhẹ (88 xuống 84) nằm trong sai số của 100 truy vấn, nhưng điểm bài **sai** tăng đều từ 1 lên 3 trong khi điểm bài đúng giữ quanh 11 đến 12: đúng chiều hướng dự đoán. Kho của ta quá nhỏ để rút ra kết luận cho quy mô hàng chục triệu bài; chỉ thấy được **chiều hướng**.

## So sánh với notebook 02

| | Notebook 02: đoán mood | Notebook 03: nhận diện bài |
|---|---|---|
| Ý tưởng | nén cả đoạn thành 93 số, để mô hình học | giữ vài điểm nổi bật, biến thành mã, tra bảng |
| Học máy | có | không |
| Đánh giá | khó: nhãn chủ quan, phải chia theo artist, so với nhiều mốc | dễ: đúng hay sai bài |
| Hiểu vì sao đúng/sai | hạn chế | rõ: nhìn thẳng vào phiếu bầu |
| Hợp khi | cần *khái quát* sang bài chưa từng thấy | cần *khớp chính xác* với bài đã biết |
| Điểm yếu lớn | độ chính xác vừa phải, nhãn mơ hồ | không khái quát: bài mới, bản cover, đổi tốc độ là mất |

Nói ngắn gọn: notebook 02 trả lời **"bài này nghe giống cái gì"**, notebook 03 trả lời **"bài này chính xác là bài nào"**.

## Hạn chế cần nói thật khi vấn đáp

- **Kho nhỏ** (1.020 bài so với hàng chục triệu của hệ thống thật). Chỉ thấy chiều hướng khi kho lớn dần.
- **Đoạn "sạch" là bản sao kỹ thuật số** của đúng bản đã lập chỉ mục. Thực tế còn tiếng vang phòng, loa, micro; ồn trắng và lọc dải chỉ là mô phỏng đơn giản.
- **Chỉ nhận ra bài gốc.** Bản live, remix, cover, đổi tông là bài toán khác, hướng này không nhận ra.
- **Bản đơn giản hóa:** cách ghép cặp (K đỉnh kế tiếp), hash 24 bit, không xử lý lệch khung bằng dung sai; Shazam thật khác nhiều chi tiết.
- **Số truy vấn có hạn** (200 cho thí nghiệm chính, 100 cho ablation): chênh lệch nhỏ không đủ kết luận.
- **Trùng lặp trong dữ liệu:** Jamendo có thể chứa các bài rất giống nhau; nếu có, một số "đoán sai" thật ra là hai bản gần như trùng. Có thể một phần 0,5 đến 1% bài lạ bị nhận nhầm ở ngưỡng cao cũng thuộc trường hợp này, nhưng ta chưa kiểm tra.
- **Số liệu phụ thuộc phiên bản thư viện** (librosa, soundfile khi nén MP3, SciPy), nên chạy lại trên máy khác có thể lệch vài điểm phần trăm.

## Cách chạy lại

1. Chạy notebook 01 trước để có `data/index_mood.csv`, và có sẵn thư mục audio (đặt biến môi trường `JAMENDO_AUDIO_DIR` nếu không để ở `data/audio-low`).
2. Mở `notebooks/03_fingerprint_shazam.ipynb`, chạy từ trên xuống. Lần đầu mất khoảng 10 đến 20 phút tùy máy (chủ yếu là dựng chỉ mục và lặp thí nghiệm).
3. Kết quả trung gian được lưu trong `data/fp_cache/` (không bị commit), nên lần sau chạy lại rất nhanh. Muốn tính lại từ đầu thì xóa thư mục đó.
4. Muốn chạy thử nhanh, đặt `MAX_CATALOG = 100` ở ô cấu hình.

## Tóm tắt 30 giây (để nói khi vấn đáp)

- **Bài toán:** nghe một đoạn 2 đến 10 giây, tìm ra là bài nào trong kho 1.020 bài. Không huấn luyện mô hình, chỉ tra bảng.
- **Cách làm:** spectrogram, giữ các **đỉnh** nổi bật (0,1% số ô), ghép **cặp đỉnh** thành hash 24 bit (tần số 1, tần số 2, khoảng cách thời gian), lưu vào bảng tra; khi truy vấn thì **bỏ phiếu theo độ lệch thời gian**: hit đúng cùng một độ lệch, hit sai rải ngẫu nhiên.
- **Kết quả:** đoạn 5 s sạch, nén MP3 hoặc qua micro điện thoại nhận đúng **100%**; ồn 10 dB: 91%; ồn 0 dB: 50%; một truy vấn mất 4 ms.
- **Điểm yếu:** đổi tốc độ 1% là sụp (còn 9 đến 13%); nhiễu rất mạnh; có hai bài cùng lúc thì chọn một bài (nhưng bài kia thường nằm ngay top-2).
- **Giá trị so với notebook 02:** bài toán có đáp án đúng-sai rõ, đối lập với đoán mood vốn mơ hồ.
