# Báo cáo lab: chọn tracker cho 5 video

**Nhóm:** Phạm Văn Hoàng Anh Tú **Thành viên:** Phạm Văn Hoàng Anh Tú (2A202602507)

Detector cố định: `yolo26n.pt`, ảnh 640 px, Re-ID `osnet_x0_25_msmt17`. Không đổi các mục này trong bài nộp chính.

Chạy trên Kaggle (GPU T4), `boxmot==10.0.42`, `scripts/run_tracking.py` không sửa. File nộp: `runs/nop_bai/video_1.txt` … `video_5.txt`, chạy đủ frame (600 / 1050 / 837 / 900 / 750 frame).

## 1. Cấu hình đã chọn

Mỗi video: tracker bạn nộp, `conf`, `iou`, điều bạn **nhìn thấy** trên video, và một cấu hình đã thử rồi loại.

Số "ID" và "độ dài track" bên dưới đếm từ file kết quả (150 frame đầu, để so cùng đoạn với lần thử). Không phải số chấm điểm.

| Video | Tracker | conf | iou | Quan sát khi xem video | Đã thử nhưng loại |
|---|---|---|---|---|---|
| video_1 (quảng trường, tĩnh, ban ngày) | botsort | 0.3 | 0.5 | Người ở gần camera có hộp và giữ ID ổn. Người ở xa, phía cuối quảng trường, phần lớn không có hộp. Khi một nhóm người đi sát nhau, chỉ vài người trong nhóm có hộp. | bytetrack, conf 0.3: ít ID nhảy hơn (12 so với 25 lần) nhưng bỏ sót nhiều người hơn, HOTA / MOTA / IDF1 đều thấp hơn (bảng mục 2). |
| video_2 (phố đêm, tĩnh, rất đông) | bytetrack | 0.2 | 0.5 | Người ở nửa dưới ảnh giữ ID lâu (người đứng bên phải mang ID 4 từ frame đầu đến frame cuối). Đám đông nhỏ, tối ở cuối phố gần như không có hộp. Không thấy hộp trên xe hay cột đèn. | botsort, conf 0.3: 23 ID trong 150 frame so với 15 ID của bản nộp, track trung vị 38 frame so với 114 frame, nghĩa là ID bị cắt và đặt lại nhiều hơn. |
| video_3 (camera di động, ảnh nhỏ) | deepocsort | 0.25 | 0.5 | Người to, gần camera, hộp bám tốt. Số ID tăng nhanh (lên tới ID 170+ cuối video): mỗi lần camera rẽ hoặc người khuất sau người khác, track bị cắt và mở ID mới. | bytetrack, conf 0.3: ít hộp hơn (4,0 so với 5,8 hộp/frame), bỏ sót người ở xa; ID cũng không giữ lâu hơn (track trung vị 11 so với 13 frame). |
| video_4 (trong nhà, camera di chuyển) | botsort | 0.4 | 0.5 | Không thấy hộp trên ảnh phản chiếu ở sàn và kính. Người đi trước camera được giữ ID dài. Người đi sát mép trái, quá gần camera, có lúc không có hộp. | ocsort, conf 0.3: 21 ID so với 16, track trung vị 21 frame so với 57 frame, ID đổi nhiều hơn khi camera tiến lên. |
| video_5 (trên xe bus, giao lộ đông) | botsort | 0.3 | 0.5 | Người đi bộ rất nhỏ trên vỉa hè, nhiều người không có hộp. Trong file kết quả, nhiều track bị ngắt quãng (135 lần) và có 31 track ngắn dưới 15 frame, tức hộp hay mất rồi mở ID mới. | bytetrack, conf 0.3: ít ID hơn (20 so với 27) và ít bị ngắt quãng hơn, nhưng ít hộp hơn (4,5 so với 6,0 hộp/frame). Chọn botsort vì phủ được nhiều người hơn; đây là lựa chọn sát nút. |

Lưu ý: ở video_2, video_3, video_4, lần thử thay tracker cũng khác `conf`, nên khác biệt không chỉ do tracker.

## 2. Số liệu video_1

Dán bảng HOTA / MOTA / IDF1 do `scripts/evaluate_practice.py` in ra.

Bản nộp (botsort, conf 0.3, iou 0.5):

```
HOTA: nop_bai_video1-pedestrian    HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA
video_1                            29.46     18.095    48.223    18.596    78.888    51.153    82.992    83.666

CLEAR: nop_bai_video1-pedestrian   MOTA      MOTP      CLR_Re    CLR_Pr    CLR_TP    CLR_FN    CLR_FP    IDSW      MT   PT   ML
video_1                            19.811    81.471    21.759    92.306    4043      14538     337       25        8    12   42

Identity: nop_bai_video1-pedestrian IDF1     IDR       IDP
video_1                            29.354    18.137    76.941

Count:                             Dets      GT_Dets   IDs       GT_IDs
video_1                            4380      18581     52        62
```

So sánh với ByteTrack cùng ngưỡng (conf 0.3, iou 0.5):

| Tracker | HOTA | MOTA | IDF1 | IDSW | Recall (CLR_Re) | Precision (CLR_Pr) |
|---|---|---|---|---|---|---|
| botsort (nộp) | 29.46 | 19.81 | 29.35 | 25 | 21.8 % | 92.3 % |
| bytetrack | 26.91 | 17.29 | 25.71 | 12 | 17.9 % | 96.9 % |

`video_2` đến `video_5` không có nhãn trong gói lab. Không điền số cho các video đó.

## 3. Phân tích

**video_1 (có nhãn).** Cả hai tracker đều thấp vì phát hiện, không phải vì gán ID: precision trên 90 % nhưng recall chỉ khoảng 20 %, tức hơn 14.000 hộp người thật bị bỏ sót. Nhãn có chiều cao trung vị khoảng 100 px trên ảnh 1920×1080; khi thu về 640 px, người này chỉ còn khoảng 35 px nên YOLO nano ít bắt được. Hộp mà tracker giữ có chiều cao trung vị khoảng 300 px, đúng là chỉ người ở gần. BoT-SORT thắng ByteTrack ở HOTA, MOTA, IDF1 vì giữ được nhiều người hơn (4.043 so với 3.332 hộp đúng), dù đổi ID nhiều gấp đôi (25 so với 12 lần). AssA của hai tracker gần như bằng nhau (48,2 so với 48,1), nên Re-ID không giúp giữ danh tính tốt hơn ở cảnh này; phần hơn chủ yếu đến từ phát hiện.

**video_2 (chỉ bằng mắt).** Cảnh ban đêm, camera tĩnh trên cao, người nhỏ và tối. Ngoại hình của từng người khó phân biệt, nên Re-ID ít giúp; BoT-SORT trong lần thử còn làm ID ngắt quãng nhiều hơn (23 ID, 23 lần ngắt so với 15 ID, 2 lần ngắt). Camera không động nên mô hình chuyển động Kalman của ByteTrack dự đoán tốt, và việc ByteTrack ghép thêm các hộp điểm thấp giúp không mất người khi bị che một phần. Hạ `conf` xuống 0.2 cho thêm người ở vùng tối mà không thấy hộp giả trên nền.

**video_4 (chỉ bằng mắt).** Camera tiến về phía trước nên mọi người trong ảnh đều "trôi", chuyển động thuần dễ dự đoán sai. BoT-SORT có bù chuyển động camera và dùng thêm ngoại hình, nên track dài hơn hẳn OC-SORT (trung vị 57 so với 21 frame). `conf` 0.4 giúp bỏ các hộp trên ảnh phản chiếu ở sàn bóng và vách kính.

## 4. Nếu có thêm thời gian

Thử `conf` 0.15 cho video_1 và video_5, vì lỗi chính là bỏ sót người nhỏ ở xa, rồi xem precision tụt bao nhiêu. Với video_3, thử strongsort để xem Re-ID mạnh hơn có giảm số ID mới khi camera rẽ không.
