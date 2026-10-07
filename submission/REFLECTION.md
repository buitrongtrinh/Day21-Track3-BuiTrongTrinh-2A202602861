# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Fine-tune đạt 0.985 trên tác vụ mà vẫn FAILED, và lý do không phải là "quên". Khi đọc từng output
trên 15 câu hỏi phổ thông, cả 15 đều là JSON triage, kể cả câu "1 km bằng bao nhiêu mét?" bị gán
`"intent": "conversion"`. Điều ngạc nhiên thứ hai là chỉ 12 dòng replay (5%) đã sửa được lỗi đó
hoàn toàn: 0% JSON, regression 0.656. Tôi từng nghĩ chống quên thảm hoạ cần lượng dữ liệu lớn.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Không phải ở training. Cả pipeline NB2→NB6 chỉ mất 12.4 phút trên RTX 4050 với model 0.8B.
Thời gian dồn vào việc *chẩn đoán* sau phán quyết FAILED: viết probe để lưu output regression
(NB5 chỉ lưu điểm tổng), sinh lại output của (b) để so từng mẫu, rồi thiết kế thí nghiệm replay
sao cho câu hỏi replay không trùng tập eval. Tôi dự đoán phần khó nhất là cấu hình LoRA, nhưng
thực tế cấu hình theo deck chạy đúng ngay lần đầu.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Tôi tin rằng "LoRA quên ít hơn full fine-tune" có nghĩa là LoRA an toàn về hồi quy. Số đo cho thấy
LoRA r=16 ở cấu hình chuẩn vẫn làm regression tụt từ 0.644 xuống 0.100, và r=64 còn xuống 0.000.
Adapter nhỏ không ngăn được model học một quy tắc định dạng phủ quyết mọi thứ. Thứ bảo vệ hồi quy
là *thành phần dữ liệu*, không phải số tham số ít.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng Claude Code (một agent chạy trong terminal) để cài môi trường vào conda `lab-vin-env`,
chạy NB1–NB6, viết `scripts/extra_experiments.py` (probe, qual, replay, quét rank) và soạn bản
nháp report này từ các file trong `results/`. Những chỗ nó sai hoặc phải sửa:
- Ban đầu nó tạo một `.venv` riêng bằng `uv` thay vì dùng env conda có sẵn. Tôi phải dừng nó lại.
- Nó đặt tên run replay theo phần trăm (`replay_20pct`) trong khi danh sách câu hỏi chỉ có 24 câu,
  nên run đó thực tế chỉ có 24 dòng (~10%). Lỗi được phát hiện khi đối chiếu số dòng. Sau đó
  script được sửa để dừng hẳn nếu không đủ câu hỏi và đặt tên theo số dòng thật, rồi chạy lại.
  Kết quả lần đầu được giữ trong `results/replay_first_attempt.json` vì nó vô tình cho thấy độ
  nhiễu giữa các lần chạy.
- Bản nháp đầu viết "(b) mặc định urgency `cao` (31/50 lỗi)", trộn hai con số khác nhau. Đếm lại
  thì đúng là (b) đoán `cao` cho 48/50 ticket. Bài học: mọi con số do AI viết đều phải đối chiếu
  lại với file.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Xây tập regression **từ chính lưu lượng thật** của khách hàng: các câu *không phải* tác vụ mà
model sẽ gặp (hỏi giờ mở cửa, chào hỏi, khiếu nại lạc đề), cỡ ≥100 câu, kèm thang chấm "có áp
schema sai chỗ không". Sau đó mới đo baseline (b). Lab này cho thấy tập target cho biết model
*làm được gì*, còn chỉ tập regression mới cho biết model *đã phá gì*. Với 15 câu, độ nhiễu
(~0.02) đã bằng ngưỡng của cổng.
