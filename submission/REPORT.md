# Lab 21 — Evaluation Report

**Họ tên**: Bùi Trọng Trịnh  **MSSV**: 2A202602861  **Ngày**: 2026-10-07
**Tier**: `LAPTOP` + `BASE_MODEL` override  **Base model**: `Qwen/Qwen3.5-0.8B`  **GPU thực tế**: NVIDIA GeForce RTX 4050 Laptop, 6 GB (sm_89, bf16 native)

> Mọi con số dưới đây lấy từ file trong `results/` (đường dẫn ghi cạnh từng bảng).
> Môi trường: conda `lab-vin-env`, Python 3.13, torch 2.14.0+cu130, transformers 5.17.0,
> TRL 1.14.2, PEFT 0.21.2, bitsandbytes 0.50.2.

**TL;DR.** Fine-tune thắng baseline (b) rất xa trên tác vụ (target **0.985** so với **0.500**),
nhưng **trượt cổng hồi quy**: regression tụt từ **0.644 → 0.100**. Probe cho thấy nguyên nhân
không phải là model "quên kiến thức". Thứ sập là **định dạng đầu ra**: 15/15 câu hỏi phổ
thông bị trả lời bằng JSON triage. Thí nghiệm bổ sung chỉ trộn **12 dòng replay tự chưng cất
(5%)** đã kéo regression về **0.656** mà vẫn giữ target **0.98**. Phán quyết chính thức của
run `correct` vẫn là **FAILED**: bản đó không nên deploy.

---

## 0. Lựa chọn thí nghiệm & lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `Qwen/Qwen3.5-0.8B` (tier `LAPTOP`, giữ batch/length của tier) | Máy có RTX 4050 **6 GB**. Theo HARDWARE-GUIDE, Qwen3.5-2B cần ~5 GB cho bf16 LoRA, nên không còn chỗ cho generate batch 4 và cho ba run liên tiếp ở NB4. Với 0.8B, peak VRAM đo được là **1.97 GB** (`runs.csv`), chạy local được toàn bộ NB1–NB6 mà không cần Colab. |
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON 4 trường | Có thang chấm khách quan (độ chính xác từng trường), không cần LLM judge. Không đổi tập eval, checksum giữ nguyên. |
| Prompt (b) | **Giữ nguyên** `OPTIMIZED_PROMPT` gốc (SHA `719e74d3b6232053`) | Không sửa, không làm mạnh lên hay yếu đi. |
| Mask | `assistant-only` | Corpus là JSON trần, không có trace suy luận, nên `masked-think`/`response-only` cho mask y hệt (NB3 có cảnh báo). |

Hệ quả của việc chọn model nhỏ: base 0.8B yếu (chính baseline (b) mới đạt 0.5), nên còn nhiều
khoảng trống để fine-tune thắng. Mặt khác, model nhỏ có ít "dư địa" để vừa học tác vụ vừa giữ
hành vi cũ. Mục 5 cho thấy đúng điều đó.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`data/train_seed.jsonl`) |
| Train / val | **225 / 25** (seed 42) |
| `max_length` | **1024** (giữ theo tier). p95 đo được là **98**, p99 = 100, max = 101; gợi ý của lab là **256** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch → **58 optimizer step** (batch 1 × grad_accum 8 = batch hiệu dụng 8 < 32) |
| LR / scheduler | 1e-4 (10× thang full-FT), cosine, warmup 6 step, bf16, gradient checkpointing |

**Vì sao giữ `max_length=1024` dù p95 gợi ý 256:** với `per_device_batch=1` và `packing=False`
thì không có padding. Chuỗi dài nhất chỉ 101 token, nên truncation không bao giờ kích hoạt:
1024 hay 256 cho cùng một tensor đầu vào và cùng chi phí tính toán. Giữ số của tier để khỏi
phải sửa `config.py`. Nếu dùng batch > 1 hoặc bật packing thì nên hạ xuống 256.

**Template có giữ khối `<think>` không?** **Có.** `results/template_check.json`:
`verdict = "reasoning preserved — safe to train on traces"`. Chuỗi render thử vẫn còn nguyên
`<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>`. Với corpus này, generation prompt của
Qwen3.5 tự chèn khối rỗng `<think>\n\n</think>` *trước* câu trả lời, nên khối đó rơi vào
phần bị che (xem masked preview bên dưới). Loss chỉ tính trên JSON + `<|im_end|>`.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.3936** (37/94 token, mẫu đầu). Trên toàn tập train: 8564/20951 = 40.9% |
| Câu trả lời nằm trong loss | **true** |
| Câu hỏi KHÔNG nằm trong loss | **true** |

Đoạn **được** tính loss (`supervised_preview`):

```
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **bị che** (`masked_preview`). System prompt, ticket và khối think rỗng đều nằm ngoài loss:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453. Cho tôi trả lại. Đã 3 ngày rồi. Cho tôi hỏi.<|im_end|>
<|im_start|>assistant
<think>

</think>
```

Để đối chiếu, chế độ `everything` cho 94/94 (100%), tức prompt cũng bị tính loss.
`scripts/check_mask_agreement.py` xác nhận thêm vì sao lab phải tự token hoá: template Qwen3.5
không có marker `{% generation %}`, nên mask của TRL ở mức tokenizer là **0 token**.

---

## 3. Ba baseline (NB2 đo TRƯỚC khi train) + fine-tune (NB5)

*(results/baselines_frozen.json, results/verdict.json)*

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.644 | 0.000 | 901.6 |
| (b) base + optimized prompt | **0.500** | 0.644 | 1.000 | 245.4 |
| (c) LoRA fine-tune (`correct`, naive prompt) | **0.985** | **0.100** | 1.000 | 353.6 |

**(b) có thật sự mạnh hơn (a) không?** **Có**, 0.000 → 0.500. Với prompt ngây thơ, base 0.8B
viết văn xuôi, không ra JSON nào (format 0.000, latency cao vì sinh tới trần 160 token).
**Không sửa `OPTIMIZED_PROMPT`.**

**(b) sai ở đâu?** Đếm lỗi từng trường trên 50 mẫu (`results/field_errors.json`):

| | intent | urgency | product | sentiment |
|---|---|---|---|---|
| (b) prompt | 38 | 31 | 3 | 28 |
| (c) fine-tune | 0 | 3 | 0 | 0 |

`product` gần như luôn đúng vì chỉ cần chép lại từ ticket. Ba trường còn lại là **ánh xạ ngữ
nghĩa riêng của corpus**: "Thiếu phụ kiện" → `san_pham_loi`, "Khi nào tiện" → `thap`,
"Cảm ơn shop nhiều" → `tich_cuc`. Một ví dụ few-shot không truyền tải được các ánh xạ này,
còn 225 mẫu huấn luyện thì có. Đây là kiểu bài toán mà fine-tune có lợi thế thật.

**Latency:** (c) chậm hơn (b) (353.6 so với 245.4 ms) dù prompt ngắn hơn nhiều. Lý do là adapter
chưa merge thêm một nhánh matmul cho mỗi module ở 12 loại module × 24 lớp. NB6 merge xong
thì overhead này biến mất. Lưu ý thêm: mọi số latency đều đo với kernel tham chiếu PyTorch,
vì máy không cài `flash-linear-attention`/`causal_conv1d`. Chúng so sánh được giữa các run,
nhưng không phải tốc độ tuyệt đối tốt nhất.

---

## 4. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)

*(results/runs.csv, results/autopsy.json, results/loss_curves.json)*. Cả bốn run đều chạy
**58 step**, cùng dữ liệu, cùng mask, seed 42. Mỗi run chỉ đổi **một biến**:

| Run | Biến đổi | vị trí | r | trainable | LR | train loss TB (NB4) | loss step cuối | **target (NB5 §4)** | format | latency ms | train s | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `correct` | — (mốc) | text-linear (12 loại module) | 16 | 10,822,656 | 1e-4 | 0.3909 | 0.0068 | **0.985** | 1.000 | 353.6 | 114.3 | 1.97 |
| `attn_only` | **vị trí** (rank nâng để khớp ngân sách) | q,v | 271 | 10,822,656 | 1e-4 | 0.4348 | 0.0213 | **0.910** | 1.000 | 237.4 | 78.4 | 1.98 |
| `wrong_lr` | **LR** (÷10) | text-linear | 16 | 10,822,656 | 1e-5 | 1.5443 | 0.7789 | **0.330** | 0.995 | 330.8 | 109.0 | 1.98 |
| `qlora` | **độ chính xác base** (4-bit NF4) | text-linear | 16 | 10,822,656 | 1e-4 | 0.4242 | 0.0196 | **0.945** | 1.000 | 460.2 | 124.4 | 1.19 |

> "train loss TB" là `training_loss` của Trainer, tức **trung bình** loss trên cả 58 step (cột
> `final_loss` trong `runs.csv`), không phải loss của step cuối. Cột "loss step cuối" lấy từ log.

**Xếp hạng theo target:** `correct` 0.985 > `qlora` 0.945 > `attn_only` 0.910 > `wrong_lr` 0.330.
**Xếp hạng theo train loss TB:** `correct` 0.391 < `qlora` 0.424 < `attn_only` 0.435 < `wrong_lr` 1.544.
**Ở lần chạy này hai thứ tự trùng nhau.** Tôi nói thẳng điều đó thay vì bịa ra một mâu thuẫn.
Tuy vậy, loss vẫn là chỉ số thay thế tồi, vì hai lý do có số đo đi kèm:
(1) độ lớn không chuyển đổi được: chênh 0.044 loss giữa `correct` và `attn_only` tương ứng
7.5 điểm target, còn chênh 0.033 giữa `correct` và `qlora` chỉ là 4 điểm;
(2) loss **mù hoàn toàn** về hồi quy. Ở quét rank (mục 4.4), r=64 có loss thấp nhất (0.0004 ở
step cuối) và target 1.000, nhưng regression **0.000**, tức là tệ nhất. Nếu chọn run theo loss,
ta sẽ chọn đúng run quên nhiều nhất.

**4.1 — `attn_only` (cùng 10,822,656 tham số) thắng, thua hay hoà?**
**Thua: 0.910 so với 0.985 (−7.5 điểm)**, dù có đúng cùng ngân sách tham số (rank 271 so với 16).
Như vậy biến duy nhất còn lại là *vị trí gắn adapter*, và vị trí đó tạo ra khác biệt.
Model này có 18 lớp linear-attention (GatedDeltaNet: `in_proj_qkv`, `in_proj_z`, `in_proj_a/b`,
`out_proj`) và chỉ 6 lớp full-attention có `q_proj/v_proj`. `attn_only` vì thế chỉ chạm vào
**6/24 lớp**, và không chạm MLP nào. Dồn 10.8M tham số vào ít chỗ hơn với rank rất cao không
bù được việc bỏ qua các MLP, vốn là nơi lưu ánh xạ "cụm từ → nhãn". Bằng chứng thứ hai
(mục 4.4): text-linear **r=8** (5.4M tham số, *một nửa* ngân sách) đạt 0.900, gần bằng
`attn_only` r=271 với gấp đôi ngân sách. Vậy đòn bẩy là **vị trí**, còn **rank** chỉ là thứ yếu.
Thứ tự theo loss cũng giống (0.435 > 0.391), nên ở đây loss không đánh lừa. Nó chỉ không cho
biết khoảng cách thật là bao nhiêu.

**4.2 — `wrong_lr` chỉ khác đúng một con số.** Đường loss của `correct` rơi từ 2.807 xuống
0.315 sau 15 step, rồi 0.0068 ở step 55. `wrong_lr` đi từ 3.026 xuống 2.338 sau 15 step, rồi
0.779 ở step 55. Đường này **không phẳng**: nó giảm đều nhưng chậm, nên trông như "đang học,
chỉ cần thêm epoch". Nếu chỉ nhìn đường cong đó mà không biết LR, người ta dễ kết luận sai một
trong hai điều: "LoRA học kém hơn full fine-tune" (đúng cái danh tiếng mà deck §11.3 bác bỏ),
hoặc "model 0.8B không đủ sức cho tác vụ này". Cả hai đều sai. Cùng kiến trúc, cùng dữ liệu,
chỉ cần ×10 LR là đạt 0.985. Trên tác vụ, `wrong_lr` đạt target 0.330, **thấp hơn cả prompt (b)
0.500**, dù format 0.995. Nghĩa là nó học được *hình dạng* JSON nhưng chưa học được *nhãn*.
Một run như thế sẽ khiến ta kết luận "fine-tune vô ích" một cách oan uổng.

**4.3 — `qlora` tiết kiệm bao nhiêu, trả giá bằng gì?** Peak VRAM giảm từ 1.97 xuống
**1.19 GB (−0.78 GB, −40%)**. Cái giá: target −4 điểm (0.985 → 0.945), train chậm hơn 9%
(114.3 → 124.4 s), latency inference tăng 30% (353.6 → 460.2 ms, do dequantize mỗi lần
forward). Số đo **ủng hộ** khuyến nghị "không dùng QLoRA cho dòng model này" *khi bf16 LoRA vừa
VRAM*: ở 0.8B, bf16 chỉ cần 1.97 GB, nên tiết kiệm 0.78 GB không đáng mất 4 điểm cùng 30%
latency. Lưu ý: một seed duy nhất, và độ nhiễu giữa các lần chạy quan sát được là cỡ 0.02
(mục 5.3), nên −4 điểm là tín hiệu yếu-vừa chứ không phải kết luận chắc chắn. QLoRA chỉ đáng
dùng khi bf16 không vừa card.

### 4.4 Thưởng B4: quét rank có kiểm soát (vị trí cố định = text-linear)

*(results/rank_sweep.json, results/extra_runs.csv)*: 58 step, alpha = 2r, LR 1e-4.

| r | trainable | train loss TB | loss step cuối | target | regression | VRAM GB |
|---|---|---|---|---|---|---|
| 8 | 5,411,328 | 0.5228 | 0.0368 | 0.900 | 0.200 | 1.88 |
| 16 (`correct`) | 10,822,656 | 0.3909 | 0.0068 | 0.985 | 0.100 | 1.97 |
| 64 | 43,290,624 | 0.2622 | 0.0004 | 1.000 | 0.000 | 2.53 |

**Khi nào rank là đòn bẩy?** Từ r=8 lên 16, target tăng **+8.5 điểm**: ở r=8 adapter còn thiếu
dung lượng (hoặc thiếu tốc độ học trong 58 step) cho ánh xạ urgency. Từ 16 lên 64, target chỉ
tăng **+1.5 điểm** (gần trần) với 4× tham số, trong khi regression rơi từ 0.10 xuống **0.00**.
Kết luận: rank là đòn bẩy **chỉ khi adapter đang thiếu dung lượng** cho tác vụ, mà ở đây ngưỡng
đó nằm khoảng r=16. Quá ngưỡng ấy, rank lớn chủ yếu mua thêm khả năng *ghi đè* hành vi cũ,
tức là quên nhiều hơn chứ không làm tác vụ tốt hơn. So với 4.1: đổi vị trí ở cùng ngân sách
mất 7.5 điểm, còn giảm *một nửa* ngân sách ở vị trí đúng chỉ mất 8.5 điểm. Vị trí mới là đòn bẩy
chính.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.485` · `regression Δ = −0.544` (ngưỡng −0.020) · `valid_trace_rate = 0.00`

### 5.1 Diễn giải

Fine-tune làm **đúng việc nó được dạy, và chỉ việc đó**. Trên tác vụ, nó vượt prompt tốt nhất
48/50 ca, hoà 2, không thua ca nào (`results/qualitative_compare.json`). Nhưng nó trượt cổng vì
năng lực chung tụt 0.544, gấp 27 lần ngưỡng cho phép. Đây không phải thua vì cấu hình LoRA sai:
mask đúng (NB1), LR đúng thang (NB4 `wrong_lr` cho thấy điều ngược lại), vị trí đúng. Đây là
lỗi **phân phối dữ liệu huấn luyện**.

**Chẩn đoán bằng chứng, không đoán** (`results/regression_probe.json`, script
`scripts/extra_experiments.py probe`). Tôi lưu từng output trên 15 câu regression:

| | regression | % câu trả lời là JSON triage |
|---|---|---|
| base (không system prompt) | 0.644 | **0%** |
| `correct` (không system prompt) | 0.100 | **100%** |

Ví dụ: "Thủ đô của Việt Nam là thành phố nào?" → `{"intent": "hoi_thong_tin", "urgency": "thap",
"product": "hoi_thong_tin"}`. "1 km bằng bao nhiêu mét?" → `{"intent": "conversion",
"urgency": "low", ...}`. Model không "quên" Hà Nội. Nó đã học quy tắc **"mọi lượt user → JSON
triage"**, vì 225/225 mẫu huấn luyện đều có đúng hình dạng đó. Hơn nữa, system prompt lúc train
chỉ là "Phân loại ticket sau.", một tín hiệu quá yếu để model học được điều kiện "chỉ phân loại
khi được yêu cầu". Ngay cả 0.100 còn lại cũng là trúng ngẫu nhiên: "sức khỏe" lọt vào trường
`product`, "reading" lọt vào trường `intent`. Đây là giới hạn của thang keyword-recall: nó chấm
cả một JSON vô nghĩa nếu tình cờ chứa từ khoá.

**Điều này nói gì về bài toán:** với corpus đơn tác vụ và model nhỏ, LoRA ở cấu hình "không hối
tiếc" vẫn **quên thảm hoạ ở tầng hành vi**. Lợi thế "LoRA quên ít" của deck chỉ đúng khi dữ liệu
không dạy model một quy tắc định dạng phủ quyết mọi thứ. Thứ cần sửa là dữ liệu, không phải
rank hay LR.

### 5.2 Kiểm chứng nhân quả: replay (thí nghiệm bổ sung, KHÔNG thay phán quyết)

Nếu chẩn đoán đúng, trộn một ít dữ liệu "trả lời bình thường" sẽ phục hồi regression mà không
mất target. Tôi giữ nguyên cấu hình `correct` và 58 step, chỉ thêm replay **tự chưng cất**:
24 câu hỏi phổ thông viết tay, đã kiểm tra không trùng câu nào và không trùng từ khoá chấm nào
với `eval_regression.jsonl`; câu trả lời là output greedy của **chính base model**.
*(results/replay.json, results/replay_corpus.json)*

| Run | replay | target | regression | % JSON trên regression | so với (b): target Δ / regression Δ |
|---|---|---|---|---|---|
| `correct` | 0 | 0.985 | 0.100 | 100% | +0.485 / −0.544 → FAIL |
| `replay_12rows` | 12 dòng (5%) | **0.980** | **0.656** | **0%** | +0.480 / +0.011 → *sẽ* PASS |
| `replay_23rows` | 23 dòng (~10%) | 0.930 | 0.656 | 6.7% | +0.430 / +0.011 → *sẽ* PASS |

Chỉ 12 dòng (5%) đã đưa tỉ lệ JSON trên câu hỏi phổ thông từ 100% về 0%, regression từ 0.100
lên 0.656, mà target chỉ giảm 0.005. Điều này xác nhận cơ chế ở 5.1: kiến thức vẫn còn, chỉ cần
dạy model *khi nào* không trả JSON. Vì sao không đổi phán quyết: thí nghiệm này được thiết kế
**sau khi** đã thấy kết quả FAILED, trên cùng tập eval. Muốn deploy bản replay thì phải đánh giá
lại trên một tập chưa từng nhìn. Replay tự chưng cất cũng tái tạo cả **lỗi** của base: trong
`replay_corpus.json`, base trả lời "Sao Mộc" là hành tinh gần Mặt Trời nhất. Nó giữ *hành vi*
chứ không sửa *sự thật*.

### 5.3 Độ nhiễu: con số nào đáng tin

Cùng cấu hình `replay_12rows` chạy hai lần (lần đầu bị bỏ vì một lỗi đặt tên, xem
`results/replay_first_attempt.json`) cho regression **0.633** và **0.656**, target 0.98 cả hai.
Lý do là kernel GPU không tất định. Độ nhiễu ~0.02 trên regression **bằng đúng ngưỡng của cổng**
(0.02), và 15 câu hỏi nghĩa là mỗi câu nặng ~0.067. Hệ quả: chênh lệch target ≤ 0.02–0.04 giữa
các run (ví dụ `qlora` so với `correct`, hay 12 so với 23 dòng replay) **không nên** diễn giải
như kết luận chắc chắn. Chênh lệch 0.075 (`attn_only`), 0.485 (fine-tune so với (b)) và 0.544
(hồi quy) thì lớn hơn nhiễu rất nhiều.

---

## 6. Định tính: có cả ca THUA

Trên tập target, so (b) với (c) theo từng mẫu: **FT thắng 48, hoà 2, thua 0**
(`results/qualitative_compare.json`). Vì vậy các ca FT **thua** thật sự nằm ở tập regression
(`results/regression_probe.json`). Tôi cũng đưa vào ca FT sai trên target, kể cả khi vẫn hơn (b).

| # | Đầu vào (rút gọn) | Đúng | (b) base + prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | target #5: "…nồi chiên không dầu… **Thiếu phụ kiện. Khi nào tiện.** Cho tôi hỏi." | san_pham_loi / thap / trung_tinh | hoan_tien / cao / tich_cuc (0.25) | đúng cả 4 (1.00) | ✅ **FT thắng**: (b) không biết "thiếu phụ kiện" là lỗi sản phẩm và "khi nào tiện" là urgency thấp |
| 2 | target #12: "…áo khoác gió… **Bị lỗi. Khi nào tiện.** Cảm ơn shop nhiều." | san_pham_loi / thap / tich_cuc | van_chuyen / cao (0.50) | đúng cả 4 (1.00) | ✅ **FT thắng**: (b) đoán urgency `cao` cho 48/50 ticket (nhãn thật chỉ 19/50 là `cao`) |
| 3 | regression #0: "Thủ đô của Việt Nam là thành phố nào?" | chứa "Hà Nội" | base không prompt: "…**Hàn Quốc**…" (0.0) | `{"intent": "hoi_thong_tin", "urgency": "thap", "product": "hoi_thong_tin"}` (0.0) | ❌ **FT thua về hành vi**: cả hai đều 0 điểm, nhưng base ít nhất còn trả lời câu hỏi (dù bịa), còn FT không nhận ra đây không phải ticket |
| 4 | regression #2: "1 km bằng bao nhiêu mét?" | chứa "1000" | base: trả lời đúng (1.0) | `{"intent": "conversion", "urgency": "low", ...}` (0.0) | ❌ **FT thua**: kiến thức đơn giản bị định dạng JSON nuốt mất; còn bịa nhãn ngoài schema (`conversion`, `low`) |
| 5 | regression #8: "Ai là tác giả của Truyện Kiều?" | chứa "Nguyễn Du" | base: đúng (1.0) | `{"intent": "hoi_thong_tin", ...}` (0.0) | ❌ **FT thua** |
| 6 | target #18: "…máy xay sinh tố… Khi nào có tiền về. **Mong shop phản hồi.** Rất thất vọng." | hoan_tien / **trung_binh** / tieu_cuc | urgency cao, sentiment tich_cuc (0.50) | urgency **thap** (0.75) | ⚠ FT vẫn hơn (b) nhưng **sai so với nhãn** |

**Mẫu chung ở các ca FT thua/sai:**
- *Trên regression:* 100% là cùng một lỗi. Model áp schema triage lên mọi đầu vào, thậm chí bịa
  giá trị ngoài schema (`conversion`, `low`, `san_pham_tieu`). Nó học **hình dạng** đầu ra,
  không học **điều kiện** áp dụng.
- *Trên target:* cả 3 lỗi của FT đều ở trường `urgency`, cả 3 ticket đều chứa "**Mong shop phản
  hồi**" (marker `trung_binh`), và FT đều đoán `thap`. Trong 6 ticket eval có cụm này, FT sai 3
  (các mẫu #9, #18, #23). Ở #18 có thêm cụm "**Khi nào** có tiền về" trùng tiền tố với marker
  `thap` "khi nào tiện". Giả thuyết: model 0.8B học marker urgency theo kiểu bề mặt, nên khi hai
  tín hiệu cùng xuất hiện thì tín hiệu yếu hơn thua.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** **Không nên deploy bản `correct`.** Lý do không nằm ở chất lượng tác vụ (0.985 so
với 0.500 là mức thắng lớn và ổn định hơn nhiễu nhiều) mà ở chỗ nó **phá hành vi của model ở mọi
đầu vào khác**. Nếu đặt sau một router chỉ gửi ticket vào, thiệt hại có thể chấp nhận được.
Nhưng cổng hồi quy tồn tại chính vì ta không kiểm soát được mọi đầu vào thực tế: khách hỏi
"shop mở cửa mấy giờ?" sẽ nhận về một JSON. Chuỗi nhân quả đo được như sau. Corpus 100% một
định dạng, cộng với system prompt yếu, khiến LoRA học quy tắc "mọi input → JSON". Quy tắc này
phủ quyết hành vi trả lời tự do của base, nên regression tụt từ 0.644 xuống 0.100. Hai bằng
chứng cho thấy đây là quan hệ nhân quả chứ không chỉ tương quan: (1) tỉ lệ JSON trên câu hỏi
phổ thông đi từ 0% lên 100%; (2) can thiệp đúng vào nguyên nhân (12 dòng replay) đảo ngược
cả hai, về 0% JSON và regression 0.656, trong khi mọi biến khác giữ nguyên. Ứng viên deploy hợp
lý là bản replay, **sau khi** đánh giá lại trên dữ liệu chưa từng nhìn.

**Đâu là đòn bẩy thật sự?** Xếp theo độ lớn hiệu ứng đo được:
(1) **learning rate**: sai thang thì mất 65.5 điểm target (0.985 → 0.330), thua cả prompt;
(2) **thành phần dữ liệu** (replay): quyết định PASS hay FAIL, 0.556 điểm regression;
(3) **vị trí adapter**: 7.5 điểm ở cùng ngân sách tham số;
(4) **rank**: đáng kể dưới r=16, gần như vô ích (và hại cho hồi quy) phía trên;
(5) **4-bit**: −4 điểm, ở mức sát nhiễu.
**Mask** là điều kiện tiên quyết chứ không phải một "đòn bẩy" để vặn: NB1 chứng minh mask đúng
trước khi train, nên không phải nguồn lỗi. Bài học lớn nhất là mọi cấu hình LoRA của deck đều có
thể đúng hết mà fine-tune vẫn trượt, vì thứ quyết định cuối cùng là *dữ liệu dạy model điều gì
ngoài tác vụ*.

**Ba điều tôi học được**
1. **"Regression" có thể là sập định dạng chứ không phải quên kiến thức, và hai thứ này cần hai
   cách sửa khác nhau.** Con số tổng 0.100 trông như mất trí nhớ, nhưng đọc output thì thấy 15/15
   là JSON. Nếu chỉ đọc bảng tổng, tôi đã tìm cách giảm rank hay LR (sai hướng). Đọc từng output
   chỉ mất vài giây và đổi hẳn chẩn đoán.
2. **Một lượng replay rất nhỏ là đủ khi lỗi nằm ở hành vi**: 12 dòng / 237 là đủ để model học
   lại điều kiện "chỉ phân loại khi được yêu cầu", target giảm chỉ 0.005. Tôi từng nghĩ chống quên
   phải trộn lượng lớn dữ liệu chung.
3. **Ngưỡng của cổng phải đặt cạnh độ nhiễu đo được.** Chạy lại đúng cùng cấu hình cho
   regression lệch 0.022 (0.633 so với 0.656), bằng ngưỡng 0.02 của cổng. Với 15 câu regression,
   một câu đổi kết quả là xê dịch 0.067. Một cổng sát như vậy trên tập nhỏ có thể PASS/FAIL theo
   may rủi. Cần tập regression lớn hơn hoặc chạy nhiều seed trước khi tin một kết quả sát ngưỡng.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
- Chạy `correct` và `replay_12rows` với **3 seed** để có khoảng tin cậy, xem chênh lệch của
  `qlora` (−4 điểm) có vượt nhiễu hay không.
- Mở rộng tập regression lên ≥100 câu và thêm một thang chấm "có từ chối áp schema không" thay
  cho keyword recall, vốn đã chấm điểm cho JSON vô nghĩa.
- Thay replay tự chưng cất bằng replay có đáp án đúng, để xem có *tăng* được regression vượt base
  không (hiện tại base bịa "Sao Mộc", "Hàn Quốc").
- Đổi system prompt huấn luyện thành câu mang tính điều kiện ("Nếu là ticket CSKH thì phân loại…"),
  so với replay, xem riêng việc điều kiện hoá prompt có đủ chặn sập định dạng không.

---

## Phụ lục: thưởng đã làm

- [x] **B1** NB6 merge + hot-swap: `results/merge_check.json`, target trước merge **0.985**, sau
  merge **0.985** (Δ 0.000, ngưỡng 0.01). Hot-swap **3 adapter** (`correct`, `attn_only`, `qlora`)
  trên **cùng một** base đang nạp; cùng một ticket cho ba output khác nhau (log NB6).
- [ ] B2 dataset miền riêng: không làm (dùng corpus mặc định).
- [ ] B3 reasoning-trace collapse: không làm. Corpus không có trace, nên `masked-think`/
  `response-only` cho mask trùng `assistant-only` (NB3 có cảnh báo). `valid_trace_rate = 0.00`
  là đúng như dự kiến vì target là JSON trần.
- [x] **B4** quét rank có kiểm soát: mục 4.4, `results/rank_sweep.json`.
- [ ] B5 HuggingFace Hub: chưa push.

**Thí nghiệm ngoài rubric:** probe regression (5.1), replay (5.2), độ nhiễu (5.3). Toàn bộ chạy
bằng `scripts/extra_experiments.py {probe,qual,replay,sweep}`, ghi vào file riêng
(`regression_probe.json`, `qualitative_compare.json`, `replay.json`, `rank_sweep.json`,
`extra_runs.csv`). Script **không** ghi vào `runs.csv`, `baselines_frozen.json` hay
`verdict.json`, và không sửa tập eval hay `OPTIMIZED_PROMPT`.

**Tái lập:**
```bash
conda activate lab-vin-env       # torch 2.14 + cu130 có sẵn; pip install -r requirements.txt (bỏ dòng torch)
# .env: COMPUTE_TIER=LAPTOP, BASE_MODEL=Qwen/Qwen3.5-0.8B, MASK_MODE=assistant-only, EPOCHS=2
python notebooks/01_data_and_mask.py
python scripts/colab_run.py nb2 nb3 nb4 nb5 nb6        # ~12.4 phút trên RTX 4050 Laptop
python scripts/extra_experiments.py probe
python scripts/extra_experiments.py qual
python scripts/extra_experiments.py replay --pct 5 10
python scripts/extra_experiments.py sweep --ranks 8 16 64
python scripts/verify.py
```
