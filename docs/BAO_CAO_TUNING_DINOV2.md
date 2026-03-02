# Báo cáo Tuning DINOv2 — Phân loại ảnh Kajima (4 lớp)

---

## 1. Tóm tắt tổng quan (Executive Summary)

Báo cáo này tổng hợp các kết quả thực nghiệm trong việc ứng dụng mô hình **DINOv2** (Self-supervised Vision Transformer) cho bài toán **phân loại ảnh 4 lớp** trên bộ dữ liệu Kajima (các lớp 7-1, 7-2, 7-3, 7-4). Mục tiêu của giai đoạn này là tối ưu hóa các siêu tham số (hyperparameters) để vượt qua ngưỡng độ chính xác (Accuracy) mục tiêu.

Trải qua nhiều vòng huấn luyện với các cấu hình khác nhau (Option A/B/C, Focal Loss, tăng độ phân giải 518, sweep Bayes, v.v.), độ chính xác của mô hình trên tập đánh giá hiện đang đạt đỉnh và bão hòa ở mức **khoảng 78–84%** (baseline eval ~78,8%, trong đó class 3 yếu ~66%). Việc thay đổi các tham số tiêu chuẩn không mang lại sự gia tăng đột phá, cho thấy nguyên nhân có thể nằm ở kiến trúc fine-tuning (LoRA/head), giới hạn của bộ dữ liệu, hoặc giới hạn đặc trưng của phiên bản DINOv2 đang sử dụng.

---

## 2. Thiết lập thực nghiệm (Experimental Setup)

| Hạng mục | Chi tiết |
|----------|----------|
| **Mô hình gốc** | DINOv2 **ViT-B/14** (base) — `facebook/dinov2-base`, đôi khi thử DINOv2 small 1-layer (`dinov2_small`) |
| **Phương pháp huấn luyện** | **LoRA (PEFT)** — fine-tune một phần các layer cuối của backbone (số layer và rank thay đổi theo từng thí nghiệm), kết hợp head phân loại (cosine hoặc linear). Không dùng Linear Probing thuần hay Full Fine-tuning toàn bộ trọng số. |
| **Bộ dữ liệu** | **Kajima dataset**: `datasets/kajima_dataset` (train), `datasets/kajima_test_dataset` (test/validation). **4 lớp** (7-1, 7-2, 7-3, 7-4). Số lượng ảnh train/test phụ thuộc cách chia từ script `split_train_test.py` / `split_train_test_1.py` (có thể điền số lượng cụ thể sau khi chạy split). |
| **Độ phân giải ảnh** | Thay đổi theo thí nghiệm: **224/256**, **448**, **518** (crop_size / resolution trong config). |

---

## 3. Các thực nghiệm đã tiến hành (Hyperparameter Tuning)

Để cố gắng bứt phá ngưỡng accuracy, các tham số sau đã được điều chỉnh và thử nghiệm (độc lập hoặc kết hợp) qua nhiều file config trong `config/classification/tuning/`:

### 3.1 Learning Rate (LR) & Optimizer

- **LR**: Đã thử nhiều giá trị từ **1e-5** (ví dụ 6e-5, 5e-5) đến **1e-3** (ví dụ 1e-3 với SGD). Cấu hình điển hình: **6e-4** (Option A/B), **1e-4** (Focal, Option C), **6.49e-4** (sweep LADE).
- **Optimizer**: **Adam**, **AdamW**, **SGD** (momentum 0.9). Một số run dùng AdamW kết hợp weight_decay cao hơn.
- **Scheduler**: Trong config không ghi rõ Cosine Annealing / Warmup; có thể đã dùng mặc định của trainer.

### 3.2 Batch Size & Gradient

- Đã thử **8, 16, 24, 32, 64** (batch_size / micro_batch_size) để kiểm tra ổn định và nhiễu gradient.

### 3.3 Weight Decay & Regularization

- **Weight decay**: Điều chỉnh từ **0** (một số config CE/SGD) đến **0.05** (unfreeze high rank), qua **0.0003, 0.001, 0.01, 0.03**.
- **LoRA**: **lora_rank** 16, 24, 32, 48, 64; **num_layer_finetuned** 6, 8, 9, 12 (và ý tưởng null = toàn bộ layer); **lora_initialization_strategy**: random, dora, rslora. Tăng rank/layer nhằm tăng capacity, giảm rank/layer nhằm chống overfit.

### 3.4 Cân bằng lớp & Sampler

- **Sampler**: `""` (mặc định), **class_aware_sampler**, **down_sampler**, **random_sampler** — ưu tiên `class_aware_sampler` khi class 3 (và có thể lớp khác) bị lệch.
- **Criterion (Loss)**:
  - **CE** (Cross-Entropy), **LA** (Logit Adjusted), **CBL** (Class Balanced), **LADE**, **focal** (Focal Loss).
  - Focal Loss kết hợp **class weight** (ví dụ `[1, 1, 1, 1.8]` cho class 3).
- **init_head**: **class_mean** (ưu tiên khi imbalanced) hoặc **no**.

### 3.5 Head phân loại & Độ phân giải

- **classifier_head**: **cosine** (phổ biến) và **linear** — so sánh capacity và độ ổn định.
- **Data / Độ phân giải**: Thử **224/256**, **448**, **518** (DINOv2 chuẩn chia hết cho 14) để kiểm tra ảnh thu nhỏ quá mức có làm mất đặc trưng vi mô hay không.

### 3.6 Các bộ config đại diện đã chạy

- **Option A** (cân bằng class): `class_aware_sampler`, LA, cosine head, 224/256 hoặc 518.
- **Option B** (capacity): Option A + lora_rank 48, num_layer_finetuned 12, num_epochs 28.
- **Option C** (ổn định / chống overfit): lr 1e-4, weight_decay 0.01, lora_rank 16, linear head, 448.
- **Focal Loss**: criterion focal, weight [1,1,1,1.8], resolution 518, lora_rank 64, 12 layer.
- **Sweep (Bayes)**: Quét batch_size, criterion_type (CBL, LA, focal, CE, LADE), lr (1e-6–1e-3), sampler, optimizer (lion, adam), weight_decay, lora init, lora_rank, num_layer_finetuned; metric tối ưu: test/mean_acc.

---

## 4. Phân tích nguyên nhân bão hòa (Root Cause Analysis)

Việc accuracy dừng ở mức ~78–84% dù đã rà soát kỹ các tham số huấn luyện cho thấy đã đạt tới giới hạn của phương pháp tiếp cận hiện tại. Các nguyên nhân khả dĩ:

1. **Nút thắt ở bộ phân loại và LoRA (Classifier / Adapter Bottleneck)**  
   Chỉ fine-tune một phần backbone qua LoRA (6–12 layer, rank 16–64) và một head (cosine/linear). Bộ adapter + head có thể không đủ capacity để học các ranh giới quyết định khó giữa 4 lớp, đặc biệt lớp 3 (7-3) thường yếu (~66%).

2. **Giới hạn dữ liệu (Data Limitations)**  
   Phần mẫu bị phân loại sai có thể chứa nhiễu nhãn (mác gán sai từ đầu) hoặc là các trường hợp cực khó (edge cases) mà bộ train không cung cấp đủ thông tin để mô hình phân biệt. Cân bằng lớp (sampler, LA, focal) cải thiện phần nào nhưng không đột phá.

3. **Kích thước ảnh đầu vào**  
   Đã thử 224, 448, 518. Nếu với 224/256 đặc trưng không gian vi mô bị mất, việc tăng lên 448/518 có thể cần kết hợp thêm (epoch, LR, augmentation) để hội tụ ổn định; các thí nghiệm 518/448 chưa đủ để kết luận vượt ngưỡng.

4. **Phiên bản DINOv2**  
   DINOv2-base (ViT-B/14) có thể đã đạt trần đặc trưng cho bài toán này với cách dùng LoRA hiện tại; có thể cân nhắc thử mô hình lớn hơn (ViT-L/14) hoặc thay đổi kiến trúc fine-tuning (full fine-tune một phần backbone, hoặc head sâu hơn).

---

## 5. Khuyến nghị tiếp theo (Next Steps)

- **Điền số liệu cụ thể**: Số ảnh train / validation (test) sau khi chạy `split_train_test*.py`; bảng accuracy từng run (theo config) để báo cáo chi tiết hơn.
- **Thử Full Fine-tuning** (unfreeze toàn bộ hoặc nhiều layer hơn) nếu tài nguyên cho phép.
- **Kiểm tra nhãn**: Rà soát nhiễu nhãn và các mẫu khó (confusion matrix, visualisation).
- **Augmentation**: Tăng cường Data Augmentation (Random Erasing, Color Jitter, Mixup/CutMix) nếu chưa áp dụng đầy đủ trong pipeline hiện tại.
- **Label smoothing**: Thử label smoothing (ví dụ 0.1) kết hợp với CE/LA/focal đã dùng.

---

*Tài liệu tham khảo trong repo: `docs/TUNING_OPTIONS.md`, `config/classification/tuning/*.json`, `scripts/split_train_test_1.py`.*
