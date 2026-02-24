# Gợi ý tuning parameters để tăng accuracy

Dựa trên config hiện tại và kết quả eval (accuracy ~78.8%, class 3 yếu ~66%). Có thể thử từng hướng rồi kết hợp.

---

## 1. Cân bằng class (ưu tiên khi có class yếu)

**Vấn đề:** Config đang dùng `"sampler": ""` → không cân bằng batch theo class; class 3 có thể ít mẫu.

| Parameter | Hiện tại | Gợi ý | Ghi chú |
|-----------|----------|--------|--------|
| **sampler** | `""` | `"class_aware_sampler"` hoặc `"down_sampler"` | Ưu tiên thử `class_aware_sampler` trước. |
| **criterion_type** | `LA` | `LA`, `CBL`, hoặc `LADE` | LA đã tốt; nếu vẫn lệch class thử `CBL` hoặc `LADE`. |

**Ví dụ block `training` để paste vào config:**
```json
"training": {
  "batch_size": 64,
  "criterion_type": "LA",
  "init_head": "class_mean",
  "lora_initialization_strategy": "random",
  "lora_rank": 32,
  "lr": 0.0006,
  "micro_batch_size": 32,
  "num_epochs": 20,
  "num_layer_finetuned": 8,
  "num_workers": 2,
  "optimizer": "adam",
  "print_freq": 1,
  "sampler": "class_aware_sampler",
  "seed": 32,
  "weight_decay": 0.0003
}
```

---

## 2. Tăng capacity / fine-tune sâu hơn

**Khi model có vẻ underfit (loss còn cao, accuracy chưa ổn).**

| Parameter | Hiện tại | Gợi ý | Ghi chú |
|-----------|----------|--------|--------|
| **lora_rank** | `32` | `48` hoặc `64` | Rank cao hơn → nhiều tham số adapter. |
| **num_layer_finetuned** | `8` | `12` hoặc `null` | `null` = fine-tune toàn bộ layers. |
| **num_epochs** | `20` | `25`–`30` | Train lâu hơn. |

---

## 3. Ổn định / giảm overfit

**Khi train loss giảm mạnh nhưng eval không cải thiện.**

| Parameter | Hiện tại | Gợi ý | Ghi chú |
|-----------|----------|--------|--------|
| **lr** | ~`0.0006` | `3e-4` hoặc `1e-4` | LR nhỏ hơn. |
| **weight_decay** | ~`0.0003` | `0.001`–`0.01` | Tăng regularization. |
| **lora_rank** | `32` | `16` | Ít tham số hơn. |
| **num_layer_finetuned** | `8` | `4`–`6` | Fine-tune ít layer hơn. |

---

## 4. Loss cho long-tail / imbalanced

| criterion_type | Mô tả |
|----------------|--------|
| **LA** (Logit Adjusted) | Điều chỉnh logit theo tần suất class. Đang dùng. |
| **CBL** (Class Balanced Loss) | Weight loss theo nghịch đảo số mẫu mỗi class. |
| **LADE** | Variant cho long-tail. |
| **focal** | Focal Loss — tập trung vào mẫu khó. |
| **CE** | Cross-entropy thuần. |

Nên giữ **LA** hoặc thử **CBL** / **LADE** nếu class 3 vẫn kém sau khi bật sampler.

---

## 5. Head và khởi tạo

| Parameter | Gợi ý |
|-----------|--------|
| **classifier_head** | `cosine` (đang dùng) thường tốt; có thể thử `linear` hoặc `layernorm`. |
| **init_head** | Giữ `class_mean` khi imbalanced. |

---

## 6. Ba bộ config mẫu để thử nhanh

### Option A — Cân bằng class (ưu tiên thử trước)
- `sampler`: `"class_aware_sampler"`
- `criterion_type`: `"LA"`
- Các tham số khác giữ như config hiện tại.

### Option B — Cân bằng + tăng capacity
- Như Option A, thêm:
  - `lora_rank`: `48`
  - `num_epochs`: `28`
  - `num_layer_finetuned`: `12` (hoặc `null` nếu GPU đủ).

### Option C — Ổn định (khi nghi ngờ overfit)
- `sampler`: `"class_aware_sampler"`
- `lr`: `0.0002`
- `weight_decay`: `0.001`
- `lora_rank`: `16`
- `num_epochs`: `25`

---

## Cách dùng

1. Copy config hiện tại thành bản mới (ví dụ `dinov2_tuning_option_a.json`).
2. Sửa block `"training"` theo một trong các option trên.
3. Train:
   ```bash
   poetry run python -m src.train -c config/classification/tuning/dinov2_tuning_option_a.json -o checkpoints/dinov2-option-a --device cuda:0
   ```
4. Cập nhật `peft_adapter_path` trỏ tới thư mục output, rồi chạy eval và so sánh.

Ưu tiên thử **Option A** trước (chỉ bật `class_aware_sampler`).
