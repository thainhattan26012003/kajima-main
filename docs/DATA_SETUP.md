# Hướng dẫn setup data để train

## 1. Cấu trúc thư mục bắt buộc

Dataset được đọc bởi `KajimaDataset` với quy ước tên thư mục như sau:

```
<train_dataset_dir>/   hoặc   <test_dataset_dir>/
├── <date_1>/              ← Cấp 1: tên bất kỳ (thường là ngày, VD: 20241212)
│   ├── <tên>-<class>-<hậu_tố>/   ← Cấp 2: BẮT BUỘC đúng format
│   │   ├── IMG_001.JPG
│   │   ├── IMG_002.png
│   │   └── ...
│   ├── 7-2-241212/
│   │   └── ...
│   └── ...
├── <date_2>/
│   └── ...
└── ...
```

### Quy tắc quan trọng

- **Cấp 1:** Mỗi thư mục con là một “đợt”/ngày (tên tự chọn, VD: `20241212`, `20250108`).
- **Cấp 2:** Mỗi thư mục con của `<date>` phải có **đúng 3 phần** khi tách bằng dấu `-`:
  - Format: `xxx-<số_class>-xxx`
  - **Số class** (phần giữa) dùng **1-based**: 1, 2, 3, 4 tương ứng nhãn 7-1, 7-2, 7-3, 7-4.
  - VD: `7-1-241212` → class 1 (index 0), `7-4-250108` → class 4 (index 3).
- **Ảnh:** Đặt trực tiếp trong thư mục cấp 2; hỗ trợ định dạng PIL (JPG, PNG, …).

Code đọc nhãn (trong `dataset.py`):

```python
_, label, _ = img_dn.split("-")   # img_dn = "7-1-241212" → label = "1"
label = int(label) - 1            # 0, 1, 2, 3
```

Nếu tên thư mục không đủ 3 phần khi `split("-")`, thư mục đó bị bỏ qua (in "Failed folder").

---

## 2. Ví dụ cụ thể

**Train:** `datasets/kajima_dataset`  
**Test:** `datasets/kajima_test_dataset`

```
datasets/
├── kajima_dataset/
│   ├── 20241212/
│   │   ├── 7-1-241212/
│   │   │   ├── IMG_6154.JPG
│   │   │   ├── IMG_6155.JPG
│   │   │   └── ...
│   │   ├── 7-2-241212/
│   │   ├── 7-3-241212/
│   │   └── 7-4-241212/
│   └── 20250108/
│       ├── 7-1-250108/
│       ├── 7-2-250108/
│       └── 7-4-250108/
└── kajima_test_dataset/
    └── 20250108/
        ├── 7-1-250108/
        ├── 7-2-250108/
        ├── 7-3-250108/
        └── 7-4-250108/
```

---

## 3. Config JSON

Trong file config (VD: `config/classification/tuning/dinov2_base_peft_vit.json`), cần khớp đường dẫn và số class:

```json
{
  "data": {
    "train_dataset_dir": "datasets/kajima_dataset",
    "test_dataset_dir": "datasets/kajima_test_dataset",
    "num_classes": 4
  }
}
```

- Đường dẫn có thể **tương đối** (từ thư mục gốc project) hoặc **tuyệt đối**.
- `num_classes` phải bằng số class thực tế (4 cho 7-1 ~ 7-4).

---

## 4. Các bước nhanh để bắt đầu train

1. **Tạo thư mục và sắp ảnh** theo cấu trúc trên (ít nhất 1 thư mục date, mỗi class ít nhất 1 thư mục `xxx-<class>-xxx`, trong đó có file ảnh).
2. **Copy và sửa config:**  
   `cp config/classification/tuning/dinov2_base_peft_vit.json config/classification/tuning/my_train.json`  
   Sửa `train_dataset_dir`, `test_dataset_dir` cho đúng đường dẫn của bạn.
3. **Kiểm tra dataset (tùy chọn):**  
   `poetry run python scripts/check_dataset.py datasets/kajima_dataset`
4. **Chạy train:**  
   `poetry run python -m src.train -c config/classification/tuning/my_train.json -o checkpoints/my-finetuned --device cuda:0`

---

## 5. Lưu ý

- Base model phải có sẵn tại `model.model_path` (VD: `checkpoints/dinov2-base`). Nếu chưa có, chạy script trong `scripts/install_huggingface_model.sh` hoặc tải DINOv2 base theo tài liệu project.
- Train và test **nên tách** (khác thư mục hoặc khác date) để đánh giá đúng.
- Nếu class imbalanced, trong config có thể dùng `sampler: "class_aware_sampler"` hoặc `"down_sampler"` và các loss CBL/LA/LADE như đã mô tả trong tài liệu train tuning.
