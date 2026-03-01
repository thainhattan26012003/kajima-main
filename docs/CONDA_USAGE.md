# Chạy train, eval và scripts với Conda

## 1. Tạo và kích hoạt environment

```powershell
# Tạo environment từ file
conda env create -f environment.yml

# Kích hoạt (Windows PowerShell)
conda activate kajima-construction
```

## 2. Cài package ở chế độ editable (bắt buộc)

Để lệnh `python -m src.train` nhận module `src`, cần cài project tại thư mục gốc:

```powershell
cd P:\TORATECH\kajima-main
pip install -e .
```

## 3. Train (huấn luyện)

```powershell
python -m src.train -c config/classification/tuning/test.json -o outputs/run1
```

**Tham số:**

| Tham số | Mô tả |
|--------|--------|
| `-c`, `--configuration` | Đường dẫn file config JSON (bắt buộc) |
| `-o`, `--output-dir` | Thư mục lưu model/checkpoint (tùy chọn) |
| `--device` | `cuda:0`, `mps`, hoặc `cpu` (mặc định: tự chọn) |

**Ví dụ:**

```powershell
# CPU
python -m src.train -c config/classification/tuning/test.json -o checkpoints/my_run --device cpu

# GPU
python -m src.train -c config/classification/tuning/dinov2_base_peft_vit.json -o checkpoints/peft_run --device cuda:0
```

## 4. Eval (đánh giá)

```powershell
python -m src.eval -c config/classification/tuning/test.json
```

**Tham số:**

| Tham số | Mô tả |
|--------|--------|
| `-c`, `--configuration` | File config JSON (bắt buộc) |
| `-d`, `--device` | `cuda:0`, `mps`, hoặc `cpu` |

## 5. Scripts khác

Chạy từ thư mục gốc project (đã `conda activate` và `pip install -e .`):

```powershell
# Chia train/test
python scripts/split_train_test.py --source image_kajima --test-ratio 0.2
python scripts/split_train_test_1.py --source image_kajima --test-ratio 0.2

# Phân tích phân bố dữ liệu
python scripts/analyze_data_distribution.py -c config/classification/tuning/test.json
python scripts/analyze_data_distribution.py -t datasets/kajima_dataset -e datasets/kajima_test_dataset
```

## 6. Tóm tắt workflow

```powershell
conda activate kajima-construction
cd P:\TORATECH\kajima-main

# (Lần đầu) Cài package
pip install -e .

# Train
python -m src.train -c config/classification/tuning/test.json -o outputs/exp1

# Eval
python -m src.eval -c config/classification/tuning/test.json
```

## 7. Cập nhật environment

Sau khi sửa `environment.yml`:

```powershell
conda env update -f environment.yml --prune
```

Xóa environment và tạo lại:

```powershell
conda env remove -n kajima-construction
conda env create -f environment.yml
```
