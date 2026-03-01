# PyTorch với RTX 5080 / 5090 (Blackwell, sm_120)

RTX 50 series (Blackwell, kiến trúc **sm_120**) chưa được bản PyTorch **stable** hỗ trợ. Bạn sẽ gặp:

- Warning: `NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible with the current PyTorch installation`
- Lỗi: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

## Cách 1: Chạy trên CPU (chạy ngay, không cần đổi env)

Dùng GPU khác hoặc tạm chạy trên CPU:

```bash
conda activate kajima-construction-gpu
python -m src.train -c config/classification/tuning/test.json -o outputs/run1 --device cpu
```

Chậm hơn GPU nhưng không cần cài thêm gì.

---

## Cách 2: Dùng PyTorch nightly (hỗ trợ RTX 5080/5090)

Cài PyTorch **nightly** với CUDA 12.8 hoặc 12.9 (có sm_120) **trong cùng env**:

```bash
conda activate kajima-construction-gpu

# Gỡ PyTorch từ conda (tránh xung đột)
pip uninstall -y torch torchvision torchaudio

# Cài PyTorch nightly + CUDA 12.8 (hoặc cu129 cho CUDA 12.9)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Sau đó chạy train bình thường (không cần `--device`):

```bash
python -m src.train -c config/classification/tuning/test.json -o outputs/run1
```

**Lưu ý:**

- Nightly có thể đổi hàng ngày; nếu lỗi lạ, thử cài lại hoặc dùng CPU.
- Driver NVIDIA trên máy nên tương thích CUDA 12.8+ (kiểm tra: `nvidia-smi`).

---

## Kiểm tra PyTorch nhận GPU

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Nếu thấy `CUDA: True` và tên RTX 5080/5090 mà train vẫn báo lỗi kernel, dùng Cách 1 (CPU) hoặc cài lại nightly (Cách 2).
