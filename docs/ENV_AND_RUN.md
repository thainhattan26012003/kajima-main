# Tạo môi trường ảo và chạy project

Yêu cầu: **Python 3.10 trở lên** (`python3 --version`).

---

## Cách 1: Dùng Poetry (khuyến nghị)

Poetry tự tạo virtual env và quản lý dependency.

### 1. Cài Poetry

Trên Linux (Debian/Ubuntu) Python do hệ thống quản lý thường không cho `pip install` trực tiếp. Dùng **pipx** (cài Poetry tách biệt, không ảnh hưởng system):

```bash
# Cài pipx nếu chưa có
sudo apt update
sudo apt install pipx
pipx ensurepath

# Cài Poetry (sau đó mở lại terminal hoặc source ~/.bashrc)
pipx install poetry
```

Kiểm tra: `poetry --version`

### 2. Vào thư mục project và cài dependency

```bash
cd /path/to/kajima-construction-main

poetry self update
poetry self add poetry-plugin-export
poetry install
```

- Lần đầu chạy `poetry install`, Poetry sẽ **tạo virtual env** (thường nằm trong `~/.cache/pypoetry/virtualenvs/` hoặc trong project nếu cấu hình `in-project = true`).
- Các package trong `pyproject.toml` và `poetry.lock` (kể cả nhóm `batch_processor`: torch, transformers, peft…) sẽ được cài vào env đó.

### 3. File .env

```bash
cp .env.example .env
```

Mở `.env`, sửa ít nhất:

- **`MODEL_CONFIG_PATH`**: đường dẫn file config (train/eval/app), ví dụ:
  ```bash
  MODEL_CONFIG_PATH="config/classification/tuning/dinov2_base_peft_vit_20250324.json"
  ```

Load biến môi trường (mỗi lần mở terminal mới):

```bash
source .env
```

### 4. Chạy

Mọi lệnh chạy qua `poetry run` để dùng đúng env:

```bash
# Web app Gradio (phân loại ảnh)
poetry run python -m src.app

# Train
poetry run python -m src.train -c config/classification/tuning/dinov2_base_peft_vit_20250324.json -o checkpoints/my-finetuned --device cuda:0

# Eval
poetry run python -m src.eval -c config/classification/tuning/dinov2_base_peft_vit_20250324.json -d cuda:0
```

**Kích hoạt shell trong env (tùy chọn):** sau khi `poetry install`, có thể gõ `poetry shell` rồi chạy `python -m src.app` (không cần `poetry run`).

---

## Cách 2: Dùng venv + pip (không dùng Poetry)

### 1. Tạo virtual env

```bash
cd /path/to/kajima-construction-main

python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# Windows: .venv\Scripts\activate
```

### 2. Cài dependency

Project có `pyproject.toml`; cần cài thủ công các package chính (kể cả torch, transformers, peft cho train):

```bash
pip install --upgrade pip
pip install numpy matplotlib pandas gradio opencv-python wandb fastapi mangum pydantic aws-lambda-powertools boto3 "python-jose>=3.4.0,<4.0.0"
pip install scikit-learn "transformers>=4.47.0,<4.48.0" "peft>=0.14.0"
pip install torch torchvision   # cài bản phù hợp với máy (CPU/CUDA)
```

Nếu có `requirements.txt` (export từ Poetry): `pip install -r requirements.txt`.

### 3. .env và MODEL_CONFIG_PATH

Giống Cách 1: copy `.env.example` → `.env`, sửa `MODEL_CONFIG_PATH`, rồi `source .env`.

### 4. Chạy

Đảm bảo đã `source .venv/bin/activate` và `source .env`, sau đó:

```bash
python -m src.app
python -m src.train -c config/classification/tuning/dinov2_base_peft_vit_20250324.json -o checkpoints/my-finetuned --device cuda:0
python -m src.eval -c config/classification/tuning/dinov2_base_peft_vit_20250324.json -d cuda:0
```

---

## Tóm tắt nhanh (Poetry)

```bash
cd /path/to/kajima-construction-main
pip install poetry
poetry install
cp .env.example .env
# Sửa .env: MODEL_CONFIG_PATH="config/classification/tuning/dinov2_base_peft_vit_20250324.json"
source .env
poetry run python -m src.app
```

Mở trình duyệt: **http://127.0.0.1:7860/** (Gradio).
