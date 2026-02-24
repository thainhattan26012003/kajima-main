#!/usr/bin/env python3
"""
Kiểm tra cấu trúc thư mục dataset có đúng format KajimaDataset hay không.
Cách chạy:
  poetry run python scripts/check_dataset.py <data_dir>
  poetry run python scripts/check_dataset.py datasets/kajima_dataset
  poetry run python scripts/check_dataset.py datasets/kajima_dataset datasets/kajima_test_dataset
"""
import sys
from pathlib import Path
from collections import Counter, defaultdict


def check_data_dir(data_dir: str) -> dict:
    data_path = Path(data_dir)
    if not data_path.exists():
        return {"ok": False, "error": f"Thư mục không tồn tại: {data_dir}"}

    image_labels = []
    failed_folders = []

    for date_name in sorted(data_path.iterdir()):
        if not date_name.is_dir():
            continue
        for img_dn in sorted(date_name.iterdir()):
            if not img_dn.is_dir():
                continue
            try:
                parts = img_dn.name.split("-")
                if len(parts) != 3:
                    failed_folders.append((str(img_dn), f"Tên cần đúng 3 phần (xxx-class-xxx), có {len(parts)} phần"))
                    continue
                _, label_str, _ = parts
                label = int(label_str) - 1
                count = 0
                for f in img_dn.iterdir():
                    if f.is_file():
                        image_labels.append((f, label))
                        count += 1
                if count == 0:
                    failed_folders.append((str(img_dn), "Không có file ảnh nào"))
            except Exception as ex:
                failed_folders.append((str(img_dn), str(ex)))

    if not image_labels:
        return {
            "ok": False,
            "error": "Không tìm thấy ảnh nào hợp lệ.",
            "failed_folders": failed_folders,
        }

    labels = [l for _, l in image_labels]
    num_classes = len(set(labels))
    counter = Counter(labels)
    sorted_idx = sorted(counter.keys())
    cls_num_list = [counter[i] for i in sorted_idx]

    return {
        "ok": True,
        "total_images": len(image_labels),
        "num_classes": num_classes,
        "class_counts": dict(zip([f"class_{i+1}" for i in sorted_idx], cls_num_list)),
        "failed_folders": failed_folders,
        "dates": sorted([d.name for d in data_path.iterdir() if d.is_dir()]),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: poetry run python scripts/check_dataset.py <data_dir> [test_data_dir]")
        print("  VD: poetry run python scripts/check_dataset.py datasets/kajima_dataset")
        sys.exit(1)

    dirs = sys.argv[1:]
    all_ok = True

    for data_dir in dirs:
        print(f"\n=== Kiểm tra: {data_dir} ===")
        result = check_data_dir(data_dir)
        if not result["ok"]:
            print(f"  LỖI: {result['error']}")
            if result.get("failed_folders"):
                for path, msg in result["failed_folders"][:10]:
                    print(f"    - {path}: {msg}")
                if len(result["failed_folders"]) > 10:
                    print(f"    ... và {len(result['failed_folders']) - 10} thư mục lỗi khác.")
            all_ok = False
            continue
        print(f"  Tổng ảnh: {result['total_images']}")
        print(f"  Số class: {result['num_classes']}")
        print(f"  Số ảnh mỗi class: {result['class_counts']}")
        print(f"  Các date folder: {result['dates'][:5]}{'...' if len(result['dates']) > 5 else ''}")
        if result.get("failed_folders"):
            print(f"  Cảnh báo - {len(result['failed_folders'])} thư mục bị bỏ qua:")
            for path, msg in result["failed_folders"][:5]:
                print(f"    - {path}: {msg}")
            all_ok = False

    if all_ok and len(dirs) == 1:
        print("\nCấu trúc dataset hợp lệ. Có thể dùng đường dẫn này trong config và chạy train.")
    elif all_ok:
        print("\nTất cả dataset hợp lệ.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
