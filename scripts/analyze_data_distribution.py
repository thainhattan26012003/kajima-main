import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def scan_data_dir(data_dir: str) -> dict:

    data_path = Path(data_dir)
    if not data_path.exists():
        return {"ok": False, "error": f"Thư mục không tồn tại: {data_dir}"}

    image_labels = []
    failed_folders = []
    per_date = defaultdict(lambda: Counter())

    for date_name in sorted(data_path.iterdir()):
        if not date_name.is_dir():
            continue
        for img_dn in sorted(date_name.iterdir()):
            if not img_dn.is_dir():
                continue
            try:
                parts = img_dn.name.split("-")
                if len(parts) != 3:
                    failed_folders.append((str(img_dn), f"Tên cần 3 phần (xxx-class-xxx), có {len(parts)}"))
                    continue
                _, label_str, _ = parts
                label = int(label_str) - 1
                count = 0
                for f in img_dn.iterdir():
                    if f.is_file():
                        image_labels.append((f, label))
                        per_date[date_name.name][label] += 1
                        count += 1
                if count == 0:
                    failed_folders.append((str(img_dn), "Không có file ảnh"))
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
        "class_counts": dict(zip(sorted_idx, cls_num_list)),
        "class_counts_1based": dict(zip([i + 1 for i in sorted_idx], cls_num_list)),
        "failed_folders": failed_folders,
        "dates": sorted([d.name for d in data_path.iterdir() if d.is_dir()]),
        "per_date": dict(per_date),
    }


def load_config_paths(config_path: str) -> tuple:
    """Đọc train_dataset_dir, test_dataset_dir từ config JSON."""
    path = Path(config_path)
    if not path.exists():
        return None, None
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    data = cfg.get("data", {})
    return data.get("train_dataset_dir"), data.get("test_dataset_dir")


def build_summary_table(train_result: dict, test_result: dict) -> pd.DataFrame:
    """Tạo bảng tổng hợp: class, count train, count test, % train, % test."""
    all_classes = set()
    if train_result and train_result.get("ok"):
        all_classes.update(train_result["class_counts"].keys())
    if test_result and test_result.get("ok"):
        all_classes.update(test_result["class_counts"].keys())
    all_classes = sorted(all_classes)

    rows = []
    for c in all_classes:
        train_count = train_result["class_counts"].get(c, 0) if train_result and train_result.get("ok") else 0
        test_count = test_result["class_counts"].get(c, 0) if test_result and test_result.get("ok") else 0
        train_total = train_result["total_images"] if train_result and train_result.get("ok") else 0
        test_total = test_result["total_images"] if test_result and test_result.get("ok") else 0
        pct_train = (train_count / train_total * 100) if train_total else 0
        pct_test = (test_count / test_total * 100) if test_total else 0
        rows.append({
            "class": c,
            "class_name": f"7-{c + 1}",
            "train_count": train_count,
            "test_count": test_count,
            "train_pct": round(pct_train, 1),
            "test_pct": round(pct_test, 1),
        })
    return pd.DataFrame(rows)


def plot_distribution(train_result: dict, test_result: dict, output_dir: Path) -> None:
    """Vẽ biểu đồ cột: phân bố theo class (train và/hoặc test)."""
    all_classes = set()
    if train_result and train_result.get("ok"):
        all_classes.update(train_result["class_counts"].keys())
    if test_result and test_result.get("ok"):
        all_classes.update(test_result["class_counts"].keys())
    all_classes = sorted(all_classes)
    if not all_classes:
        return

    labels = [f"7-{c + 1}" for c in all_classes]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    if train_result and train_result.get("ok"):
        train_counts = [train_result["class_counts"].get(c, 0) for c in all_classes]
        ax.bar(x - width / 2, train_counts, width, label="Train", color="steelblue")
    if test_result and test_result.get("ok"):
        test_counts = [test_result["class_counts"].get(c, 0) for c in all_classes]
        ax.bar(x + width / 2, test_counts, width, label="Test", color="coral")

    ax.set_ylabel("Số ảnh")
    ax.set_xlabel("Class")
    ax.set_title("Phân bố số ảnh theo class (Train / Test)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_file = output_dir / "distribution_by_class.png"
    fig.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Đã lưu biểu đồ: {out_file}")


def plot_pie_per_split(result: dict, split_name: str, output_dir: Path) -> None:
    """Vẽ biểu đồ tròn cho một split (train hoặc test)."""
    if not result.get("ok"):
        return
    counts = result["class_counts"]
    if not counts:
        return
    labels = [f"7-{c + 1}" for c in sorted(counts.keys())]
    sizes = [counts[c] for c in sorted(counts.keys())]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Phân bố {split_name} (n={result['total_images']})")
    out_file = output_dir / f"pie_{split_name.lower()}.png"
    fig.savefig(out_file, dpi=150)
    plt.close()
    print(f"  Đã lưu: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Phân tích phân bố dữ liệu dataset")
    parser.add_argument("-c", "--config", type=str, default=None, help="Đường dẫn file config JSON")
    parser.add_argument("-t", "--train-dir", type=str, default=None, help="Thư mục train dataset")
    parser.add_argument("-e", "--test-dir", type=str, default=None, help="Thư mục test dataset")
    parser.add_argument("-o", "--output-dir", type=str, default="analysis", help="Thư mục lưu biểu đồ (mặc định: analysis)")
    args = parser.parse_args()

    train_dir = args.train_dir
    test_dir = args.test_dir
    if args.config:
        t, e = load_config_paths(args.config)
        if t is not None:
            train_dir = t
        if e is not None:
            test_dir = e
        print(f"Đọc config: {args.config}")
        print(f"  train_dataset_dir: {train_dir}")
        print(f"  test_dataset_dir:  {test_dir}")

    if not train_dir and not test_dir:
        print("Cần chỉ định -c config hoặc -t train_dir (và tùy chọn -e test_dir).")
        sys.exit(1)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_result = None
    test_result = None

    if train_dir:
        print(f"\n=== Train: {train_dir} ===")
        train_result = scan_data_dir(train_dir)
        if not train_result["ok"]:
            print(f"  Lỗi: {train_result['error']}")
        else:
            print(f"  Tổng ảnh: {train_result['total_images']}")
            print(f"  Số class: {train_result['num_classes']}")
            print(f"  Số ảnh mỗi class: {train_result['class_counts_1based']}")

    if test_dir:
        print(f"\n=== Test: {test_dir} ===")
        test_result = scan_data_dir(test_dir)
        if not test_result["ok"]:
            print(f"  Lỗi: {test_result['error']}")
        else:
            print(f"  Tổng ảnh: {test_result['total_images']}")
            print(f"  Số class: {test_result['num_classes']}")
            print(f"  Số ảnh mỗi class: {test_result['class_counts_1based']}")

    if (train_result and not train_result["ok"]) and (test_result and not test_result["ok"]):
        print("\nKhông có dữ liệu hợp lệ để phân tích.")
        sys.exit(1)

    # Bảng tổng hợp
    df = build_summary_table(train_result or {}, test_result or {})
    print("\n--- Bảng phân bố ---")
    print(df.to_string(index=False))

    csv_path = output_path / "distribution_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nĐã lưu bảng: {csv_path}")

    # Biểu đồ
    print("\n--- Biểu đồ ---")
    plot_distribution(train_result or {}, test_result or {}, output_path)
    if train_result and train_result.get("ok"):
        plot_pie_per_split(train_result, "Train", output_path)
    if test_result and test_result.get("ok"):
        plot_pie_per_split(test_result, "Test", output_path)

    # Cảnh báo mất cân bằng
    if not df.empty and "train_count" in df.columns:
        counts = df["train_count"].values
        if counts.max() > 0 and (counts.max() / max(counts.min(), 1)) > 2:
            print("\n⚠ Cảnh báo: Dữ liệu train mất cân bằng giữa các class. Có thể dùng class_aware_sampler hoặc loss CBL/LA trong config.")


if __name__ == "__main__":
    main()
