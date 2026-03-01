#!/usr/bin/env python3
"""
Chia thư mục datasets thành train và test, đồng thời đổi tên thư mục class
sang format mà KajimaDataset yêu cầu: 7-<class>-<date>.

Cấu trúc nguồn mong đợi:
  <source>/250123/7-1_20250123/IMG_0027.jpg ...
  <source>/250123/7-2_20250123/...
  (hoặc format cũ: 試料7-1, 試料7-2, 試料6-1, 試料7-4)

Mapping:
  7-1_* hoặc 試料7-1 → 7-1-<date>  (class 1)
  7-2_* hoặc 試料7-2 → 7-2-<date>  (class 2)
  7-3_* hoặc 6-1_* hoặc 試料6-1 → 7-3-<date>  (class 3)
  7-4_* hoặc 試料7-4 → 7-4-<date>  (class 4)

Cách chạy:
  poetry run python scripts/split_train_test.py --source /path/to/image_kajima
  poetry run python scripts/split_train_test.py --source image_kajima --test-ratio 0.2
  poetry run python scripts/split_train_test.py --source image_kajima --test-dates 251001 251002 251106 251111
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Mapping: tên thư mục hiện tại (đúng từng chữ) → (số class, tên mới không có date)
# Dùng cho format cũ: 試料7-1, 試料7-2, ...
FOLDER_MAP_EXACT = {
    "試料7-1": (1, "7-1"),
    "試料7-2": (2, "7-2"),
    "試料6-1": (3, "7-3"),
    "試料7-4": (4, "7-4"),
}

# Mapping: prefix tên thư mục (bắt đầu bằng) → (số class, base name)
# Dùng cho format mới: 7-1_20250123, 7-2_20250123, 6-1_..., 7-4_...
FOLDER_MAP_PREFIX = {
    "7-1_": (1, "7-1"),
    "7-2_": (2, "7-2"),
    "7-3_": (3, "7-3"),
    "6-1_": (3, "7-3"),
    "7-4_": (4, "7-4"),
}


def _get_class_and_base(src_class_dir: Path) -> tuple[int, str] | None:
    """Từ thư mục class nguồn, trả về (class_num, base_name) hoặc None nếu không map."""
    name = src_class_dir.name
    if name in FOLDER_MAP_EXACT:
        return FOLDER_MAP_EXACT[name]
    for prefix, (class_num, base_name) in FOLDER_MAP_PREFIX.items():
        if name.startswith(prefix):
            return (class_num, base_name)
    return None


def copy_tree_with_rename(
    src_date_dir: Path,
    dst_date_dir: Path,
    dry_run: bool = False,
) -> int:
    """Copy toàn bộ ảnh từ src_date_dir sang dst_date_dir với tên thư mục đã đổi. Trả về số file."""
    date_suffix = src_date_dir.name  # VD: 250123
    count = 0
    for src_class_dir in src_date_dir.iterdir():
        if not src_class_dir.is_dir():
            continue
        if src_class_dir.name.startswith(".") or src_class_dir.name == "__MACOSX":
            continue
        mapped = _get_class_and_base(src_class_dir)
        if mapped is None:
            continue
        _class_num, base_name = mapped
        new_folder_name = f"{base_name}-{date_suffix}"  # 7-1-250123
        dst_class_dir = dst_date_dir / new_folder_name
        if not dry_run:
            dst_class_dir.mkdir(parents=True, exist_ok=True)
        for f in src_class_dir.iterdir():
            if f.is_file():
                count += 1
                if not dry_run:
                    shutil.copy2(f, dst_class_dir / f.name)
    return count


def main():
    parser = argparse.ArgumentParser(description="Chia datasets thành train và test")
    parser.add_argument(
        "--source",
        type=str,
        default="datasets",
        help="Thư mục nguồn chứa các date (mặc định: datasets)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default="datasets/kajima_dataset",
        help="Thư mục train đầu ra (mặc định: datasets/kajima_dataset)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="datasets/kajima_test_dataset",
        help="Thư mục test đầu ra (mặc định: datasets/kajima_test_dataset)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Tỉ lệ số date dùng làm test (mặc định: 0.15). Bỏ qua nếu dùng --test-dates.",
    )
    parser.add_argument(
        "--test-dates",
        type=str,
        nargs="*",
        default=None,
        help="Liệt kê chính xác các date dùng làm test (VD: 251001 251002). Nếu có thì bỏ qua --test-ratio.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ in kế hoạch, không copy file",
    )
    parser.add_argument(
        "--list-structure",
        action="store_true",
        help="In cấu trúc thư mục nguồn (vài date đầu) rồi thoát, để kiểm tra tên class",
    )
    args = parser.parse_args()

    root = Path(args.source)
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)

    if not root.exists():
        print(f"Thư mục nguồn không tồn tại: {root}")
        return 1

    date_folders = sorted([d.name for d in root.iterdir() if d.is_dir() and d.name.isdigit()])
    if not date_folders:
        print(f"Không tìm thấy thư mục date (tên toàn số) trong {root}")
        return 1

    if args.list_structure:
        print("Cấu trúc thư mục nguồn (script chỉ copy khi tên class khớp 7-1_, 7-2_, 6-1_, 7-4_ hoặc 試料7-1,...):\n")
        for date in date_folders[:5]:
            p = root / date
            subdirs = [d.name for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
            files_in_first = []
            if subdirs:
                first_sub = p / subdirs[0]
                files_in_first = [f.name for f in first_sub.iterdir() if f.is_file()][:3]
            print(f"  {date}/")
            for s in subdirs[:8]:
                print(f"    {s}/")
            if len(subdirs) > 8:
                print(f"    ... và {len(subdirs) - 8} thư mục khác")
            if subdirs and files_in_first:
                print(f"    (ví dụ file trong {subdirs[0]}: {files_in_first})")
            print()
        print("Nếu tên thư mục class không giống 7-1_..., 7-2_..., cần thêm mapping trong script (FOLDER_MAP_PREFIX / FOLDER_MAP_EXACT).")
        return 0

    if args.test_dates:
        test_dates_set = set(args.test_dates)
        train_dates = [d for d in date_folders if d not in test_dates_set]
        test_dates = [d for d in date_folders if d in test_dates_set]
        unknown = test_dates_set - set(date_folders)
        if unknown:
            print(f"Cảnh báo: --test-dates có date không tồn tại: {unknown}")
    else:
        n = len(date_folders)
        n_test = max(1, int(n * args.test_ratio))
        train_dates = date_folders[:-n_test]
        test_dates = date_folders[-n_test:]

    print(f"Tổng số date: {len(date_folders)}")
    print(f"Train: {len(train_dates)} date → {train_dir}")
    print(f"Test:  {len(test_dates)} date → {test_dir}")
    if args.dry_run:
        print("\n[DRY RUN] Không ghi file.")
        for d in train_dates[:3]:
            print(f"  Train: {d}")
        if len(train_dates) > 3:
            print(f"  ... và {len(train_dates) - 3} date khác")
        for d in test_dates:
            print(f"  Test:  {d}")
    else:
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

    total_train, total_test = 0, 0
    for date in train_dates:
        n = copy_tree_with_rename(root / date, train_dir / date, dry_run=args.dry_run)
        total_train += n
    for date in test_dates:
        n = copy_tree_with_rename(root / date, test_dir / date, dry_run=args.dry_run)
        total_test += n

    print(f"\nTrain: {total_train} ảnh trong {len(train_dates)} date")
    print(f"Test:  {total_test} ảnh trong {len(test_dates)} date")
    if total_train == 0 and total_test == 0:
        print("\n⚠️  Không copy được ảnh nào. Tên thư mục class có thể không khớp mapping.")
        print("   Chạy với --list-structure để xem cấu trúc thực tế:")
        print("   python scripts/split_train_test_1.py --list-structure")
    if not args.dry_run:
        print(f"\nĐã tạo:\n  {train_dir}\n  {test_dir}")
        print("\nCập nhật config với:")
        print(f'  "train_dataset_dir": "{train_dir}",')
        print(f'  "test_dataset_dir": "{test_dir}",')
    return 0


if __name__ == "__main__":
    exit(main())
