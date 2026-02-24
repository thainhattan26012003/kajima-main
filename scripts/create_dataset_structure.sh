#!/bin/bash
# Tạo cấu trúc thư mục trống cho train/test dataset.
# Sau khi chạy, copy ảnh vào từng thư mục 7-1-xxx, 7-2-xxx, 7-3-xxx, 7-4-xxx.
#
# Cách dùng:
#   ./scripts/create_dataset_structure.sh datasets/kajima_dataset 20241212
#   ./scripts/create_dataset_structure.sh datasets/kajima_test_dataset 20250108

set -e
if [ $# -lt 2 ]; then
  echo "Usage: $0 <base_dir> <date_folder_name>"
  echo "  VD: $0 datasets/kajima_dataset 20241212"
  echo "  Tạo: base_dir/20241212/7-1-241212, 7-2-241212, 7-3-241212, 7-4-241212"
  exit 1
fi

BASE_DIR="$1"
DATE_NAME="$2"
# Phần hậu tố sau số class (thường rút gọn từ date: 241212 từ 20241212)
SUFFIX="${DATE_NAME:2}"

mkdir -p "$BASE_DIR/$DATE_NAME/7-1-$SUFFIX"
mkdir -p "$BASE_DIR/$DATE_NAME/7-2-$SUFFIX"
mkdir -p "$BASE_DIR/$DATE_NAME/7-3-$SUFFIX"
mkdir -p "$BASE_DIR/$DATE_NAME/7-4-$SUFFIX"

echo "Đã tạo: $BASE_DIR/$DATE_NAME/{7-1-$SUFFIX, 7-2-$SUFFIX, 7-3-$SUFFIX, 7-4-$SUFFIX}"
echo "Tiếp theo: copy ảnh vào từng thư mục tương ứng class 7-1, 7-2, 7-3, 7-4."
