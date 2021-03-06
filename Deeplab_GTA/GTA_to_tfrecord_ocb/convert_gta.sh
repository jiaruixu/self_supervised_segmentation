#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to preprocess the Cityscapes dataset. Note (1) the users should
# register the Cityscapes dataset website at
# https://www.cityscapes-dataset.com/downloads/ to download the dataset,
# and (2) the users should download the utility scripts provided by
# Cityscapes at https://github.com/mcordts/cityscapesScripts.
#
# Usage:
#   bash ./preprocess_cityscapes.sh
#
# The folder structure is assumed to be:
#  + datasets
#    - build_cityscapes_data.py
#    - convert_cityscapes.sh
#    + cityscapes
#      + cityscapesscripts (downloaded scripts)
#      + gtFine
#      + leftImg8bit
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="."

# Root path for Cityscapes dataset.
GTA_LEFT_ROOT="/mnt/fcav/PASCAL/tracking_610_filter_3/VOC2012/0cb59a00-51b1-4c27-9808-49ee1734bd71/JPEGImages"
GTA_RIGHT_ROOT="/mnt/fcav/PASCAL/tracking_610_filter_3/VOC2012/0cb59a00-51b1-4c27-9808-49ee1734bd71/PNGStereoImagesRight"

# Create training labels.
# python "${CITYSCAPES_ROOT}/cityscapesscripts/preparation/createTrainIdLabelImgs.py"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${CURRENT_DIR}/leftTfrecord"
OUTPUT_RIGHT_DIR="${CURRENT_DIR}/rightTfrecord"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_RIGHT_DIR}"

BUILD_SCRIPT="${CURRENT_DIR}/build_gta_data.py"

echo "Converting GTA dataset..."
python "${BUILD_SCRIPT}" \
  --gta_root="${GTA_LEFT_ROOT}" \
  --image_side="left" \
  --output_dir="${OUTPUT_DIR}" \
