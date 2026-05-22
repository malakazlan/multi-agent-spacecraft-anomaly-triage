#!/usr/bin/env bash
# Fetch the real NASA SMAP/MSL telemetry (~260 MB).
# The original JPL S3 bucket (s3-us-west-2.amazonaws.com/telemanom/data.zip)
# is no longer publicly accessible, so this pulls from the Kaggle mirror
# (patrickfleith/nasa-anomaly-detection-dataset-smap-msl) which is the same
# Hundman et al. 2018 dataset.
#
# Requires the Kaggle CLI and an API token:
#   pip install kaggle
#   # then place your kaggle.json at ~/.kaggle/kaggle.json
#   # (get it from https://www.kaggle.com/settings -> Create New API Token)
#
# After it completes you'll have data/train/*.npy and data/test/*.npy and
# `python src/run.py --backend lstm` produces the real benchmark number.
set -e
cd "$(dirname "$0")"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found. Install it with: pip install kaggle"
  echo "Then put your API token at ~/.kaggle/kaggle.json (chmod 600 on Unix)."
  exit 1
fi

echo "Downloading NASA SMAP/MSL telemetry from Kaggle mirror..."
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl -p data --unzip

# Kaggle zip nests as data/data/data/{train,test,...}. Flatten it.
if [ -d data/data/data/train ] && [ -d data/data/data/test ]; then
  echo "Flattening nested folders..."
  mv data/data/data/train data/train
  mv data/data/data/test data/test
  rm -rf data/data
fi

echo "Done."
echo "train channels: $(ls data/train | wc -l)"
echo "test  channels: $(ls data/test  | wc -l)"