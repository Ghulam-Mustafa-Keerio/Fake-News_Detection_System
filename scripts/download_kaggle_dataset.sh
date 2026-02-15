#!/usr/bin/env bash
set -euo pipefail

DATASET="clmentbisaillon/fake-and-real-news-dataset"
TARGET_DIR="${1:-.}"

print_help(){
  cat <<EOF
Usage: $0 [target_dir]

This script downloads the Kaggle dataset "${DATASET}" using the kaggle CLI.
Authentication options (choose one):
  - Place your kaggle.json at ~/.kaggle/kaggle.json (recommended)
  - Export env vars KAGGLE_USERNAME and KAGGLE_KEY before running

Examples:
  $0 /path/to/save
  KAGGLE_USERNAME=you KAGGLE_KEY=key $0 /path/to/save
EOF
  exit 0
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  print_help
fi

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install it with: pip install --user kaggle" >&2
  exit 2
fi

mkdir -p ~/.kaggle

# If env vars are provided and no file exists, create kaggle.json
if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
    cat > ~/.kaggle/kaggle.json <<EOF
{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_KEY"}
EOF
    chmod 600 ~/.kaggle/kaggle.json
    echo "Wrote ~/.kaggle/kaggle.json from KAGGLE_USERNAME/KAGGLE_KEY env vars"
  elif [[ -n "${KAGGLE_API_TOKEN:-}" && -n "${KAGGLE_USERNAME:-}" ]]; then
    # Some setups provide a single API token. Use it as 'key' with provided username.
    cat > ~/.kaggle/kaggle.json <<EOF
{"username":"$KAGGLE_USERNAME","key":"$KAGGLE_API_TOKEN"}
EOF
    chmod 600 ~/.kaggle/kaggle.json
    echo "Wrote ~/.kaggle/kaggle.json from KAGGLE_API_TOKEN and KAGGLE_USERNAME env vars"
  fi
fi

if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo "No ~/.kaggle/kaggle.json found and KAGGLE_USERNAME/KAGGLE_KEY not set." >&2
  echo "Create an API token at https://www.kaggle.com/ -> Account -> Create API Token" >&2
  exit 3
fi

echo "Downloading dataset ${DATASET} to ${TARGET_DIR} ..."
kaggle datasets download -d "${DATASET}" --path "${TARGET_DIR}" --unzip

echo "Download complete. Files are in: ${TARGET_DIR}"
