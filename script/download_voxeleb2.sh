#!/usr/bin/env bash
set -euo pipefail

base_url="https://huggingface.co/datasets/Reverb/voxceleb2/resolve/main"

for i in $(seq -f "%03g" 1 15); do
    file="aac.7z.$i"
    url="${base_url}/${file}?download=true"

    echo "Downloading ${file}..."
    curl -L --fail --retry 5 --retry-delay 5 -o "$file" "$url"
done

meta_file="vox2_meta.csv"
meta_url="${base_url}/${meta_file}?download=true"

echo "Downloading ${meta_file}..."
curl -L --fail --retry 5 --retry-delay 5 -o "$meta_file" "$meta_url"