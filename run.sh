#!/bin/bash
set -e
source ~/.miniconda3/bin/activate tgnn_3.10

export OMP_NUM_THREADS=24

MODEL_CHOICES=("mpnn_lstm")
DATA_CHOICES=("WIKI")
# DATA_CHOICES=("ia-slashdot-reply-dir" "soc-flickr-growth" "rec-amazon-ratings" "soc-youtube-growth" "soc-bitcoin")

for model in "${MODEL_CHOICES[@]}"; do
  for dataset in "${DATA_CHOICES[@]}"; do
    echo "[StarryGL] Running model: $model on dataset: $dataset"
    torchrun \
      --nproc_per_node 4 \
      --standalone \
      ./test/test_offline.py \
        --model "$model" \
        --dataset "$dataset" \
        --epochs 2 \
        #--chunk-decay "auto:0.1" \
        #--chunk-order "rand" \
        #--snaps-count 8 \
        #--fulls-count 2
  done;
done
