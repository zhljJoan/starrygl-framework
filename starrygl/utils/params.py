MODEL_CHOICES = [
    "mpnn_lstm",
]

DATASET_CHOICES = [
    "ia-slashdot-reply-dir",
    "soc-bitcoin",
    "soc-flickr-growth",
    "rec-amazon-ratings",
    "soc-youtube-growth",
]

CONFIG_DATASET_TO_DIM = {
    "ia-slashdot-reply-dir": [16],
    "soc-bitcoin": [2],
    "soc-flickr-growth": [8],
    "rec-amazon-ratings": [16],
    "soc-youtube-growth": [8],
}

