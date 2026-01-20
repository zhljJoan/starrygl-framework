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
TEST_CONFIG = {
    "sample": {
        "num_neighbors": 10,
        "replace": True,
        "time_window":1000000,
        "batch_size":1000,
        "chunks_per_batch":4,
        "shuffle_chunks":True,
        'sample_type':'uniform', 
        'layers':2, 
        'fanout':10, 
        'allowed_offset':-1,
        'equal_root_time':False, 
        'keep_root_time':False,
        
        
    },
    "train": {
        "self_loop": True,
        "memory_type":"GRU",
        "dim_time": 16,
        "hidden_dim": 16,
        "num_heads": 2,
        "num_layers": 2,
        "dropout": 0.1,
        "att_dropout": 0.1,
        "lr":0.0001
    }
}

def get_config(dataset:str, model:str):
    return TEST_CONFIG