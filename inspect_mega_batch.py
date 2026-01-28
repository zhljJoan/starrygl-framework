import torch
import sys
import os
import numpy as np

def inspect_mega_batch(path):
    # 加载文件
    try:
        data = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    print(f"=== File: {path} ===")
    print(f"Total Micro-Batches: {len(data)}")
    
    if len(data) == 0:
        print("Empty file.")
        return

    # 取第一个微批次分析
    micro_batch = data[0]
    print(f"Micro-Batch Type: {type(micro_batch)}")
    
    # 如果是列表，说明包含多层结构
    if isinstance(micro_batch, list):
        print(f"Layers/Blocks per batch: {len(micro_batch)}")
        # 通常第0层包含 Target Nodes 和 Task Info
        layer_to_inspect = micro_batch[0]
        print("\n--- Inspecting Micro-Batch 0, Layer 0 (Root/Target) ---")
    elif isinstance(micro_batch, dict):
        layer_to_inspect = micro_batch
        print("\n--- Inspecting Micro-Batch 0 (Flat) ---")
    else:
        print("Unknown structure")
        return

    total_elements = 0
    
    # 遍历字段
    for k, v in layer_to_inspect.items():
        if isinstance(v, torch.Tensor):
            size_mb = v.element_size() * v.numel() / 1024 / 1024
            print(f"{k:<15}: shape={str(list(v.shape)):<15} | dtype={str(v.dtype):<15} | size={size_mb:.4f} MB")
            
            # [关键检查] 警告：如果不是 int32/float32
            if v.dtype == torch.int64 and v.numel() > 0:
                print(f"    [WARN] Potential compression: int64 -> int32 could save 50%")
            if v.dtype == torch.float64:
                print(f"    [WARN] Potential compression: float64 -> float32 could save 50%")
                
        elif isinstance(v, list): 
            print(f"{k:<15}: List len={len(v)} k is{k}")
            if len(v) > 0:
                print(f"    -> First item type: {type(v[0])}")
        else:
            print(f"{k:<15}: {type(v)}")
            
    # 估算
    if os.path.exists(path):
        total_size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"\nTotal File Size on Disk: {total_size_mb:.2f} MB")
        print(f"Average size per micro-batch: {total_size_mb / len(data):.2f} MB")

if __name__ == "__main__":
    # 默认路径或命令行参数
    file_path = "/mnt/data/zlj/starrygl-data/processed_atomic/WIKI_004/part_0/mega_batch_0000.pt" 
    if len(sys.argv) > 1: 
        file_path = sys.argv[1]
    
    inspect_mega_batch(file_path)