import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# 如果安装了 torch_scatter 建议使用，速度快几十倍
try:
    from torch_scatter import scatter_std
    HAS_SCATTER = True
except ImportError:
    HAS_SCATTER = False
    print("Warning: torch_scatter not found, using slower loop implementation.")


# 计算节点活跃时间的标准差并绘制分布图
def plot_node_activity_distribution(data_path, output_file="node_activity_std.png"):
    path = Path(data_path).expanduser().resolve()
    print(f"Loading {path.name}...")
    state_dict = torch.load(path)
    
    # 1. 提取边和时间戳数据
    dataset = state_dict["dataset"]
    
    # 兼容 snapshot 和 continuous 格式
    if state_dict.get("num_snapshots", 0) == 0:
        # Continuous format: edge_index is [3, E] (src, dst, time) or [2, E] + time separate
        # 根据 prepare_nparts.py 的逻辑
        edge_index = dataset["edge_index"]
        if edge_index.shape[0] > 2:
            src = edge_index[0]
            ts = edge_index[2].float()
        else:
            print("Error: 该数据集 edge_index 只有2行，没有时间戳信息。")
            return
    else:
        # Snapshot format: 拼接所有 snapshot
        ts_list = []
        src_list = []
        for t, data in enumerate(dataset):
            e = data["edge_index"]
            src_list.append(e[0])
            # 使用 snapshot index 作为时间戳
            ts_list.append(torch.full_like(e[0], t, dtype=torch.float))
        src = torch.cat(src_list)
        ts = torch.cat(ts_list)

    print(f"Analyzing {len(ts)} events for {src.max().item() + 1} nodes...")

    # 2. 计算每个节点的活跃时间标准差 (Std Dev)
    # 标准差越小 -> 活跃时间越集中
    if HAS_SCATTER:
        # GPU/Vectorized version
        if torch.cuda.is_available():
            src = src.cuda()
            ts = ts.cuda()
        
        # 计算标准差
        node_stds = scatter_std(ts, src.long(), dim=0)
        
        # 过滤掉非活跃节点 (Std Dev 为 0 的可能是只有一条边或真集中，这里主要关心非0值分布，或者包含0)
        # 这里保留所有度数 > 1 的节点 (度数为1的标准差默认为0)
        num_nodes = int(src.max()) + 1
        degrees = torch.bincount(src.long(), minlength=num_nodes)
        mask = degrees > 1
        valid_stds = node_stds[mask].cpu().numpy()
        
    else:
        # Loop version
        num_nodes = int(src.max().item() + 1)
        node_ts = [[] for _ in range(num_nodes)]
        src_np = src.numpy()
        ts_np = ts.numpy()
        
        for s, t in tqdm(zip(src_np, ts_np), total=len(ts), desc="Aggregating"):
            node_ts[s].append(t)
        
        valid_stds = []
        for t_list in tqdm(node_ts, desc="Calc Std"):
            if len(t_list) > 1:
                valid_stds.append(np.std(t_list))
        valid_stds = np.array(valid_stds)

    # 3. 使用 Matplotlib 画图
    print(f"Plotting histogram for {len(valid_stds)} nodes...")
    
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    # bins: 桶的数量，density: 是否归一化
    n, bins, patches = plt.hist(valid_stds, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title(f'Distribution of Node Activity Time Std-Dev ({path.stem})', fontsize=15)
    plt.xlabel('Standard Deviation of Activity Time (Time Ticks)', fontsize=12)
    plt.ylabel('Number of Nodes', fontsize=12)
    plt.grid(axis='y', alpha=0.5)
    
    # 添加平均值线
    mean_std = np.mean(valid_stds)
    plt.axvline(mean_std, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Std: {mean_std:.2f}')
    plt.legend()
    
    # 保存图片
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved to {output_file}")
    
    # 可选：如果环境支持显示（如 Jupyter），则显示
    # plt.show()


#最大间隔
def analyze_continuity(ts_list, duration):
    """
    输入一个节点的时间戳列表 (已排序)，返回连续性指标
    """
    if len(ts_list) < 2:
        return 0.0 # 只有一个点，无所谓连续
    
    arr = np.array(ts_list)
    #duration = arr[-1] - arr[0]
    if duration == 0:
        return 0.0
        
    # 计算所有相邻时间差
    gaps = np.diff(arr)
    max_gap = np.max(gaps)
    
    return max_gap / duration

def get_node_metrics(ts_list):
    """
    输入一个节点的时间戳列表 (List)，返回 (max_gap, duration)
    """
    if len(ts_list) < 2:
        return None, None # 忽略只有1个或0个交互的节点
    
    # 必须排序才能计算正确的 gap
    arr = np.sort(np.array(ts_list))
    
    # 1. 生命周期长度
    node_duration = arr[-1] - arr[0]
    
    # 2. 最大时间间隔
    gaps = np.diff(arr)
    max_gap = np.max(gaps) if len(gaps) > 0 else 0.0
    
    return max_gap, node_duration

def plot_node_lifecycle_stats(data_path, output_file="node_stats.png"):
    path = Path(data_path).expanduser().resolve()
    print(f"Loading {path.name}...")
    if not path.exists():
        print(f"File not found: {path}")
        return

    try:
        state_dict = torch.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return
    
    # 1. 提取边和时间戳数据
    dataset = state_dict["dataset"]
    
    # 兼容 snapshot 和 continuous 格式
    if state_dict.get("num_snapshots", 0) == 0:
        # Continuous format
        edge_index = dataset["edge_index"]
        if edge_index.shape[0] > 2:
            src = edge_index[0]
            ts = edge_index[2].float()
        else:
            print(f"Error: {path.name} edge_index 只有2行，没有时间戳信息。")
            return
    else:
        # Snapshot format
        ts_list = []
        src_list = []
        for t, data in enumerate(dataset):
            e = data["edge_index"]
            src_list.append(e[0])
            ts_list.append(torch.full_like(e[0], t, dtype=torch.float))
        src = torch.cat(src_list)
        ts = torch.cat(ts_list)

    num_nodes = int(src.max().item() + 1)
    print(f"Analyzing {len(ts)} events for {num_nodes} nodes...")

    # 2. 聚合每个节点的时间戳
    # 由于计算 Max Gap 需要排序，难以单纯用 scatter 完成，这里使用循环
    node_ts = [[] for _ in range(num_nodes)]
    src_np = src.cpu().numpy()
    ts_np = ts.cpu().numpy()
    
    # 聚合
    for s, t in tqdm(zip(src_np, ts_np), total=len(ts), desc="Aggregating"):
        node_ts[s].append(t)
    
    # 3. 计算指标
    valid_max_gaps = []
    valid_durations = []
    
    for t_list in tqdm(node_ts, desc="Calc Metrics"):
        max_gap, duration = get_node_metrics(t_list)
        if max_gap is not None:
            valid_max_gaps.append(max_gap)
            valid_durations.append(duration)
            
    valid_max_gaps = np.array(valid_max_gaps)
    valid_durations = np.array(valid_durations)

    if len(valid_max_gaps) == 0:
        print("No valid nodes found (nodes with > 1 event).")
        return

    # 4. 绘制双图 (上下排列)
    print(f"Plotting histograms for {len(valid_max_gaps)} nodes...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # --- Plot 1: Max Interval (Gap) ---
    ax1.hist(valid_max_gaps, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_title(f'Distribution of Node Max Inter-Event Time ({path.stem})', fontsize=14)
    ax1.set_xlabel('Max Time Interval (Time Ticks)', fontsize=12)
    ax1.set_ylabel('Number of Nodes', fontsize=12)
    ax1.grid(axis='y', alpha=0.5)
    
    mean_gap = np.mean(valid_max_gaps)
    ax1.axvline(mean_gap, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Max Gap: {mean_gap:.2f}')
    ax1.legend()

    # --- Plot 2: Node Lifecycle Duration ---
    ax2.hist(valid_durations, bins=100, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.set_title(f'Distribution of Node Lifecycle Duration ({path.stem})', fontsize=14)
    ax2.set_xlabel('Lifecycle Duration (Max T - Min T)', fontsize=12)
    ax2.set_ylabel('Number of Nodes', fontsize=12)
    ax2.grid(axis='y', alpha=0.5)
    
    mean_dur = np.mean(valid_durations)
    ax2.axvline(mean_dur, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean Duration: {mean_dur:.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Graph saved to {output_file}")
    plt.close() # 关闭图形释放内存




def analyze_node_burstiness(data_path, output_file="burstiness_dist.png"):
    path = Path(data_path).expanduser().resolve()
    print(f"Loading {path.name}...")
    state_dict = torch.load(path)
    dataset = state_dict["dataset"]
    
    # 1. 提取时间戳
    if state_dict.get("num_snapshots", 0) == 0:
        edge_index = dataset["edge_index"]
        src = edge_index[0].numpy()
        ts = edge_index[2].float().numpy()
    else:
        # Snapshot 模式暂略，逻辑类似
        print("Not implemented for snapshots yet.")
        return

    print("Sorting and calculating intervals...")
    # 按节点排序时间戳
    sort_idx = np.lexsort((ts, src))
    sorted_src = src[sort_idx]
    sorted_ts = ts[sort_idx]
    
    # 2. 计算每个节点的统计量 (Mean & Std of Intervals)
    # 利用 np.diff 计算间隔，然后利用 np.add.reduceat 或 pandas 进行分组计算
    # 这里为了代码清晰，使用循环 (可优化为 Pandas groupby)
    
    unique_nodes, node_starts = np.unique(sorted_src, return_index=True)
    # 切分数组
    node_ts_split = np.split(sorted_ts, node_starts[1:])
    
    b_values = []
    valid_node_count = 0
    
    for t_list in tqdm(node_ts_split, desc="Calculating B"):
        if len(t_list) < 3: # 只有两个点算不出方差，忽略
            continue
            
        intervals = np.diff(t_list)
        # 过滤掉 0 间隔 (同一时刻多条边)
        intervals = intervals[intervals > 0]
        
        if len(intervals) < 2:
            continue
            
        mu = np.mean(intervals)
        sigma = np.std(intervals)
        
        # 计算爆发性系数 B
        # B = (sigma - mu) / (sigma + mu)
        if (sigma + mu) > 0:
            b = (sigma - mu) / (sigma + mu)
            b_values.append(b)
            valid_node_count += 1

    b_values = np.array(b_values)
    
    # 3. 绘制 B 值分布直方图
    plt.figure(figsize=(10, 6))
    
    # 绘制 -1 到 1 的分布
    plt.hist(b_values, bins=100, range=(-1, 1), color='purple', alpha=0.7, edgecolor='black')
    
    plt.axvline(0, color='gray', linestyle='--', label='Poisson (Random)')
    plt.axvline(-1, color='green', linestyle='--', label='Periodic (Regular)')
    plt.axvline(1, color='red', linestyle='--', label='Bursty (Power-law)')
    
    mean_b = np.mean(b_values)
    plt.axvline(mean_b, color='yellow', linewidth=2, label=f'Mean B: {mean_b:.2f}')
    
    plt.title(f'Distribution of Burstiness Parameter (B)\n{path.stem}', fontsize=14)
    plt.xlabel('Burstiness Parameter B = (σ - μ) / (σ + μ)', fontsize=12)
    plt.ylabel('Number of Nodes', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    
    # 4. 自动给出结论
    print("-" * 30)
    print(f"Analysis Result for {path.stem}:")
    print(f"Mean B = {mean_b:.3f}")
    if mean_b > 0.3:
        print(">> 结论: 数据集整体呈现强突发性 (Bursty)。")
        print(">> 建议: 使用分级均衡 (Stratified Balancing) 和 帕累托优化。")
    elif mean_b < -0.3:
        print(">> 结论: 数据集整体呈现强周期性 (Periodic)。")
        print(">> 建议: 必须使用相位均衡 (Phase Balancing)。")
    else:
        print(">> 结论: 数据集呈混合模式或接近随机。")
        print(">> 建议: 检查 B 分布是否为双峰 (即一部分周期，一部分突发)。")



def analyze_comprehensive_patterns(data_path, output_file="node_archetypes.png"):
    
    path = Path(data_path).expanduser().resolve()
    print(f"Loading {path.name}...")
    try:
        state_dict = torch.load(path)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return

    dataset = state_dict["dataset"]
    
    # 1. 提取数据
    if state_dict.get("num_snapshots", 0) == 0:
        edge_index = dataset["edge_index"]
        if edge_index.shape[0] > 2:
            src = edge_index[0].numpy()
            ts = edge_index[2].float().numpy()
        else:
            print(f"Skipping {path.name}: No timestamps.")
            return
    else:
        print("Snapshot format not supported yet.")
        return

    print("Sorting data...")
    sort_idx = np.lexsort((ts, src))
    sorted_src = src[sort_idx]
    sorted_ts = ts[sort_idx]
    
    # 2. 遍历计算三大指标
    unique_nodes, node_starts = np.unique(sorted_src, return_index=True)
    node_ts_split = np.split(sorted_ts, node_starts[1:])
    
    lifecycles = []
    burstiness_vals = []
    timestamp_stds = []
    degrees = []  # <--- 修复点1：初始化 degrees 列表
    
    print("Calculating metrics for each node...")
    for t_list in tqdm(node_ts_split):
        # --- 过滤器 Start ---
        if len(t_list) < 3: continue 
            
        duration = t_list[-1] - t_list[0]
        if duration == 0: continue
            
        intervals = np.diff(t_list)
        intervals = intervals[intervals > 0] 
        if len(intervals) < 2: continue
        # --- 过滤器 End ---
            
        # Metric calculation
        t_std = np.std(t_list)
        mu = np.mean(intervals)
        sigma = np.std(intervals)
        
        if (sigma + mu) > 0:
            b = (sigma - mu) / (sigma + mu)
            
            # --- 关键修复点：数据对齐 ---
            # 只有通过了上面所有 if 的节点，才会被加入列表
            # 必须保证所有列表在同一次迭代中同时 append
            lifecycles.append(duration)
            burstiness_vals.append(b)
            timestamp_stds.append(t_std)
            degrees.append(len(t_list)) # <--- 修复点2：在这里记录度数

    # 转换数组
    lifecycles = np.array(lifecycles)
    burstiness_vals = np.array(burstiness_vals)
    timestamp_stds = np.array(timestamp_stds)
    degrees = np.array(degrees) # <--- 修复点3：转换为 numpy array
    
    if len(lifecycles) == 0:
        print(f"No valid nodes found for {path.name}")
        return

    print(f"Plotting {len(lifecycles)} nodes...")

    # 计算点的大小 (Size)
    # 使用 log 归一化，乘以系数调整视觉大小
    sizes = np.log1p(degrees) * 5  
    
    # 3. 绘图
    plt.figure(figsize=(12, 8))
    
    # X=Lifecycle, Y=Burstiness, Color=Std, Size=Degree
    sc = plt.scatter(lifecycles, burstiness_vals, 
                     c=np.log1p(timestamp_stds), 
                     s=sizes,                   
                     cmap='viridis', alpha=0.6)
    
   #plt.xscale('log') 
    plt.colorbar(sc, label='Log(Timestamp Std Dev)')
    
    plt.title(f'Node Archetypes: Lifecycle vs Burstiness\n({path.stem})', fontsize=16)
    plt.xlabel('Node Lifecycle Duration (Log Scale)', fontsize=14)
    plt.ylabel('Burstiness Parameter (B)', fontsize=14)
    
    # 绘制参考区域
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Poisson (Random)')
    plt.axhline(0.5, color='red', linestyle=':', alpha=0.5)
    plt.axhline(-0.5, color='green', linestyle=':', alpha=0.5)
    
    # 标注区域含义
    plt.text(lifecycles.min(), 0.8, "Bursty/Human", fontsize=12, color='red', fontweight='bold')
    plt.text(lifecycles.min(), -0.8, "Periodic/Clock", fontsize=12, color='green', fontweight='bold')
    # 安全获取最大值用于标注位置
    x_max_pos = lifecycles.max() / 100 if len(lifecycles) > 0 else 1
    plt.text(x_max_pos, 0.1, "Random/Noise", fontsize=12, color='gray')
    
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()
    
    
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from tqdm import tqdm

def plot_2d_hist_comparison(data_path, output_file="2d_hist_patterns.png", top_k_percent=10):
    path = Path(data_path).expanduser().resolve()
    print(f"Loading {path.name}...")
    try:
        state_dict = torch.load(path)
        dataset = state_dict["dataset"]
        
        # 1. 提取数据
        if state_dict.get("num_snapshots", 0) == 0:
            edge_index = dataset["edge_index"]
            if edge_index.shape[0] > 2:
                src = torch.cat((edge_index[0],edge_index[1])).numpy()
                ts = torch.cat((edge_index[2],edge_index[2])).float().numpy()
            else:
                print(f"Skipping {path.name}: No timestamps.")
                return
        else:
            print("Snapshot format not supported yet.")
            return

    except Exception as e:
        print(f"Error loading {path}: {e}")
        return

    print("Sorting and Calculating Metrics...")
    sort_idx = np.lexsort((ts, src))
    sorted_src = src[sort_idx]
    sorted_ts = ts[sort_idx]
    
    unique_nodes, node_starts = np.unique(sorted_src, return_index=True)
    print(len(unique_nodes), len(node_starts), len(sorted_ts))
    node_ts_split = np.split(sorted_ts, node_starts[1:])
    
    # 2. 计算 (Duration, Burstiness, Degree)
    data_list = [] # [Duration, B, Degree]
    
    for t_list in tqdm(node_ts_split):
        degree = len(t_list)
        if degree < 3: continue 
        
        std = np.std(t_list)
        #duration = t_list[-1] - t_list[0]
        # Burstiness
        intervals = np.diff(t_list)
        intervals = intervals[intervals > 0]
        if len(intervals) < 2: continue
            
        mu = np.mean(intervals)
        sigma = np.std(intervals)
        
        if (sigma + mu) > 0:
            b = (sigma - mu) / (sigma + mu)
            data_list.append([std, np.average(t_list), degree])
            
    if not data_list:
        print("No valid stats calculated.")
        return

    data_matrix = np.array(data_list) # Col 0: Dur, Col 1: B, Col 2: Deg
    
    # 3. 划分 Hot vs Cold
    print(f"Splitting Top {top_k_percent}% Hot Nodes...")
    sorted_indices = np.argsort(data_matrix[:, 2])[::-1] # 按 Degree 降序
    n_total = len(data_matrix)
    n_hot = int(np.ceil(n_total * (top_k_percent / 100.0)))
    
    hot_indices = sorted_indices[:n_hot]
    cold_indices = sorted_indices[n_hot:]
    
    datasets = {
        "All Nodes": data_matrix,
        f"Hot Nodes (Top {top_k_percent}%)": data_matrix[hot_indices],
        f"Cold Nodes (Tail {100 - top_k_percent}%)": data_matrix[cold_indices]
    }

    # 4. 绘图配置
    # 我们画 2行 x 3列
    # Row 1: Linear X-Axis (Duration)
    # Row 2: Log X-Axis (Duration)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 公共参数
    y_bins = 50 # B值分50个桶
    cmap = 'Spectral_r' # 颜色映射：红-黄-蓝 (红色代表高密度)
    
    # 获取全局 Duration 范围，保证所有子图 X 轴对齐
    min_dur = max(data_matrix[:, 0].min(), 1.0) # 避免 log(0)
    max_dur = data_matrix[:, 0].max()
    
    columns = ["All Nodes", f"Hot Nodes (Top {top_k_percent}%)", f"Cold Nodes (Tail {100 - top_k_percent}%)"]
    
    for col_idx, (name, subset) in enumerate(datasets.items()):
        durations = subset[:, 0]
        burstiness = subset[:, 1]
        
        if len(durations) == 0:
            continue
            
        # --- Row 1: Linear Scale X ---
        ax_lin = axes[0, col_idx]
        # Linear bins
        h = ax_lin.hist2d(durations, burstiness, bins=[100, y_bins], cmin=1,
                          range=[[0, max_dur], y_range],
                         cmap=cmap)
        if col_idx == 0: ax_lin.set_ylabel('Avg ts ($B$)\n(Linear Axis)', fontsize=12)
        ax_lin.set_title(f'{name}\n(Linear Scale)', fontsize=14)
        ax_lin.grid(alpha=0.3)
        plt.colorbar(h[3], ax=ax_lin, label='Count (Log Scale)')
        
        # --- Row 2: Log Scale X ---
        ax_log = axes[1, col_idx]
        # Log bins for X
        # 创建对数分布的桶
        #log_bins_x = np.logspace(np.log10(min_dur), np.log10(max_dur), 100)
        #lin_bins_y = np.linspace(-1, 1, y_bins + 1)
        
        h = ax_log.hist2d(durations, burstiness, cmin=1,
                           cmap=cmap)
        
        ax_log.set_xscale('log') # 设置 X 轴为 Log 显示
        if col_idx == 0: ax_log.set_ylabel('Avg ts ($B$)\n(Log Axis)', fontsize=12)
        ax_log.set_xlabel('ts std', fontsize=12)
        ax_log.set_title(f'{name}\n(Log Scale)', fontsize=14)
        ax_log.grid(alpha=0.3, which='both')
        plt.colorbar(h[3], ax=ax_log, label='Count (Log Scale)')
        
        # 标注参考线
        for ax in [ax_lin, ax_log]:
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5) # Poisson
            ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8) # Bursty zone
            ax.axhline(-0.5, color='gray', linestyle=':', linewidth=0.8) # Periodic zone

    plt.suptitle(f'2D Histogram Analysis: Lifecycle vs Burstiness ({path.stem})', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    plt.savefig(output_file, dpi=300)
    print(f"Saved 2D histograms to {output_file}")
    plt.close()

if __name__ == "__main__":
    # 配置你的路径
    root = Path("/mnt/data/zlj/starrygl-data/ctdg").expanduser().resolve()
    for p in root.glob("*.pth"):
        #if p.stem not in ['REDDIT', 'WIKI']: continue # 测试时可以用白名单
        
        print(f"Processing {p.stem}...")
        plot_2d_hist_comparison(p, f"{p.stem}_avg-std_2d_hist.png")