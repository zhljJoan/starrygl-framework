import torch

import numpy as np
import pandas as pd

from torch import Tensor
from pathlib import Path
from typing import *

from torch_scatter import scatter_add



class WebDataLoader:
    def __init__(self,
        name: str,
        window = 200,
        skiprows = 2,
        sep = '\s+',
        skiptime = 1,
        lags = 30,
    ) -> None:
        self.name = name
        self.window = window
        self.skiprows = skiprows
        self.sep = sep
        self.skiptime = skiptime
        self.lags = lags
    
    def _get_uvwt(self, data: np.ndarray):
        src = data[:, 0].astype(int)
        dst = data[:, 1].astype(int)
        if data.shape[1] == 3:
            weights = np.ones_like(src)
            timestamps = data[:, 2].astype(int)
        elif data.shape[1] == 4:
            weights = data[:, 2].astype(int)
            timestamps = data[:, 3].astype(int)
        else:
            raise ValueError("data dimensions should be 4 or 5")
        return src, dst, weights, timestamps
    
    def _get_mapped_uv(self, src: np.ndarray, dst: np.ndarray):
        uniq_nodes = np.unique(np.concatenate([src, dst]))
        num_nodes = len(uniq_nodes)
        node_mapping = {old: new for new, old in enumerate(uniq_nodes)}
        src_mapped = np.vectorize(node_mapping.get)(src)
        dst_mapped = np.vectorize(node_mapping.get)(dst)
        return src_mapped, dst_mapped, num_nodes
    
    def _get_masked_snapshots(self, data: np.ndarray):
        src, dst, wei, tss = self._get_uvwt(data)
        src_mapped, dst_mapped, num_nodes = self._get_mapped_uv(src, dst)

        uniq_tss = np.unique(tss)
        uniq_tss = np.sort(uniq_tss)[self.skiptime:]
        start_ts = uniq_tss[0]
        time_itv = (uniq_tss[-1] - uniq_tss[0]) / self.window

        edge_snapshots = [None for i in range(self.window - self.lags)]
        edge_weight_snapshots = [None for i in range(self.window - self.lags)]
        N = num_nodes

        for i in range(self.window - self.lags):
            mask = (tss >= start_ts + i * time_itv) & (tss < start_ts + (i + self.lags + 1) * time_itv)
            edge_snapshots[i] = np.array([src_mapped[mask], dst_mapped[mask]])
            edge_weight_snapshots[i] = np.array(wei[mask])
        
        edge_snapshots = [arr for arr in edge_snapshots if arr.size > 100] # filter out small snapshots
        edge_weight_snapshots = [arr for arr in edge_weight_snapshots if arr.size > 50]

        return edge_snapshots, edge_weight_snapshots, num_nodes

    def _read_web_data(self, path):
        path = Path(path).expanduser().resolve()
        data = pd.read_csv(path, skiprows=self.skiprows, sep=self.sep).to_numpy()

        num_edges = data.shape[0]
        edge_snapshots, edge_weight_snapshots, num_nodes = self._get_masked_snapshots(data)

        edge_snapshots = [torch.from_numpy(arr).type(torch.int32) for arr in edge_snapshots]
        edge_weight_snapshots = [torch.from_numpy(arr).type(torch.float32) for arr in edge_weight_snapshots]
        
        num_nodes = int(num_nodes)
        num_edges = int(num_edges)

        return edge_snapshots, edge_weight_snapshots, num_nodes, num_edges
    
    def get_dataset(self, root = None):
        root = Path(root).expanduser().resolve()
        path = root / self.name / f"{self.name}.edges"
        edges, edge_weights, num_nodes, num_edges = self._read_web_data(path)
        
        num_snapshots: int = len(edges)
        num_life_edges = sum(x.size(0) for x in edge_weights)
        
        xs: List[Tensor] = []
        ys: List[Tensor | None] = []
        for i in range(num_snapshots):
            e, w = edges[i].long(), edge_weights[i]
            i_deg = scatter_add(w, e[1], dim=0, dim_size=num_nodes).float()
            o_deg = scatter_add(w, e[0], dim=0, dim_size=num_nodes).float()
            xs.append(torch.stack([i_deg, o_deg], dim=1))
            if i > 0:
                ys.append(torch.log(i_deg + 1))
        ys.append(None) # no label for the last snapshot

        dataset = []
        for i in range(num_snapshots):
            e, w = edges[i], edge_weights[i]
            x, y = xs[i], ys[i]
            dataset.append({
                "edge_index": e,
                "edge_weight": w,
                "x": x,
                "y": y,
            })
        
        state_dict = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_snapshots": num_snapshots,
            "num_life_edges": num_life_edges,
            "dataset": dataset,
        }
        return state_dict



IaSlashdotReplyDirDatasetLoader = WebDataLoader(
    name="ia-slashdot-reply-dir",
    window=200,
    skiprows=2,
    sep='\s+',
    skiptime=1,
    lags=30,
)

RecAmazonRatingsDatasetLoader = WebDataLoader(
    name="rec-amazon-ratings",
    window=100,
    skiprows=2,
    sep=',',
    skiptime=1,
    lags=30,
)

RecAmzBooksDatasetLoader = WebDataLoader(
    name="rec-amz-Books",
    window=100,
    skiprows=0,
    sep=',',
    skiptime=0,
    lags=0,
)

SocBitcoinDatasetLoader = WebDataLoader(
    name="soc-bitcoin",
    window=100,
    skiprows=0,
    sep='\s+',
    skiptime=1,
    lags=10,
)

SocFlickrGrowthDatasetLoader = WebDataLoader(
    name="soc-flickr-growth",
    window=100,
    skiprows=1,
    sep='\s+',
    skiptime=1,
    lags=30,
)

SocYoutubeGrowthDatasetLoader = WebDataLoader(
    name="soc-youtube-growth",
    window=100,
    skiprows=2,
    sep='\s+',
    skiptime=1,
    lags=20,
)

StackexchDatasetLoader = WebDataLoader(
    name="stackexch",
    window=200,
    skiprows=2,
    sep='\s+',
    skiptime=1,
    lags=30,
)

web_data_loaders = [
    IaSlashdotReplyDirDatasetLoader,
    #RecAmazonRatingsDatasetLoader,
    #SocBitcoinDatasetLoader,
    #SocFlickrGrowthDatasetLoader,
    #SocYoutubeGrowthDatasetLoader,
    # RecAmzBooksDatasetLoader,
    # StackexchDatasetLoader,
]


if __name__ == "__main__":
    src_root = "/mnt/data/zlj/starrygl-data/raw"
    tgt_root = "/mnt/data/zlj/starrygl-data"

    root = Path(tgt_root).expanduser().resolve() / "web"
    root.mkdir(parents=True, exist_ok=True)

    for loader in web_data_loaders:
        data = loader.get_dataset(src_root)
        
        num_nodes: int = data["num_nodes"]
        num_edges: int = data["num_edges"]
        num_life_edges: int = data["num_life_edges"]

        torch.save(data, root / f"{loader.name}.pth")
        print(loader.name, num_nodes, num_edges, num_life_edges)
    