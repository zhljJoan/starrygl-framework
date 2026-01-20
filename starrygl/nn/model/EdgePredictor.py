import torch
class EdgePredictor(torch.nn.Module):

    def __init__(self, dim_in):
        super(EdgePredictor, self).__init__()
        self.dim_in = dim_in
        self.src_fc = torch.nn.Linear(dim_in, dim_in)
        self.dst_fc = torch.nn.Linear(dim_in, dim_in)
        self.out_fc = torch.nn.Linear(dim_in, 1)

    def forward(self, h_pos_src, h_pos_dst, h_neg_src=None,h_neg_dst=None, 
                neg_samples=1,mode='triplet'):
        h_pos_src = self.src_fc(h_pos_src)
        h_pos_dst = self.dst_fc(h_pos_dst)
        h_neg_dst = self.dst_fc(h_neg_dst)
        if mode == 'triplet':
            h_pos_edge = torch.nn.functional.relu(h_pos_src + h_pos_dst)
            
            h_neg_edge = torch.nn.functional.relu(h_pos_src.tile(neg_samples, 1) + h_neg_dst)
        else:
            h_neg_src = self.src_fc(h_neg_src)
            h_pos_edge = torch.nn.functional.relu(h_pos_src + h_pos_dst)
            h_neg_edge = torch.nn.functional.relu(h_neg_src + h_neg_dst)
            
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
