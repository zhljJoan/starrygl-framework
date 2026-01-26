import torch

from starrygl.utils.context import DistributedContext
import torch_scatter
class mailbox:
    def __init__(self, num_nodes, mailbox_size, dim_out, dim_edge_feat, _next_mail_pos = None, _update_mail_pos = False, device = None):
        self.num_nodes = num_nodes
        self.device = device
        self.mailbox_size = mailbox_size
        self.dim_out = dim_out
        self.dim_edge_feat = dim_edge_feat
        self.mailbox = torch.zeros(self.num_nodes, 
                                mailbox_size, 
                              2 * dim_out + dim_edge_feat,
                              device = device, dtype=torch.float32)
        self.mailbox_ts = torch.zeros((self.num_nodes, 
                                  mailbox_size), 
                                dtype=  torch.float32,device = device)
        self.next_mail_pos = torch.zeros((num_nodes), dtype=torch.long) if _next_mail_pos is None else _next_mail_pos
        self.update_mail_pos = _update_mail_pos
        self.device = torch.device('cpu')
        
    @property
    def shape(self):
        return (self.num_nodes, 
                self.mailbox_size, 
                2 * self.dim_out + self.dim_edge_feat)
    
    def reset(self):
        self.mailbox.fill_(0)
        self.mailbox_ts.fill_(0)
        self.next_mail_pos.fill_(0)

    def move_to_gpu(self):
        device = DistributedContext.get_default_context().get_device()
        self.mailbox = self.mailbox.to(device)
        self.mailbox_ts = self.mailbox_ts.to(device)
        self.next_mail_pos = self.next_mail_pos.to(device)
        self.device = device

    def set_mailbox_local(self,index,source,source_ts,Reduce_Op = None):
        if Reduce_Op == 'max' and self.num_parts > 1:
            unq_id,inv = index.unique(return_inverse = True)
            max_ts,id =  torch_scatter.scatter_max(source_ts.to(self.device),inv.to(self.device),dim=0)
            source_ts = max_ts
            source = source[id]
            index = unq_id
        #print(self.next_mail_pos[index])
        self.mailbox_ts.accessor.data[index.to(self.device), self.next_mail_pos[index.to(self.device)]] = source_ts
        self.mailbox.accessor.data[index.to(self.device), self.next_mail_pos[index.to(self.device)]] = source
        if self.memory_param['mailbox_size'] > 1:
            self.next_mail_pos[index.to(self.device)] = torch.remainder(
                self.next_mail_pos[index.to(self.device)] + 1, 
                self.memory_param['mailbox_size'])
    
    def get_update_mail(self,
                 src,dst,ts,edge_feats,
                 mem_src, mem_dst ,embedding=None,use_src_emb=False,use_dst_emb=False,
                 block = None,Reduce_score=None,):
        if edge_feats is not None:
            edge_feats = edge_feats.to(self.device).to(self.mailbox.dtype)
        src = src.to(self.device)
        dst = dst.to(self.device)
        index = torch.cat([src, dst]).reshape(-1)
        if embedding is not None:
            emb_src = embedding[:src.numel()]
            emb_dst = embedding[src.numel():]
        src_mail = torch.cat([emb_src if use_src_emb else mem_src, emb_dst if use_dst_emb else mem_dst], dim=1)
        dst_mail = torch.cat([emb_dst if use_src_emb else mem_dst, emb_src if use_dst_emb else mem_src], dim=1)
        if edge_feats is not None:
            src_mail = torch.cat([src_mail.to(self.device), edge_feats.to(self.device)], dim=1)
            dst_mail = torch.cat([dst_mail.to(self.device), edge_feats.to(self.device)], dim=1)
            
        mail = torch.cat([src_mail, dst_mail], dim=0)
        mail_ts = torch.cat((ts,ts),-1).to(self.device).to(self.mailbox_ts.dtype)

       
        
        unq_index,inv = torch.unique(index,return_inverse = True)
        max_ts,idx = torch_scatter.scatter_max(mail_ts,inv.to(mail_ts.device),0)
        mail_ts = max_ts
        mail = mail[idx]
        index = unq_index
        return index,mail,mail_ts
    
    def update_next_mail_pos(self):
        if self.update_mail_pos is not None:
            nid = torch.where(self.update_mail_pos == 1)[0]
            self.next_mail_pos[nid] = torch.remainder(self.next_mail_pos[nid] + 1, self.memory_param['mailbox_size'])
            self.update_mail_pos.fill_(0)

    def get_message(self, idx):
        return self.mailbox[idx], self.mailbox_ts[idx]
    
    @property
    def shape(self):
        return (self.num_nodes, 
                self.mailbox_size, 
                2 * self.dim_out + self.dim_edge_feat)