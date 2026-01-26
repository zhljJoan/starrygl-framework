import os
import time
import torch
from matplotlib import pyplot as plt
import numpy as np
class time_count:
    
    time_forward = 0
    time_backward = 0
    time_memory_updater = 0
    time_embedding = 0
    time_local_update = 0
    time_memory_sync = 0
    time_sample_and_build = 0
    time_memory_fetch = 0
    

    weight_count_remote = 0
    weight_count_local = 0
    ssim_remote = 0
    ssim_cnt = 0
    ssim_local = 0
    ssim_cnt = 0

    attention_delta_ts = []
    attention_value = []
    attention_number = []
    @staticmethod
    def _zero_attention():
        time_count.attention_delta_ts = []
        time_count.attention_value = []
        time_count.attention_number = []
    @staticmethod
    def insert_attention(delta_ts,value,topk):
        pass
        """
        time_count.attention_delta_ts.append(delta_ts)
        time_count.attention_value.append(value)
        time_count.attention_number.append(topk)
        """
    @staticmethod
    def draw_attention(data,epoch):
        pass
        """
        torch.save((time_count.attention_delta_ts,time_count.attention_value,time_count.attention_number),'{}_{}.pt'.format(data,epoch))
        #plt.rcParams['agg.path.chunksize'] = 20000
        ts = torch.cat(time_count.attention_delta_ts).cpu().detach().numpy()
        value = (torch.cat(time_count.attention_value,dim=0)[:,0]).cpu().detach().numpy()
        number = torch.cat(time_count.attention_number).cpu().detach().numpy()
        print(value)
        print(len(ts),max(value),len(number))
        #plt.subplot(2, 1, 1)  # 2行1列，第1个子图
        #plt.plot(ts, value)
        fig, axes = plt.subplots(1, 2,  sharey=False)
        x = ts
        y = value
        counts, xedges, yedges = np.histogram2d(x, y, bins=10, range=[[0,x.max().item()], [0, 0.5]])


        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        counts_normalized = counts / row_sums

        ax = axes[0]
        c= ax.pcolormesh(xedges, yedges, counts_normalized.T)
        cbar = fig.colorbar(c,ax=ax,label='Frequency')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Frequency',fontsize=14)
        ax.set_title('dt vs attention weight',fontsize=14)
        #plt.title('Memory increment vs degree')
        ax.set_xlabel('dt',fontsize=14)
        ax.set_ylabel('attention value',fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
# 绘制第二个子图
        x = number
        y = value
        counts, xedges, yedges = np.histogram2d(x, y, bins=10, range=[[0,x.max().item()], [0, 0.5]])


        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        counts_normalized = counts / row_sums

        ax = axes[1]
        c= ax.pcolormesh(xedges, yedges, counts_normalized.T)
        cbar = fig.colorbar(c,ax=ax,label='Frequency')
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Frequency',fontsize=14)
        ax.set_title('kth neighbor vs attention weight',fontsize=14)
        #plt.title('Memory increment vs degree')
        ax.set_xlabel('kth neighbor',fontsize=14)
        ax.set_ylabel('attention value',fontsize=14)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.tight_layout()
        plt.savefig('{}_{}.png'.format(data,epoch))
        """
    @staticmethod
    def _zero():

        time_count._zero_attention()
        time_count.time_forward = 0
        time_count.time_backward = 0
        time_count.time_memory_updater = 0
        time_count.time_embedding = 0
        time_count.time_local_update = 0
        time_count.time_memory_sync = 0
        time_count.time_sample_and_build = 0
        time_count.time_memory_fetch = 0
    @staticmethod
    def start_gpu():
        # Uncomment for better breakdown timings
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_event,end_event
        #return 0,0
    @staticmethod
    def start():
       return time.perf_counter()
       #return 0,0
    @staticmethod
    def elapsed_event(start_event):
        if isinstance(start_event,tuple):
           start_event,end_event = start_event
           end_event.record()
           end_event.synchronize()
           return start_event.elapsed_time(end_event)
        else:
           torch.cuda.synchronize()
           return time.perf_counter() - start_event

    @staticmethod
    def print():
        print('time_count.time_forward={} time_count.time_backward={} time_count.time_memory_updater={} time_count.time_embedding={} time_count.time_local_update={} time_count.time_memory_sync={} time_count.time_sample_and_build={} time_count.time_memory_fetch={}\n'.format(
            time_count.time_forward,
            time_count.time_backward,
            time_count.time_memory_updater,
            time_count.time_embedding,
            time_count.time_local_update,
            time_count.time_memory_sync,
            time_count.time_sample_and_build,
            time_count.time_memory_fetch ))
    
    