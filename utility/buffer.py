import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import numpy as np


class Buffer(Dataset):
    def __init__(self, graph, freq_alloc, action, act_log_prob, value, ongoing, cir, device):
        self._device = device
        self._ptr = graph['ptr']
        self._graph_list = graph.to_data_list()
        self._batch_size = len(self._graph_list)
        # Some algorithm to separate the last frequency allocation
        self._freq_alloc = freq_alloc[:-1]
        self._final_freq_alloc = freq_alloc[-1]

        self._action = action
        self._act_log_prob = act_log_prob
        self._value = value
        self._ongoing = ongoing
        self._cir = cir
        self._reward = None
        self._return = None
        self._idx = []
        num_step = torch.sum(self._ongoing.int(), dim=0)
        for s in range(self._batch_size):
            self._idx.append(np.stack((s * np.ones((num_step[s],)), np.arange(0, num_step[s])), axis=1).astype(int))
        self._idx = np.concatenate(self._idx, axis=0)
        
    def __len__(self):
        return self._idx.shape[0]

    def __getitem__(self, idx):
        samp, step = self._idx[idx][0], self._idx[idx][1]
        out = {}
        out['graph'] = self._graph_list[samp].to(self._device)
        out['freq_alloc'] = self._freq_alloc[step, self._ptr[samp]: self._ptr[samp+1], :].to(self._device)            
        out['action'] = self._action[step, samp, :].to(self._device)
        out['act_log_prob'] = self._act_log_prob[step, samp].to(self._device)
        out['value'] = self._value[step, samp].to(self._device)
        if self._reward is not None:
            out['reward'] = self._reward[step, samp].to(self._device)
        if self._return is not None:
            out['return'] = self._return[step, samp].to(self._device)
        return out

    def cal_reward(self, cir_thresh):
        success = (self._cir >= cir_thresh)
        r = []
        for s in range(self._batch_size):
            tmp = success[:, self._ptr[s]: self._ptr[s + 1]]
            r.append(torch.sum(tmp.int(), dim=1))
        r = torch.stack(r, dim=1).to(torch.float)
        cur_r = torch.zeros((r.shape[0] + 1, r.shape[1]))
        cur_r[1:, :] = r
        prev_r = torch.zeros_like(cur_r)
        prev_r[2:, :] = r[:-1, :]
        self._reward = cur_r - prev_r

    def get_performance(self, cir_thresh):
        cir = self._cir[-1, :].detach().cpu()
        success = torch.sum((cir >= cir_thresh).int()).item()
        total = cir.shape[0]
        return cir, success, total

    def cal_lambda_return(self, gamma, lamb):
        num_step = self._reward.shape[0] - 1
        ret = torch.zeros((num_step, self._batch_size))
        ret[num_step - 1, :] = self._reward[num_step, :]
        value = torch.where(self._ongoing, self._value, 0.0)
        for step in range(num_step - 2, -1, -1):
            ret[step, :] = ((1 - lamb) * gamma * value[step + 1, :] + self._reward[step + 1, :]
                            + lamb * gamma * ret[step + 1, :])
        self._return = ret

def collate_fn(samp):
    keys = list(samp[0].keys())
    out = {k: [] for k in keys}
    for s in samp:
        for k in keys:
            out[k].append(s[k])
    out['graph'] = Batch.from_data_list(out['graph'])
    out['freq_alloc'] = torch.concatenate(out['freq_alloc'], dim=0)
    out['action'] = torch.stack(out['action'], dim=0)
    out['act_log_prob'] = torch.stack(out['act_log_prob'], dim=0)
    out['value'] = torch.stack(out['value'], dim=0)
    if 'reward' in keys:
        out['reward'] = torch.stack(out['reward'], dim=0)
    if 'return' in keys:
        out['return'] = torch.stack(out['return'], dim=0)
    return out


def get_buffer_dataloader(buf, batch_size, shuffle=True):
    dataloader = DataLoader(buf, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    pass


