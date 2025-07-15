import numpy as np
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from simulator.network_generator import InterfGraphDataset, DynamicDataset
from model.actor_critic import Actor, Critic
from utility.buffer import Buffer, get_buffer_dataloader
import wandb
import networkx as nx

class Engine:
    def __init__(self, params_file, device):
        self._device = device
        # load configuration file
        self._config = {}
        conf_dir = Path(__file__).parents[0]
        with open(conf_dir / params_file, 'r') as f:
            self._config = yaml.safe_load(f)
        self._num_freq_ch = self._config['num_freq_ch']
        self._eval_iter = self._config['eval.iter']
        # calculate the boundaries for quantizing power attenuation
        start = self._config['power_attn.min']
        end = self._config['power_attn.max']
        step = self._power_attn_n_cls = self._config['power_attn.num_level']
        self._power_attn_boundaries = torch.linspace(start, end, step).to(self._device)
        # set up models
        model_params = {k[6:]: self._config[k] for k in self._config.keys() if k.startswith('model.')}
        self.neuron_layer = model_params["d_model"]
        self._actor = Actor(num_freq_ch=self._num_freq_ch, power_attn_num_level=self._power_attn_n_cls,
                            model_params=model_params, device=device)
        self._critic = Critic(num_freq_ch=self._num_freq_ch, power_attn_num_level=self._power_attn_n_cls,
                              model_params=model_params, device=device)
        # dataset
        self._network_type = self._config['train.network_type'].lower()
        if self._network_type not in ['iab', 'trunk']:
            raise ValueError("config.yaml의 'train.network_type'은 'IAB' 또는 'Trunk'여야 합니다.")
        train_data_path = self._config['train.dataset_path']
        self._train_graph_dataset = DynamicDataset(
            data_path=train_data_path,
            network_type=self._network_type
        )
        self._train_graph_batch_size = self._config['train.graph_batch_size']
        self._train_graph_dataloader = DataLoader(
            self._train_graph_dataset,
            batch_size=self._train_graph_batch_size,
            shuffle=True
        )

        # --- 평가 데이터셋 로딩 ---
        eval_data_path = self._config['eval.dataset_path']
        self._eval_graph_dataset = DynamicDataset(
            data_path=eval_data_path,
            network_type=self._network_type
        )
        self._eval_batch_size = self._config['eval.eval_batch_size']
        self._eval_graph_dataloader = DataLoader(
            self._eval_graph_dataset,
            batch_size=self._eval_batch_size,
            shuffle=False
        )
        
        # train parameters
        self._num_graph_repeat = self._config['train.num_graph_repeat']
        self._cir_thresh = self._config['train.cir_thresh']
        self._gamma = self._config['train.gamma']
        self._lambda = self._config['train.lambda']
        self._buffer_batch_size = self._config['train.buffer_batch_size']
        self._actor_lr = self._config['train.actor_lr']
        self._critic_lr = self._config['train.critic_lr']
        self._clip_max_norm = self._config['train.clip_max_norm']
        self._entropy_loss_weight = self._config['train.entropy_loss_weight']
        self._num_train_iter = self._config['train.num_train_iter']
        self._PPO_clip = torch.Tensor([self._config['train.PPO_clip']]).to(torch.float).to(self._device)
        self._act_prob_ratio_exponent_clip = self._config['train.act_prob_ratio_exponent_clip']
        self._eval_period = self._config['train.eval_period']  
        
        self._num_graph_sample = self._config['train.graph_sample']      
        self._graph_color_map = self._config['eval.graph_color_map']
        self._use_graph = self._config['eval.use_graph']


    def quantize_power_attn(self, g_batch):
        g_list = g_batch.to_data_list()
        g2_list = []
        for g in g_list:
            # convert node power attenuation to one hot form
            node_power_attn = g.get_tensor('x').to(self._device)
            node_power_attn = torch.bucketize(node_power_attn, self._power_attn_boundaries, right=True) - 1
            node_power_attn[node_power_attn == -1] = 0
            node_power_attn = F.one_hot(node_power_attn, num_classes=self._power_attn_n_cls).to(torch.float32)
            # convert edge power attenuation to one hot form
            edge_power_attn = g.get_tensor('edge_attr').to(self._device)
            edge_power_attn = torch.bucketize(edge_power_attn, self._power_attn_boundaries, right=True) - 1
            valid_edge_idx = edge_power_attn >= 0
            edge_power_attn = edge_power_attn[valid_edge_idx]
            edge_power_attn = F.one_hot(edge_power_attn, num_classes=self._power_attn_n_cls).to(torch.float32)
            edge_index = g.edge_index.to(self._device)
            edge_index = edge_index[:, valid_edge_idx]
            
            # Add IAB properties
            net_map = g.net_map.to(self._device)
            node_channel_req = g.node_channel_req.to(self._device)
            # make a new graph
            g2 = Data(x=node_power_attn, edge_index=edge_index, edge_attr=edge_power_attn, net_map=net_map,
                      node_channel_req = node_channel_req)
            g2_list.append(g2)
        g2_batch = Batch.from_data_list(g2_list)
        return g2_batch
    
    def random_freq_alloc(self, g, alloc_ratio=1.0):
        num_node = g.num_nodes
        freq_alloc = torch.randint(low=0, high=self._num_freq_ch, size=(num_node,), device=self._device)
        freq_alloc_one_hot = F.one_hot(freq_alloc, num_classes=self._num_freq_ch).to(torch.float32)
        mask = torch.rand((num_node,), device=self._device) <= alloc_ratio
        freq_alloc[~mask] = -1
        freq_alloc_one_hot[~mask, :] = 0
        return freq_alloc, freq_alloc_one_hot

    def cal_cir(self, g, freq_alloc):
        # get tx and rx power
        num_batch_node = freq_alloc.shape[0]
        freq_alloc.to(self._device)
        node_power_attn = g.x[:, None].to(self._device)
        
        tx_power = g.node_tx_power[:, None].to(self._device)
        tx_power = tx_power + 10 * torch.log10(freq_alloc)
        rx_power = tx_power + node_power_attn
        # get interference
        edge_power_attn = g.edge_attr[:, None].to(self._device)
        num_edge = edge_power_attn.shape[0]
        edge_index = g.edge_index.to(self._device)
        index_j, index_i = edge_index[0, :], edge_index[1, :]
        index_j = torch.broadcast_to(index_j[:, None], size=(num_edge, self._num_freq_ch))
        index_i = torch.broadcast_to(index_i[:, None], size=(num_edge, self._num_freq_ch))
        tx_power_j = torch.gather(input=tx_power, dim=0, index=index_j)
        interf_db = tx_power_j + edge_power_attn
        interf = torch.pow(10, interf_db * 0.1)
        sum_interf = torch.zeros(size=(num_batch_node, self._num_freq_ch), device=self._device)
        sum_interf = torch.scatter_add(input=sum_interf, dim=0, index=index_i, src=interf)
        sum_interf_db = 10 * torch.log10(sum_interf)
        # get cir
        node_freq_unalloc = (freq_alloc < 1)
        rx_power[node_freq_unalloc] = 0.0
        rx_power = torch.sum(rx_power, dim=1)
        sum_interf_db[node_freq_unalloc] = 0.0
        sum_interf_db = torch.sum(sum_interf_db, dim=1)
        cir = rx_power - sum_interf_db
        node_unalloc = torch.all(node_freq_unalloc, dim=1)
        cir[node_unalloc] = -torch.inf
        
        return cir

    def roll_out(self, g):
        g.to(self._device)
        batch, ptr = g.batch, g.ptr
        batch_size = int(ptr.shape[0]) - 1
        num_batch_node = batch.shape[0]
        g2 = self.quantize_power_attn(g)
        freq_alloc = torch.zeros(size=(num_batch_node, self._num_freq_ch)).to(torch.float).to(self._device)
        unallocated_node = torch.full(size=(num_batch_node,), fill_value=True).to(self._device)
        ongoing = torch.full(size=(batch_size,), fill_value=True).to(self._device)
        self._actor.eval()
        self._critic.eval()
        freq_alloc_buf = []  # seq, (batch, node), freq
        r_freq_alloc_buf = []
        action_buf = []  # seq, batch, 2(node, freq)
        act_log_prob_buf = []  # seq, batch
        value_buf = []  # seq, batch
        ongoing_buf = []  # seq, batch
        cir_buf = []  # seq, (batch, node)

        with torch.no_grad():
            while torch.any(ongoing):
                # Actor Critic network
                act_dist = self._actor(freq_alloc=freq_alloc, node_power_attn=g2['x'],
                                    edge_power_attn=g2['edge_attr'], edge_index=g2['edge_index'], ptr=ptr)
                value = self._critic(freq_alloc=freq_alloc, node_power_attn=g2['x'],
                                    edge_power_attn=g2['edge_attr'], edge_index=g2['edge_index'], batch=batch)
                
                action = act_dist.sample()    
                act_log_prob = act_dist.log_prob(action)          
                      
                # record data
                freq_alloc_buf.append(freq_alloc.detach().clone().cpu())
                action_buf.append(action.detach().clone().cpu())
                act_log_prob_buf.append(act_log_prob.detach().clone().cpu())
                value_buf.append(value.detach().clone().cpu())
                ongoing_buf.append(ongoing.detach().clone().cpu())
                # update frequency allocation
                for idx, act in enumerate(action):
                    if ongoing[idx]:
                        node, freq = act[0], act[1]
                        freq_alloc[ptr[idx] + node, freq] = 1.0
                        unallocated_node[ptr[idx] + node] = False
                        unallocated_node = (torch.sum(freq_alloc, dim=1, keepdim=True).squeeze(1)
                                            < g['node_channel_req'])   
                        ongoing[idx] = torch.any(unallocated_node[ptr[idx]: ptr[idx+1]])

                # compute cir and record
                cir = self.cal_cir(g, freq_alloc)
                cir_buf.append(cir.detach().clone().cpu())
        # Update the last Frequency Allocation
        freq_alloc_buf.append(freq_alloc.detach().clone().cpu())
        
        freq_alloc_buf = torch.stack(freq_alloc_buf, dim=0)
        action_buf = torch.stack(action_buf, dim=0)
        act_log_prob_buf = torch.stack(act_log_prob_buf, dim=0)
        value_buf = torch.stack(value_buf, dim=0)
        ongoing_buf = torch.stack(ongoing_buf, dim=0)
        cir_buf = torch.stack(cir_buf, dim=0)
        buf = Buffer(graph = g2, freq_alloc=freq_alloc_buf, 
                     action=action_buf, act_log_prob=act_log_prob_buf, value=value_buf,
                     ongoing=ongoing_buf, cir=cir_buf, device=self._device)
        return buf

    def train(self, use_wandb=False, save_model=True):
        if use_wandb:
            wandb.init(project='spectrum_allocation_trunk', config=self._config)
            wandb.define_metric("train/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("evaluate/*")
            if not save_model:
                wandb.watch((self._actor, self._critic), log="all")
        train_step = 0
        actor_param_dicts = [{"params": [p for n, p in self._actor.named_parameters() if p.requires_grad]}]
        actor_optimizer = torch.optim.Adam(actor_param_dicts, lr=self._actor_lr)
        critic_param_dicts = [{"params": [p for n, p in self._critic.named_parameters() if p.requires_grad]}]
        critic_optimizer = torch.optim.Adam(critic_param_dicts, lr=self._critic_lr)
        for repeat_idx in range(self._num_graph_repeat):
            for graph_idx, g in enumerate(self._train_graph_dataloader):
                buf = self.roll_out(g)

                buf.cal_reward(cir_thresh=self._cir_thresh)
                buf.cal_lambda_return(gamma=self._gamma, lamb=self._lambda)
                buffer_dataloader = get_buffer_dataloader(buf, batch_size=self._buffer_batch_size, shuffle=True)  
                self._actor.train()
                self._critic.train()
                for minibatch_idx, d in enumerate(buffer_dataloader):
                    g = d['graph']
                    freq_alloc = d['freq_alloc']
                    action = d['action']
                    init_act_log_prob = d['act_log_prob']
                    lambda_return = d['return']                    
                    advantage = lambda_return - d['value']

                    # Train actor
                    for it in range(self._num_train_iter):
                        torch.cuda.empty_cache()

                        act_dist = self._actor(freq_alloc=freq_alloc, node_power_attn=g['x'],
                                    edge_power_attn=g['edge_attr'], edge_index=g['edge_index'], ptr=g["ptr"])

                        # Calculate PPO actor loss
                        act_log_prob = act_dist.log_prob(action)
                        act_prob_ratio = torch.exp(torch.clamp(act_log_prob - init_act_log_prob,
                                                               max=self._act_prob_ratio_exponent_clip))
                        
                        actor_loss = torch.where(advantage >= 0,
                                                 torch.minimum(act_prob_ratio, 1 + self._PPO_clip),
                                                 torch.maximum(act_prob_ratio, 1 - self._PPO_clip))
                        actor_loss = -torch.mean(actor_loss * advantage)
                        entropy_loss = -torch.mean(act_dist.entropy())
                        actor_entropy_loss = actor_loss + self._entropy_loss_weight * entropy_loss
                        actor_optimizer.zero_grad()
                        actor_entropy_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._clip_max_norm)
                        actor_optimizer.step()
                     # Train critic
                    value = self._critic(freq_alloc=freq_alloc, node_power_attn=g['x'],
                                    edge_power_attn=g['edge_attr'], edge_index=g['edge_index'], batch=g["batch"])
                    value_loss = nn.MSELoss()(value, lambda_return)
                    critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._critic.parameters(), self._clip_max_norm)
                    critic_optimizer.step()
                    train_step += 1
                    # Logging
                    log = (f"repeat: {repeat_idx+1}/{self._num_graph_repeat}, "
                           f"graph: {graph_idx+1}/{len(self._train_graph_dataloader)}, "
                           f"minibatch: {minibatch_idx+1}/{len(buffer_dataloader)}, "
                           f"actor loss: {actor_loss}, value loss: {value_loss}, "
                           f"entropy loss: {entropy_loss}")
                    print(log)
                    if use_wandb:
                        wandb_log = {"train/step": train_step, "train/actor_loss": actor_loss,
                                     "train/value_loss": value_loss, "train/entropy_loss": entropy_loss}
                        wandb.log(wandb_log)
                # Save memory after each iteration
                torch.cuda.empty_cache()
                if graph_idx % self._eval_period == 0:
                    self.evaluate(use_wandb)
                    if save_model:
                        self.save_model(use_wandb)

    def evaluate(self, use_wandb=False):
        success = 0
        total = 0
        cir = []
        for g in self._eval_graph_dataloader:
            buf = self.roll_out(g)
            buf.cal_reward(cir_thresh=self._cir_thresh)
            c, succ, tot = buf.get_performance(cir_thresh=self._cir_thresh)
            success += succ
            total += tot
            cir.append(c)
            if self._use_graph:
                networkx_plot = self.plot_freq_alloc(g, buf, show_graph = self._use_graph)

        success_ratio = success / total
        print(f"success ratio: {success_ratio}")
        cir = torch.concatenate(cir, dim=0).numpy()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(title='ecdf', xlabel='cir', ylabel='prob')
        ax.ecdf(cir)
        plt.close()

        if use_wandb:
            log = {"evaluate/success_ratio": success_ratio}
            wandb.log(log)
            wandb.log({'cir': wandb.Image(fig)})
            plt.close()
            #plt.close("all") # If this doesn't work, use this one
            if self._use_graph:
                # Add NetworkX Plot into the Wandb
                for i in range(len(networkx_plot)):
                    wandb.log({f'Network{i}': wandb.Image(networkx_plot[i])})

    def plot_freq_alloc(self, g, buf, show_graph = False):
        graph_iter = self._num_graph_sample if self._num_graph_sample < len(g) else len(g)
        # Take the fully allocated frequency array
        freq_alloc = buf._final_freq_alloc

        # Generate n random non-repeating numbers between eval_batch_size using torch.randperm
        random_numbers = torch.randperm(len(g))[:graph_iter]
        graph_eval = g.to_data_list()
          
                
        idx = torch.tensor(buf._idx)
        
        networkx_plot = []
        # Loop over the chosen graph sample        
        for i in random_numbers:
            # Do some additional indexing for IAB data
            if g.net_map.numel() != 0:
                tmp_net_map = g.net_map.clone().cpu()
                ptr = g.ptr.clone().cpu()
                old_idx = 0
                for j in range(len(ptr)-1):
                    tmp_net_map[ptr[j]:ptr[j+1],0] += old_idx 
                    tmp_net_map[ptr[j]:ptr[j+1],1] = j
                    old_idx =  tmp_net_map[ptr[j+1]-1,0] + 1
                tmp_net_map = torch.flip(tmp_net_map, [1]) 
                idx = tmp_net_map
                
            batch_indices_for_batch = (idx[:, 0] == i).nonzero(as_tuple=True)[0]
            new_freq_alloc = freq_alloc[batch_indices_for_batch]
            tx = graph_eval[i].node_tx
            rx = graph_eval[i].node_rx
            
            # Generates all nodes available on Trunk / IAB
            nodes = torch.cat((tx,rx))
            nodes = torch.unique(nodes)
            nodes = torch.sort(nodes).values
            pos = graph_eval[i].node_pos[nodes].to('cpu')
            pos = {idx: pos for idx, pos in zip(nodes.tolist(), pos.tolist())}

            # Create NetworkX
            G = nx.DiGraph()
            G.add_nodes_from(nodes.tolist())    # Add the Nodes
            node_labels = {node: str(node) for node in G.nodes()}  # Create labels with node identifiers
            edges = []
            
            # Add the edges
            edges = [(x, y) for x, y in zip(tx.tolist(), rx.tolist())]
            G.add_edges_from(edges)    

            freq_alloc_index = torch.nonzero(new_freq_alloc)[:,1]
            edge_colors = [self._graph_color_map[x.item()] for x in freq_alloc_index]

            color_map = {edge: color for edge, color in zip(edges, edge_colors)}

            fig = plt.figure()
            
            # Draw the NetworkX on the fig
            nx.draw_networkx_nodes(G, pos = pos, node_color="lightblue", node_size=100)
            nx.draw_networkx_edges(G, pos = pos, connectionstyle="arc3,rad=0.1", arrows=True, edge_color=[color_map[edge] for edge in G.edges()])
            nx.draw_networkx_labels(G, pos = pos, labels=node_labels, font_size=6)  # Draw node labels

            # Append into the list
            networkx_plot.append(fig) 
        
        if show_graph:
            plt.show()
        else:
            plt.close()
        return networkx_plot
         
    def save_model(self, use_wandb=False):
        if use_wandb:
            path = Path(wandb.run.dir)
        else:
            path = Path(__file__).parents[0].resolve() / 'saved_model'
        torch.save(self._actor, path / 'actor.pt')
        torch.save(self._critic, path / 'critic.pt')

    def load_model(self):
        path = Path(__file__).parents[0].resolve() / 'saved_model'
        self._actor = torch.load(path / 'actor.pt')
        self._critic = torch.load(path / 'critic.pt')

if __name__ == '__main__':

    device = 'cuda:0'
    en = Engine(params_file='config.yaml', device=device)
    #en.load_model()
    en.train(use_wandb=False, save_model=False)

    




