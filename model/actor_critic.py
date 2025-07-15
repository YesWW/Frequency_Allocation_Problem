import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.init import xavier_uniform_, constant_
from torch.distributions.categorical import Categorical
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.pool import global_mean_pool
from torch_scatter import scatter_mean
class Actor(nn.Module):
    def __init__(self, num_freq_ch, power_attn_num_level, model_params, device):
        super(Actor, self).__init__()
        self._device = device
        self._num_freq_ch = num_freq_ch
        self._power_attn_num_level = power_attn_num_level
        self._d_model = model_params['d_model']
        self._n_head = model_params['n_head']
        self._dim_feedforward = model_params['dim_feedforward']
        self._num_layers = model_params['actor_num_layers']
        self._dropout = model_params['dropout']
        self._graph_transformer = GraphTransformer(input_dim=self._num_freq_ch, embedding_dim=self._power_attn_num_level,
                                                   num_layers=self._num_layers, d_model=self._d_model, n_head=self._n_head,
                                                   edge_dim=self._power_attn_num_level,
                                                   dim_feedforward=self._dim_feedforward, dropout=self._dropout,
                                                   activation="relu", device=self._device)
        
        self._output_linear = Linear(in_features=self._d_model, out_features=self._num_freq_ch, bias=True, device=device)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self._output_linear.weight)
        constant_(self._output_linear.bias, 0.)

    def forward(self, freq_alloc, node_power_attn, edge_power_attn, edge_index, ptr):
        
            
        x = self._graph_transformer(input=freq_alloc, embedding=node_power_attn, edge_attr=edge_power_attn,
                                    edge_index=edge_index, ptr = ptr)

        # Do additional aggregation for IAB connection
        # if net_map is not None:
            
        #     # Identify unique values and their corresponding indices
        #     _, index = torch.unique(net_map[1,:], return_inverse=True)

        #     index = index.view(index.size(0),1)
        #     index =  torch.broadcast_to(index, size=(index.size(0), x.size(1)))
        #     # x_AP = torch.zeros(r_ptr[-1],x.size(1), dtype = x.dtype).to(self._device)
        #     x = scatter_mean(dim=0, index=index, src=x)    
            
        #     # net_map = tmp_net_map
        #     # freq_alloc = tmp_freq_alloc
        #     ptr = r_ptr
        logit = self._output_linear(x)
        # if net_map is not None:
        #     unallocated_node = (torch.sum(r_freq_alloc, dim=1, keepdim=True) < 1.0)    
        # else:
        #     unallocated_node = (torch.sum(freq_alloc, dim=1, keepdim=True) < channel_req.unsqueeze(1))  
        #unallocated_node = (torch.sum(freq_alloc, dim=1, keepdim=True) < channel_req.unsqueeze(1))

        #logit = torch.where(condition = (freq_alloc == 0), input = logit, other = -torch.inf)
        #logit = torch.where(condition=unallocated_node, input=logit, other=-torch.inf)
    
        act_dist = ActDist(logit, ptr, device=self._device)
        return act_dist


class Critic(nn.Module):
    def __init__(self, num_freq_ch, power_attn_num_level, model_params, device):
        super(Critic, self).__init__()
        self._device = device
        self._num_freq_ch = num_freq_ch
        self._power_attn_num_level = power_attn_num_level
        self._d_model = model_params['d_model']
        self._n_head = model_params['n_head']
        self._dim_feedforward = model_params['dim_feedforward']
        self._num_layers = model_params['critic_num_layers']
        self._dropout = model_params['dropout']
        self._graph_transformer = GraphTransformer(input_dim=self._num_freq_ch, embedding_dim=self._power_attn_num_level,
                                                   num_layers=self._num_layers, d_model=self._d_model, n_head=self._n_head,
                                                   edge_dim=self._power_attn_num_level,
                                                   dim_feedforward=self._dim_feedforward, dropout=self._dropout,
                                                   activation="relu", device=self._device)
        self._output_linear = Linear(in_features=self._d_model, out_features=1, bias=True, device=device)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self._output_linear.weight)
        constant_(self._output_linear.bias, 0.)

    def forward(self, freq_alloc, node_power_attn, edge_power_attn, edge_index, batch, ptr = None,
                r_freq_alloc = None, r_embedding = None, r_edge_index = None, r_batch = None, r_ptr = None, AP_ptr = None, net_map = None):
        x = self._graph_transformer(input=freq_alloc, embedding=node_power_attn, edge_attr=edge_power_attn,
                                    edge_index=edge_index, r_input = r_freq_alloc, r_edge_index = r_edge_index,  net_map = net_map,r_embedding = r_embedding,
                                    ptr = ptr, AP_ptr = AP_ptr, r_ptr = r_ptr)
        value = global_mean_pool(x=x, batch=batch)
        #node_value = some(x)
        value = self._output_linear(value)[:, 0]
        return value


class ActDist:
    def __init__(self, logit, ptr,device):
        self._device = device
        self._ptr = ptr
        self._batch_size = int(ptr.shape[0]) - 1
        self._num_freq_ch = logit.shape[1]
        self._dist_list = []
        
        for idx in range(self._batch_size):
           
            l = logit[ptr[idx]: ptr[idx+1], :].to(self._device)
            l = torch.flatten(l)
            if torch.all(torch.isinf(l)):
                dist = None
            else:
                dist = Categorical(logits=l)
            self._dist_list.append(dist)
            
            
    def sample(self):
        action = []
        for dist in self._dist_list:
            if dist is not None:
                idx = int(dist.sample())
                node = idx // self._num_freq_ch
                freq = idx % self._num_freq_ch
            else:
                node, freq = -1, -1
            action.append([node, freq])
        action = torch.Tensor(action).to(torch.int).to(self._device)
        return action

    def entropy(self):
        entropy = []
        for dist in self._dist_list:
            entropy.append(dist.entropy())
        entropy = torch.Tensor(entropy).to(self._device)
        return entropy

    def log_prob(self, action):  # action: (batch, 2(node, freq))
        action = torch.Tensor(action).to(self._device)
        lp = []
        for a, dist in zip(action, self._dist_list):
            if dist is not None:
                node, freq = a[0], a[1]
                idx = node * self._num_freq_ch + freq
                lp.append(dist.log_prob(idx))
            else:
                lp.append(torch.tensor(-torch.inf).to(self._device))
        lp = torch.stack(lp, dim=0)
        return lp


class GraphTransformer(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_layers, d_model, n_head, edge_dim, dim_feedforward, dropout, activation="relu", device='cpu',r_edge_dim=None):
        super(GraphTransformer, self).__init__()
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._num_layers = num_layers
        self._d_model = d_model
        self._n_head = n_head
        self._edge_dim = edge_dim
        self._r_edge_dim = r_edge_dim
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._device = device
        self._activation = activation
        self._input_linear = Linear(in_features=self._input_dim, out_features=self._d_model, bias=True, device=device)
        self._embedding_linear = Linear(in_features=self._embedding_dim, out_features=self._d_model, bias=True, device=device)
        # IAB
        self._r_input_linear = Linear(in_features=self._input_dim, out_features=self._d_model, bias=True, device=device)
        self._r_embedding_linear = Linear(in_features=1, out_features=self._d_model, bias=True, device=device)

       
        self.interf_size = None
        self.relation_size = None
        self._layer_list = nn.ModuleList()
        self._r_layer_list = nn.ModuleList()

        for _ in range(self._num_layers):
            
            layer = GraphTransformerLayer(d_model=self._d_model, n_head=self._n_head,
                                          edge_dim=self._edge_dim,
                                          dim_feedforward=self._dim_feedforward, dropout=self._dropout,
                                          activation=self._activation, device=self._device)
            r_layer = GraphTransformerLayer(d_model=self._d_model, n_head=self._n_head,
                                        edge_dim=None,
                                        dim_feedforward=self._dim_feedforward, dropout=self._dropout,
                                        activation=self._activation, device=self._device)
            
            self._layer_list.append(layer)
            self._r_layer_list.append(r_layer)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self._input_linear.weight)
        xavier_uniform_(self._embedding_linear.weight)
        constant_(self._input_linear.bias, 0.)
        constant_(self._embedding_linear.bias, 0.)

    def forward(self, input, embedding, edge_attr, edge_index, 
                r_input = None, r_edge_index=None, net_map= None, r_embedding= None,
                ptr = None, AP_ptr = None,r_ptr = None):
        input = self._input_linear(input)
        x = self._embedding_linear(embedding)
        # IAB frequency
        if r_edge_index is not None:
            r_embedding = self._r_embedding_linear(r_embedding)
            r_input = self._r_input_linear(r_input)  
            r_attn = torch.zeros(x.size(0)+r_input.size(0),self._d_model).to(self._device)
            
            r_index,_ = torch.unique(net_map[1,:], return_inverse=True)
            x_index = net_map[0,:]
            
            for layer,r_layer in zip(self._layer_list, self._r_layer_list):
                x = x + input
                x = layer(x, edge_attr, edge_index)

                # IAB frequency
                if r_edge_index is not None:
                    r_embedding = r_embedding + r_input 
                    # Concatenation
                    r_attn.index_copy_(0, x_index, x)
                    r_attn.index_copy_(0, r_index, r_embedding)
                    
                    r_attn = r_layer(r_attn, None, r_edge_index)
                    # Spreading the data
                    x = r_attn[x_index]
                    r_embedding = r_attn[r_index]
                    
        else:
            for layer in self._layer_list:
                x = x + input
                x = layer(x, edge_attr, edge_index)
        
        return x


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, edge_dim, dim_feedforward, dropout, activation="relu", device='cpu'):
        super(GraphTransformerLayer, self).__init__()
        self._d_model = d_model
        self._n_head = n_head
        self._edge_dim = edge_dim
        self._dim_feedforward = dim_feedforward
        self._dropout = dropout
        self._device = device
        self._activation = activation
        # Transformer convolution
        out_channel = d_model // n_head
        self._trans_conv = TransformerConv(in_channels=d_model, out_channels=out_channel, heads=n_head,
                                           concat=True, beta=False, dropout=dropout, edge_dim=edge_dim,
                                           bias=True, root_weight=True).to(device)
        # Feedforward neural network
        self.ffnn_linear1 = Linear(in_features=d_model, out_features=dim_feedforward, bias=True, device=device)
        self.ffnn_dropout = Dropout(dropout)
        self.ffnn_linear2 = Linear(in_features=dim_feedforward, out_features=d_model, bias=True, device=device)
        # Layer norm and dropout
        layer_norm_eps = 1e-5
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps).to(device)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        # Activation
        self.activation = self._get_activation_fn(activation)
        # Reset parameters
        self._reset_parameters()

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")

    def _reset_parameters(self):
        xavier_uniform_(self.ffnn_linear1.weight)
        xavier_uniform_(self.ffnn_linear2.weight)
        constant_(self.ffnn_linear1.bias, 0.)
        constant_(self.ffnn_linear2.bias, 0.)
        self._trans_conv.reset_parameters()

    def forward(self, x, edge_attr, edge_index):
        x2 = self._trans_conv(x=x, edge_index=edge_index, edge_attr=edge_attr, return_attention_weights=None)
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        x2 = self.ffnn_linear2(self.ffnn_dropout(self.activation(self.ffnn_linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        return x
