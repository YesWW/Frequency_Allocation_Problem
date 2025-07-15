import networkx as nx
import numpy as np
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import csv
import dill
from tqdm import tqdm
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import itertools

class Network:
    def __init__(self, G, type, node_pos, net_link_dict=None):
        self._G = G  # Interference graph
        self._type = type  # trunk, IAB
        self._net_link_dict = {n: [n] for n in self._G.nodes} if net_link_dict is None else net_link_dict  # Dictionary mapping network (AP) to links
        self._node_pos = node_pos

    def to_pyg(self):
        if not self._G.nodes:
            return Data()
        node_power_attn = torch.tensor([self._G.nodes[node]['power_attn'] for node in self._G.nodes()], dtype=torch.float)

        node_tx_power = torch.tensor([self._G.nodes[node]['tx_power'] for node in self._G.nodes()], dtype=torch.float)
        edge_index = torch.tensor(list(self._G.edges()), dtype=torch.long).t().contiguous()
        edge_power_attn = torch.tensor([self._G[u][v]['power_attn'] for u, v in self._G.edges()], dtype=torch.float)
        
        node_tx = torch.tensor([self._G.nodes[node]['tx'] for node in self._G.nodes()], dtype=torch.int32)
        node_rx = torch.tensor([self._G.nodes[node]['rx'] for node in self._G.nodes()], dtype=torch.int32)
        
        # 전역 변수 대신, 객체가 가진 self._node_pos 사용
        node_position = torch.tensor(self._node_pos, dtype=torch.float)

        if self._type == "IAB":
            nodes = torch.tensor(list(self._net_link_dict.keys()))
            ul_values = torch.tensor([value['UL'] for value in self._net_link_dict.values() if 'UL' in value], dtype=torch.int32)
            dl_values = torch.tensor([value['DL'] for value in self._net_link_dict.values() if 'DL' in value], dtype=torch.int32)
            net_values = torch.cat((ul_values, dl_values), dim=1)
            if net_values.numel() > 0 and net_values.dim() > 1:
                    net_values, _ = torch.sort(net_values, dim=1)
            net_values, _ = torch.sort(net_values, dim=1)
            sorted_indices = torch.argsort(nodes)
            nodes = torch.searchsorted(nodes[sorted_indices], nodes)
            net_index = torch.arange(net_values.size(0)).unsqueeze(1).expand(-1, net_values.size(1))
            net_index = torch.gather(nodes, 0, net_index.flatten())
            net_data = torch.stack((net_values.flatten(), net_index), dim=1)
            net_map = net_data
        else:
            net_map = torch.tensor([])

        data = Data(x=node_power_attn, node_tx_power=node_tx_power, edge_index=edge_index, edge_attr=edge_power_attn, node_tx=node_tx, 
                    node_rx=node_rx, node_pos=node_position, net_map=net_map)
        return data

    def random_allocation(self, num_freq):
        for n in self._net_link_dict:
            freq = np.random.randint(low=0, high=num_freq)
            freq_alloc = np.full(shape=(num_freq,), fill_value=False)
            freq_alloc[freq] = True
            links = (self._net_link_dict[n]['UL'] + self._net_link_dict[n]['DL'] if self._type == 'IAB'
                     else self._net_link_dict[n])
            for l in links:
                self._G.nodes[l]['freq_alloc'] = freq_alloc

    def cal_cir(self):
        for lt in self._G.nodes:
            freq_alloc = self._G.nodes[lt]['freq_alloc']
            tx_power = self._G.nodes[lt]['tx_power']
            power_attn = self._G.nodes[lt]['power_attn']
            rx_power = (tx_power + power_attn) * np.ones(freq_alloc.shape)
            sum_interf = np.zeros(freq_alloc.shape)
            for ls in self._G.predecessors(lt):
                interf_freq_alloc = self._G.nodes[ls]['freq_alloc']
                interf_tx_power = self._G.nodes[ls]['tx_power']
                interf_power_attn = self._G.edges[ls, lt]['power_attn']
                interf_rx_power = interf_tx_power + interf_power_attn
                sum_interf[interf_freq_alloc] += np.power(10, interf_rx_power * 0.1)
            tmp = np.full(freq_alloc.shape, -np.inf)
            np.log10(sum_interf, out=tmp, where=sum_interf > 0.0)
            sum_interf = 10 * tmp
            cir = rx_power[freq_alloc] - sum_interf[freq_alloc]
            self._G.nodes[lt]['cir'] = cir

    def draw_cir_ecdf(self):
        cir = []
        for l in self._G.nodes:
            cir.extend(self._G.nodes[l]['cir'])
        cir = np.array(cir)
        plt.ecdf(cir)
        plt.show()


class InterfGraphDataset(InMemoryDataset):
    def __init__(self, file_name):
        super().__init__()
        self._file_name = Path(__file__).parents[0] / 'network' / file_name
        self.load(str(self._file_name))

class DynamicDataset(InMemoryDataset):
    def __init__(self, file_name, network_type):
        """
        네트워크 타입에 따라 동적 또는 정적 데이터셋을 로드합니다.

        Args:
            data_path (str): 불러올 .pt 파일의 전체 경로.
            network_type (str): 'iab' 또는 'trunk'.
        """
        super().__init__()
        
        file_path = Path(__file__).parents[0] / 'network' / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {file_path}")
            
        # 1. torch.load로 데이터 파일 불러오기
        loaded_data = torch.load(file_path)
        
        # 2. 네트워크 타입에 따라 데이터 구조 처리
        if network_type == 'iab':
            # IAB 데이터는 리스트의 리스트이므로, 단일 리스트로 펼칩니다.
            self.graph_data = list(itertools.chain.from_iterable(loaded_data))
            print(f"--- IAB 데이터 로딩 완료. 총 타임스텝 수: {len(self.graph_data)} ---")
        elif network_type == 'trunk':
            # Trunk 데이터는 이미 단일 리스트 형태입니다.
            self.graph_data = loaded_data
            print(f"--- Trunk 데이터 로딩 완료. 총 그래프 수: {len(self.graph_data)} ---")
        else:
            raise ValueError("network_type은 'iab' 또는 'trunk'여야 합니다.")

    def __len__(self):
        return len(self.graph_data)

    def __getitem__(self, idx):
        return self.graph_data[idx]
    

class NetworkGenerator:
    def __init__(self, params_file='config.yaml', parabolic_gain_file='parabolic_gain.csv'):
        self._config = {}
        conf_dir = Path(__file__).parents[0]
        with open(conf_dir / params_file, 'r') as f:
            self._config = yaml.safe_load(f)
        with open(conf_dir / parabolic_gain_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            self._parabolic_gain = []
            for d in reader:
                self._parabolic_gain.append(np.array([eval(d[0]), eval(d[1])]))
            self._parabolic_gain = np.stack(self._parabolic_gain, axis=0)
        self._omni_gain = 4.91

        # Variable for Node Position
        self.node_pos = []

    def generate_network(self):
        tG = self.generate_topology()
        node_pos = self.get_node_pos(tG)
        iGt, iGa, net_link_dict = self.get_interference_graph(tG)
        
        # Network 생성자에 node_pos 인자 전달
        nt = Network(iGt, type='trunk', node_pos=node_pos, net_link_dict=None)
        na = Network(iGa, type='IAB', node_pos=node_pos, net_link_dict=net_link_dict)
        return nt, na

    def generate_topology(self):
        G = nx.Graph()
        self._deploy_base_node(G)
        self._deploy_combat_unit(G)
        return G

    def _deploy_base_node(self, G):
        top_unit_level = self._config['top_unit_level']
        region = np.array(self._config['AOR'][top_unit_level])
        region = region * np.array([1, 0.8])
        base_dist_min = self._config['node_dist_min'][top_unit_level]
        base_dist_max = self._config['node_dist_max']
        base_node_pos = np.ndarray([0, 2])
        pos_candidate = np.random.rand(self._config['node_base_trial_cnt'], 2) * region
        for pos in pos_candidate:
            dist = np.linalg.norm(base_node_pos - pos, axis=1)
            if dist.shape[0] == 0 or (base_dist_min < np.min(dist) < base_dist_max):
                node_idx = base_node_pos.shape[0]
                G.add_node(node_idx, pos=pos, level=0, n_base_link=0, n_child_link=0)
                base_node_pos = np.concatenate((base_node_pos, pos[np.newaxis, :]), axis=0)
        G.graph['base_node_pos'] = base_node_pos
        for n1, d1 in G.nodes.data():
            for n2, d2 in G.nodes.data():
                dist = np.linalg.norm(d1['pos'] - d2['pos'])
                if (d1['n_base_link'] < self._config['max_base_link']
                        and d2['n_base_link'] < self._config['max_base_link']
                        and n1 < n2 and dist <= base_dist_max):
                    G.add_edge(n1, n2, type='trunk', dist=dist)
                    d1['n_base_link'] += 1
                    d2['n_base_link'] += 1

    def _deploy_combat_unit(self, G):
        top_unit_level = self._config['top_unit_level']
        outer_AOR = np.array(self._config['AOR'][top_unit_level])
        pending_unit_list = []
        # Deploy initial unit (level 1 unit, usually Corps)
        node_idx = G.number_of_nodes()
        level = top_unit_level
        max_comm_dist = self._config['comm_distance']['trunk']
        max_sub_unit = self._config['max_sub_unit'][level]
        comm_ability = self._config['comm_ability'][level]
        comm_node_idx = None # Base unit connection
        pos_low = pos_high = np.array(self._config['AOR'][level]) * np.array([0.5, 0.1])
        # Initial unit has no parent, sibling_index is 0
        self._deploy_single_combat_unit(G, node_idx, level, max_comm_dist, max_sub_unit, comm_ability, pos_low, pos_high,
                                        comm_node_idx, parent_idx=None, sibling_index=0) 
        pending_unit_list.append(node_idx)

        # Deploy units recursively
        while pending_unit_list:
            parent_idx = pending_unit_list[0]
            parent_level = G.nodes[parent_idx]['level']
            parent_max_sub_unit = G.nodes[parent_idx]['max_sub_unit']
            parent_n_sub_unit = G.nodes[parent_idx]['n_sub_unit'] # Number of sub-units already deployed for this parent
            
            if parent_level >= self._config['bottom_unit_level'] or parent_n_sub_unit >= parent_max_sub_unit:
                pending_unit_list.pop(0)
                continue
            
            node_idx += 1
            level = parent_level + 1 # Child unit level
            
            max_comm_dist = (self._config['comm_distance']['trunk'] * 0.5 if level in [1, 2, 3, 4]
                             else self._config['comm_distance']['IAB'])
            max_sub_unit = self._config['max_sub_unit'][level]
            comm_ability = self._config['comm_ability'][level]
            
            # comm_node_idx for physical connection (to base or parent)
            comm_node_idx = None if level in [1, 2, 3] else parent_idx 

            parent_AOR = np.array(self._config['AOR'][parent_level])
            parent_pos = G.nodes[parent_idx]['pos']
            
            # Determine deploy region ratio based on level and sibling index
            if level in [1, 2, 3]:  # Corps, Division, Brigade
                ratio = self._config['deploy_region']['level_1_3'][parent_n_sub_unit]
            elif level in [4, 5, 6]:  # Battalion, Company, Platoon
                ratio = self._config['deploy_region']['level_4_6'][parent_n_sub_unit]
            else:  # Squad
                ratio = self._config['deploy_region']['level_7']
            
            pos_low = parent_pos + parent_AOR * np.array(ratio['low'])
            pos_high = parent_pos + parent_AOR * np.array(ratio['high'])
            
            pos_low = np.clip(a=pos_low, a_min=np.array([0, 0]), a_max=outer_AOR)
            pos_high = np.clip(a=pos_high, a_min=np.array([0, 0]), a_max=outer_AOR)
            
            # Pass parent_idx and current n_sub_unit (which is the sibling_index)
            self._deploy_single_combat_unit(G, node_idx, level, max_comm_dist, max_sub_unit, comm_ability, pos_low,
                                            pos_high, comm_node_idx, 
                                            parent_idx=parent_idx, sibling_index=parent_n_sub_unit)
            
            G.nodes[parent_idx]['n_sub_unit'] += 1
            pending_unit_list.append(node_idx)

    def _deploy_single_combat_unit(self, G, node_idx, level, max_comm_dist, max_sub_unit, comm_ability,
                                   pos_low, pos_high, comm_node_idx, parent_idx=None, sibling_index=None):
        pos = None
        # Base unit connection logic
        if comm_node_idx is None:
            while pos is None:
                p = np.random.uniform(low=pos_low, high=pos_high)
                base_dist = np.linalg.norm(G.graph['base_node_pos'] - p, axis=1)
                dist = np.min(base_dist)
                if dist <= max_comm_dist:
                    pos = p
                    comm_node_idx = np.argmin(base_dist)
                elif np.all(pos_low == pos_high):
                    raise Exception("Communication link cannot be established.")
        # Combat unit connection logic
        else:
            comm_node_pos = G.nodes[comm_node_idx]['pos']
            while pos is None:
                p = np.random.uniform(low=pos_low, high=pos_high)
                dist = np.linalg.norm(comm_node_pos - p)
                if dist <= max_comm_dist:
                    pos = p
                elif np.all(pos_low == pos_high):
                    raise Exception("Communication link cannot be established.")
        
        # Add node with new 'parent' and 'sibling_index' attributes
        G.add_node(node_idx, pos=pos, level=level, n_sub_unit=0, max_sub_unit=max_sub_unit,
                   n_child_link=0, comm_ability=comm_ability,
                   parent=parent_idx, sibling_index=sibling_index) # New attributes
        
        # Increment child link count for the communication node
        G.nodes[comm_node_idx]['n_child_link'] += 1
        
        # Add edge based on level
        if level in [1, 2, 3, 4]:
            G.add_edge(comm_node_idx, node_idx, type='trunk')
        else:
            G.add_edge(comm_node_idx, node_idx, type='IAB', AP=comm_node_idx)
        
    @staticmethod
    def hata_model(freq, hb, hr, d):
        # COST hata path loss model (rural area)
        # freq: frequency (MHz), hb: height of bs antenna (m), hr: height of mobile antenna (m), d: distance (km)
        d[d <= 0] = 0.000000001
        a = (1.1 * np.log10(freq) - 0.7) * hr - (1.56 * np.log10(freq) - 0.8)  # antenna height correction factor
        cm = 0  # constant offset
        lb = 46.3 + 33.9 * np.log10(freq) - 13.82 * np.log10(hb) - a + (44.9 - 6.55 * np.log10(hb)) * np.log10(d) + cm
        return lb

    @staticmethod
    def free_space_path_loss(freq, d):
        # freq: frequency array (MHz), d: distance (km)
        d[d <= 0] = 0.000000001
        c = 299792458.0  # m/s
        pl = 20 * (np.log10(4*np.pi*(d*1000)) + np.log10(freq*1000000) - np.log10(c))
        return pl

    def parabolic_gain(self, azimuth):
        # gain of parabolic antenna
        # azimuth (radian)
        a = ((azimuth * 180.0 / np.pi) + 180) % 360 - 180
        gain = np.interp(a, xp=self._parabolic_gain[:, 0], fp=self._parabolic_gain[:, 1])
        return gain

    def get_interference_graph(self, nG):  # nG: network graph
        iGt = nx.DiGraph()  # trunk interference graph
        iGa = nx.DiGraph()  # IAB interference graph

        # Node construction
        t_link_idx = 0
        a_link_idx = 0
        net_link_dict = {}
        tx_pos_trunk, rx_pos_trunk, tx_pos_iab, rx_pos_iab = [], [], [], []
        for u, v, a in nG.edges(data=True):
            pos_u, pos_v = nG.nodes[u]['pos'], nG.nodes[v]['pos']
            link_type = a['type']
            if link_type == 'trunk':
                iGt.add_node(t_link_idx, tx=u, rx=v, type=link_type, pair=t_link_idx + 1)
                iGt.add_node(t_link_idx + 1, tx=v, rx=u, type=link_type, pair=t_link_idx)
                t_link_idx += 2
                tx_pos_trunk += [pos_u, pos_v]
                rx_pos_trunk += [pos_v, pos_u]
            if link_type == 'IAB':
                AP = a['AP']
                if AP not in net_link_dict:
                    net_link_dict[AP] = {'UL': [], 'DL': []}
                iGa.add_node(a_link_idx, tx=u, rx=v, AP=AP, type=link_type, pair=a_link_idx + 1)
                direction = 'DL' if u == AP else 'UL'
                net_link_dict[AP][direction].append(a_link_idx)
                iGa.add_node(a_link_idx + 1, tx=v, rx=u, AP=AP, type=link_type, pair=a_link_idx)
                direction = 'DL' if v == AP else 'UL'
                net_link_dict[AP][direction].append(a_link_idx + 1)
                a_link_idx += 2
                tx_pos_iab += [pos_u, pos_v]
                rx_pos_iab += [pos_v, pos_u]
        tx_pos_trunk = np.stack(tx_pos_trunk, axis=0) if tx_pos_trunk else np.empty((0, 2))
        rx_pos_trunk = np.stack(rx_pos_trunk, axis=0) if rx_pos_trunk else np.empty((0, 2))

        tx_pos_iab = np.stack(tx_pos_iab, axis=0) if tx_pos_iab else np.empty((0, 2))
        rx_pos_iab = np.stack(rx_pos_iab, axis=0) if rx_pos_iab else np.empty((0, 2))

        # Compute interference matrix (Trunk)
        vec_trunk = rx_pos_trunk[:, np.newaxis, :] - tx_pos_trunk[np.newaxis, :, :]
        dist_trunk = np.linalg.norm(vec_trunk, axis=2)
        freq_trunk = float(self._config['freq']['trunk'])  # MHz
        pl_trunk = self.free_space_path_loss(freq_trunk, dist_trunk)
        angle = np.arctan2(vec_trunk[:, :, 1], vec_trunk[:, :, 0])
        tx_angle_diff = angle - np.diag(angle)[np.newaxis, :]
        rx_angle_diff = angle - np.diag(angle)[:, np.newaxis]
        tx_gain_trunk = self.parabolic_gain(tx_angle_diff)
        rx_gain_trunk = self.parabolic_gain(rx_angle_diff)
        power_attn_trunk = tx_gain_trunk + rx_gain_trunk - pl_trunk
        power_attn_trunk = np.clip(a=power_attn_trunk, a_min=None, a_max=0)

        # Compute interference matrix (IAB)
        vec_iab = rx_pos_iab[:, np.newaxis, :] - tx_pos_iab[np.newaxis, :, :]
        dist_iab = np.linalg.norm(vec_iab, axis=2)
        freq_iab = float(self._config['freq']['IAB'])  # MHz
        hb = self._config['ant_height']['hb']  # meter
        hr = self._config['ant_height']['hr']  # meter
        pl_iab = self.hata_model(freq_iab, hb, hr, dist_iab)
        tx_gain_iab = self._omni_gain
        rx_gain_iab = self._omni_gain
        power_attn_iab = tx_gain_iab + rx_gain_iab - pl_iab
        power_attn_iab = np.clip(a=power_attn_iab, a_min=None, a_max=0)

        # Link construction (trunk)
        tx_power_trunk = 10 * np.log10(self._config['tx_power']['trunk'])
        for ls in iGt.nodes:  # interference source
            for lt in iGt.nodes:  # interference target
                if ls == lt:
                    iGt.nodes[lt]['power_attn'] = power_attn_trunk[lt, ls]  # dB
                    iGt.nodes[lt]['tx_power'] = tx_power_trunk  # dB
                else:
                    iGt.add_edge(ls, lt, power_attn=power_attn_trunk[lt, ls])

        # Link construction (IAB)
        if not self._config['iab_sync']:
            tx_power_iab = 10 * np.log10(self._config['tx_power']['IAB'])
            for sn in net_link_dict:  # interference source network
                for tn in net_link_dict:  # interference target network
                    if sn == tn:
                        for l in net_link_dict[tn]['UL'] + net_link_dict[tn]['DL']:
                            iGa.nodes[l]['power_attn'] = power_attn_iab[l, l]
                            iGa.nodes[l]['tx_power'] = tx_power_iab  # dB
                    else:
                        # Select the link in source network causing the largest interference
                        attn_dict = {}
                        lt = net_link_dict[tn]['UL'][0]
                        for ls in net_link_dict[sn]['UL'] + net_link_dict[sn]['DL']:
                            attn_dict[ls] = power_attn_iab[lt, ls]
                        ls = max(attn_dict, key=attn_dict.get)  # uplink interference source
                        for lt in net_link_dict[tn]['UL'] + net_link_dict[tn]['DL']:
                            iGa.add_edge(ls, lt, power_attn=power_attn_iab[lt, ls])
        else:
            tx_power_iab = 10 * np.log10(self._config['tx_power']['IAB'])
            for sn in net_link_dict:  # interference source network
                for tn in net_link_dict:  # interference target network
                    if sn == tn:
                        for l in net_link_dict[tn]['UL'] + net_link_dict[tn]['DL']:
                            iGa.nodes[l]['power_attn'] = power_attn_iab[l, l]
                            iGa.nodes[l]['tx_power'] = tx_power_iab  # dB
                    else:
                        # Select the link in source network causing the largest interference
                        attn_dict = {}
                        lt = net_link_dict[tn]['UL'][0]
                        for ls in net_link_dict[sn]['UL']:
                            attn_dict[ls] = power_attn_iab[lt, ls]
                        uls = max(attn_dict, key=attn_dict.get)  # uplink interference source
                        dls = iGa.nodes[uls]['pair']  # downlink interference source
                        for lt in net_link_dict[tn]['UL'] + net_link_dict[tn]['DL']:
                            ls = uls if lt in net_link_dict[tn]['UL'] else dls
                            iGa.add_edge(ls, lt, power_attn=power_attn_iab[lt, ls])

        return iGt, iGa, net_link_dict

    def get_node_pos(self, G):
        node_pos = nx.get_node_attributes(G, 'pos')
        return np.array(list(node_pos.values()))
    
    def generate_dynamic_pyg_sequence(self, time_steps=30, move_dist=1.0):
        initial_topology = self.generate_topology(show_graph=False)
        topology_sequence = [initial_topology]
        
        for _ in range(1, time_steps):
            prev_topology = topology_sequence[-1]
            current_topology = self._move_nodes_and_rebuild_links(prev_topology, move_dist)
            topology_sequence.append(current_topology)
            
        trunk_pyg_list, iab_pyg_list = [], []
        
        for topology_graph in topology_sequence:
            node_pos = self.get_node_pos(topology_graph)
            iGt, iGa, net_link_dict = self.get_interference_graph(topology_graph)
            nt = Network(iGt, type='trunk', node_pos=node_pos)
            trunk_pyg_list.append(nt.to_pyg())
            na = Network(iGa, type='IAB', node_pos=node_pos, net_link_dict=net_link_dict)
            iab_pyg_list.append(na.to_pyg())
            
        return trunk_pyg_list, iab_pyg_list
    
    def generate_dynamic_pyg_sequence(self, time_steps=30, move_dist=1.0):
        # 1. 시뮬레이션 시작 시, 초기 토폴로지 생성
        initial_topology = self.generate_topology()
        
        # 2. 초기 상태에서 Trunk와 IAB 간섭 그래프를 한 번만 계산
        initial_node_pos = self.get_node_pos(initial_topology)
        iGt_initial, iGa_initial, net_link_dict_initial = self.get_interference_graph(initial_topology)

        # 3. 고정된 Trunk 네트워크는 단일 PyG 객체로 변환
        nt_static = Network(iGt_initial, type='trunk', node_pos=initial_node_pos)
        trunk_static_data = nt_static.to_pyg()

        # 4. 시간에 따라 변하는 IAB 네트워크는 리스트로 관리 (초기 상태 먼저 추가)
        na_initial = Network(iGa_initial, 'IAB', initial_node_pos, net_link_dict_initial)
        iab_dynamic_list = [na_initial.to_pyg()]
        
        # 5. 이후 스텝에서는 IAB 노드 이동 및 IAB 간섭 그래프 계산만 반복
        current_topology = initial_topology
        for _ in tqdm(range(1, time_steps), desc="Simulating Movement"):
            # 노드 이동 (링크 구조는 보존)
            current_topology = self._move_nodes(current_topology, move_dist)
            
            # 새 위치에 대한 IAB 간섭 그래프만 다시 계산
            current_node_pos = self.get_node_pos(current_topology)
            _, iGa_current, net_link_dict_current = self.get_interference_graph(current_topology)

            # 새 IAB 데이터를 리스트에 추가
            na_current = Network(iGa_current, 'IAB', current_node_pos, net_link_dict_current)
            iab_dynamic_list.append(na_current.to_pyg())
            
        return trunk_static_data, iab_dynamic_list

    def _move_nodes(self, prev_graph, move_dist):
        new_graph = prev_graph.copy()
        
        # 레벨 순으로 노드를 정렬하여 부모가 자식보다 먼저 이동하도록 보장
        nodes_sorted_by_level = sorted(new_graph.nodes(data=True), key=lambda x: x[1].get('level', 0))

        for node_id, node_data in nodes_sorted_by_level:
            # level 5 이상이고, parent 정보가 있는 노드만 이동 대상
            if node_data.get('level', 0) >= 5 and 'parent' in node_data and node_data['parent'] is not None:
                parent_id = node_data['parent']
                
                # 부모 노드의 '새로운' 위치를 사용 (이미 이동 완료됨)
                parent_pos = np.array(new_graph.nodes[parent_id]['pos'])
                parent_level = new_graph.nodes[parent_id]['level']
                
                child_level = node_data['level']
                sibling_index = node_data.get('sibling_index', 0)

                # Config를 참조하여 이 노드의 작전 영역(AOR) 경계 계산
                ratio = {}
                if child_level in [4, 5, 6]:
                    ratio = self._config['deploy_region']['level_4_6'].get(sibling_index)
                elif child_level == 7:
                    ratio = self._config['deploy_region']['level_7']
                
                if not ratio:
                    continue # AOR 정보가 없으면 이동하지 않음

                parent_AOR_dims = np.array(self._config['AOR'][parent_level])
                pos_low = parent_pos + parent_AOR_dims * np.array(ratio['low'])
                pos_high = parent_pos + parent_AOR_dims * np.array(ratio['high'])
                
                # 1. 자신의 이전 위치에서 약간의 변위(displacement)를 추가
                current_pos = np.array(node_data['pos'])
                displaced_pos_x = current_pos[0] + np.random.uniform(-move_dist, move_dist)
                displaced_pos_y = current_pos[1] + np.random.uniform(-move_dist, move_dist)
                
                # 2. 그 결과가 자신의 AOR 경계를 벗어나지 않도록 조정(clip)
                final_pos_x = np.clip(displaced_pos_x, pos_low[0], pos_high[0])
                final_pos_y = np.clip(displaced_pos_y, pos_low[1], pos_high[1])

                new_graph.nodes[node_id]['pos'] = (final_pos_x, final_pos_y)

        # 링크는 그대로 보존
        return new_graph

    def generate_and_plot_dynamic_sequence(self, time_steps, move_dist, plot_dir="."):
        print("시각화 모드: 초기 네트워크 토폴로지 생성...")
        initial_topology = self.generate_topology()
        topology_sequence = [initial_topology]

        print(f"시각화 모드: {time_steps-1} 타임스텝 동안 시뮬레이션 및 이미지 저장 진행...")
        for _ in tqdm(range(1, time_steps), desc="Plotting Steps"):
            prev_topology = topology_sequence[-1]
            current_topology = self._move_nodes(prev_topology, move_dist)
            topology_sequence.append(current_topology)
        
        # 생성된 모든 스텝을 이미지로 저장
        color_map = self._config.get('color_map', {})
        for t, G_at_t in enumerate(topology_sequence):
            filename = Path(plot_dir) / f"step_{t:03d}.png"
            title = f"Network Topology - Time Step {t}"
            plot_network_and_save(G_at_t, filename, color_map, title)
    
def plot_network_and_save(G, filename, color_map, title="Network Topology"):
        # (이전 코드와 동일한 시각화 헬퍼 함수)
        plt.figure(figsize=(12, 8))
        pos = nx.get_node_attributes(G, 'pos')
        node_colors = [color_map.get(G.nodes[n].get('level', 0), 'grey') for n in G.nodes()]
        trunk_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'trunk']
        iab_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'IAB']
        
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edgelist=trunk_edges, width=1.5, edge_color='red')
        nx.draw_networkx_edges(G, pos, edgelist=iab_edges, width=1.0, edge_color='green', style='dashed')
        
        plt.title(title, fontsize=16)
        plt.xlabel("X Coordinate"), plt.ylabel("Y Coordinate"), plt.grid(True), plt.tight_layout()
        plt.savefig(filename), plt.close()

def save_pyg_data(data, file_name, directory="pyg_data"):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(data, path / file_name)
    print(f"성공: 데이터를 '{path / file_name}' 파일에 저장했습니다.")

def plot_network_topology(G, title="Network Topology"):
    """
    NetworkX 그래프를 입력받아 Trunk와 IAB 링크를 구분하여 시각화합니다.
    """
    plt.figure(figsize=(12, 8))
    
    # 1. 노드 위치 정보 가져오기
    pos = nx.get_node_attributes(G, 'pos')
    
    # 2. 링크(엣지)를 종류별로 분리
    trunk_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'trunk']
    iab_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'IAB']
    
    # 3. 노드 그리기 (레벨별로 다른 색상 지정)
    node_colors = [G.nodes[n].get('level', 0) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.8)
    
    # 4. 링크 그리기
    # Trunk 링크: 굵은 빨간색 실선
    nx.draw_networkx_edges(G, pos, edgelist=trunk_edges, width=1.5, edge_color='red', label='Trunk')
    # IAB 링크: 가는 녹색 점선
    nx.draw_networkx_edges(G, pos, edgelist=iab_edges, width=1.0, edge_color='green', style='dashed', label='IAB')
    
    plt.title(title, fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.axis('on')
    plt.show()


if __name__ == '__main__':
    MODE = 'generate'
    
    try:
        ng = NetworkGenerator()
    except FileNotFoundError:
        print("오류: 'config.yaml' 파일을 찾을 수 없습니다.")
        exit()

    if MODE == 'generate':
       if __name__ == '__main__':
        # === 데이터셋 생성 파라미터 설정 ===
        NUM_TRAIN_SIMULATIONS = 200
        NUM_TEST_SIMULATIONS = 20
        TIME_STEPS_PER_SIMULATION = 20
        MOVE_DISTANCE = 0.3

        # === 저장 경로 설정 ===
        OUTPUT_DIR = "network"

        
        try:
            ng = NetworkGenerator()
        except FileNotFoundError:
            print("오류: 'config.yaml' 파일을 찾을 수 없습니다.")
            exit()

        # --- 1. 학습용(Train) 데이터셋 생성 ---
        print(f"--- 학습용 데이터 생성 시작: 총 {NUM_TRAIN_SIMULATIONS}개의 시뮬레이션 ---")
        
        # 결과를 담을 마스터 리스트 초기화
        train_trunk_master_list = []
        train_iab_master_list = []
        
        for i in tqdm(range(NUM_TRAIN_SIMULATIONS), desc="Generating Train Data"):
            trunk_static_data, iab_dynamic_list = ng.generate_dynamic_pyg_sequence(
                time_steps=TIME_STEPS_PER_SIMULATION,
                move_dist=MOVE_DISTANCE
            )
            # 매번 저장하는 대신, 마스터 리스트에 추가
            train_trunk_master_list.append(trunk_static_data)
            train_iab_master_list.append(iab_dynamic_list)
            
        # 모든 루프가 끝난 후, 마스터 리스트를 단일 파일로 저장
        print("\n--- 학습용 데이터 종합 저장 ---")
        save_pyg_data(train_trunk_master_list, 'train_trunk.pt', directory=OUTPUT_DIR)
        save_pyg_data(train_iab_master_list, 'train_iab.pt', directory=OUTPUT_DIR)

        # --- 2. 테스트용(Test) 데이터셋 생성 ---
        print(f"\n--- 테스트용 데이터 생성 시작: 총 {NUM_TEST_SIMULATIONS}개의 시뮬레이션 ---")
        
        test_trunk_master_list = []
        test_iab_master_list = []
        
        for i in tqdm(range(NUM_TEST_SIMULATIONS), desc="Generating Test Data"):
            trunk_static_data, iab_dynamic_list = ng.generate_dynamic_pyg_sequence(
                time_steps=TIME_STEPS_PER_SIMULATION,
                move_dist=MOVE_DISTANCE
            )
            test_trunk_master_list.append(trunk_static_data)
            test_iab_master_list.append(iab_dynamic_list)
        
        print("\n--- 테스트용 데이터 종합 저장 ---")
        save_pyg_data(test_trunk_master_list, 'test_trunk.pt', directory=OUTPUT_DIR)
        save_pyg_data(test_iab_master_list, 'test_iab.pt', directory=OUTPUT_DIR)
            
        print("\n--- 모든 데이터셋 생성 완료 ---")


    # --- 모드 2: 단일 시뮬레이션 시각화 ---
    elif MODE == 'plot':
        TIME_STEPS_FOR_PLOT = 20
        MOVE_DISTANCE = 0.3
        plot_output_dir = "simulation_plots"
        
        # 플롯 저장 디렉토리 생성
        Path(plot_output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"--- 시각화 모드 실행: 단일 시뮬레이션을 {TIME_STEPS_FOR_PLOT} 스텝 동안 진행하고 이미지로 저장합니다 ---")
        ng.generate_and_plot_dynamic_sequence(
            time_steps=TIME_STEPS_FOR_PLOT,
            move_dist=MOVE_DISTANCE,
            plot_dir=plot_output_dir
        )
        print(f"성공: 모든 스텝의 플롯을 '{plot_output_dir}' 폴더에 저장했습니다.")
        
    else:
        print(f"오류: 잘못된 모드('{MODE}')입니다. 'generate' 또는 'plot'을 선택해주세요.")
