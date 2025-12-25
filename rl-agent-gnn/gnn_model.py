import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch_geometric.nn import HeteroConv, GATv2Conv, LayerNorm
from torch_geometric.data import HeteroData

class ReplicationGNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # GNN Layer (The "Eye")
        self.gnn = HeteroConv({
            ('key', 'stored_on', 'server'): GATv2Conv(-1, 128, edge_dim=2, add_self_loops=False),
            ('server', 'rev_stored_on', 'key'): GATv2Conv(-1, 128, edge_dim=2, add_self_loops=False),
        }, aggr='mean')

        # Stabilization Layer (CRITICAL FIX for NaNs)
        # Normalizes embeddings to prevent exploding values
        self.layer_norm_key = LayerNorm(128)
        self.layer_norm_server = LayerNorm(128)

        # Action Scorer (The "Judge")
        # Takes [Key_Embed(128) + Server_Embed(128)] -> Score(1)
        self.scorer = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256), # Extra stability
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self._cur_value = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        x_keys = obs['x_keys']
        x_servers = obs['x_servers']
        edge_index = obs['edge_index']
        edge_attr = obs['edge_attr']
        real_counts = obs['real_counts'] 

        batch_size = x_keys.shape[0]
        logits_list = []
        
        # Helper to maintain gradient flow even for empty graphs
        # We take a parameter (like the first weight of the scorer) and multiply by 0
        dummy_grad_hook = self.scorer[0].weight[0,0] * 0.0

        for i in range(batch_size):
            nk, ns, ne = real_counts[i].int().tolist()
            
            if nk == 0 or ns == 0:
                total_slots = x_keys.shape[1] * x_servers.shape[1]
                
                # Create the tensor attached to the graph via dummy_grad_hook
                padded_logits = torch.full((total_slots,), -1e10).to(x_keys.device) + dummy_grad_hook
                
                logits_list.append(padded_logits)
                self._cur_value = torch.tensor(0.0).to(x_keys.device) + dummy_grad_hook
                continue
            
            # Extract Valid Subgraph
            k_feat = x_keys[i, :nk]
            s_feat = x_servers[i, :ns]
            e_idx = edge_index[i, :, :ne].long()
            e_attr = edge_attr[i, :ne]

            data = HeteroData()
            data['key'].x = k_feat
            data['server'].x = s_feat
            data['key', 'stored_on', 'server'].edge_index = e_idx
            data['key', 'stored_on', 'server'].edge_attr = e_attr
            data['server', 'rev_stored_on', 'key'].edge_index = torch.stack([e_idx[1], e_idx[0]])
            data['server', 'rev_stored_on', 'key'].edge_attr = e_attr

            out_dict = self.gnn(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            
            k_emb = self.layer_norm_key(out_dict['key'])
            s_emb = self.layer_norm_server(out_dict['server'])

            # Calculate Global Context (New!)
            # Average all keys to get "System Data State"
            # Average all servers to get "System Hardware State"
            # We need batch indexing for this, but assuming batch_size=1 for inference logic inside loop:
            global_k = torch.mean(k_emb, dim=0, keepdim=True) # [1, 128]
            global_s = torch.mean(s_emb, dim=0, keepdim=True) # [1, 128]

            # Pairwise Scoring with Context
            # Expand Context to match the pairs [nk, ns, 128]
            ctx_k_rep = global_k.unsqueeze(1).expand(nk, ns, -1)
            ctx_s_rep = global_s.unsqueeze(0).expand(nk, ns, -1)

            k_rep = k_emb.unsqueeze(1).expand(-1, ns, -1)
            s_rep = s_emb.unsqueeze(0).expand(nk, -1, -1)
            pairs = torch.cat([k_rep, s_rep, ctx_k_rep, ctx_s_rep], dim=-1)
            
            scores = self.scorer(pairs).view(-1)

            total_slots = x_keys.shape[1] * x_servers.shape[1]
            padded_logits = torch.full((total_slots,), -1e10).to(scores.device)
            valid_len = scores.shape[0]
            padded_logits[:valid_len] = scores

            logits_list.append(padded_logits)
            self._cur_value = torch.mean(scores)

        return torch.stack(logits_list), state

    @override(TorchModelV2)
    def value_function(self):
        return self._cur_value.reshape(-1)