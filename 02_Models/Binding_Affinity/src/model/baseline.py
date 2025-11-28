import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_scatter import scatter_add, scatter_softmax

class RBF(torch.nn.Module):
    """
    Radial Basis Function for distance encoding.
    """
    def __init__(self, num=32, low=0.0, high=8.0):
        super().__init__()
        centers = torch.linspace(low, high, num)
        gamma = 1.0 / ((high - low) / num)**2
        self.register_buffer('centers', centers)
        self.gamma = gamma
    
    def forward(self, d):
        # Handle multi-dimensional edge_attr by taking the norm
        if d.dim() > 1 and d.size(-1) > 1:
            d = torch.norm(d, dim=-1, keepdim=True)  # [E, k] -> [E, 1]
        elif d.dim() == 1:
            d = d.unsqueeze(-1) # [E] -> [E, 1]
        
        diff = d - self.centers.view(1, -1)  # [E, 1] - [1, 32] = [E, 32]
        return torch.exp(-self.gamma * diff.pow(2))


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.
    """
    def __init__(self, dims, act=nn.SiLU):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if i < len(dims)-2: 
                layers += [act(), nn.Dropout(0.0)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.net(x)


class EdgeGNN(nn.Module):
    """
    Graph Neural Network with a GAT-style Attention mechanism.
    - Input: Data(x, pos, edge_index, edge_attr)
    - Output: y_hat [B]
    """
    def __init__(self, in_dim, hidden=128, layers=4):
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden)
        self.rbf = RBF(num=32, low=0.0, high=8.0)
        
        self.msg_mlps = nn.ModuleList()
        self.upd_mlps = nn.ModuleList()
        self.att_mlps = nn.ModuleList() # 🔥 MLP for calculating attention scores

        for _ in range(layers):
            self.msg_mlps.append(MLP([hidden*2+32, hidden, hidden]))
            self.upd_mlps.append(MLP([hidden*2, hidden, hidden]))
            # Attention MLP: takes (h_src, h_dst, rbf) and outputs a single attention score
            self.att_mlps.append(MLP([hidden*2+32, hidden, 1]))

        self.head = MLP([hidden, hidden, 1])
    
    def forward(self, data):
        x, ei, ea, batch = (
            data.x, 
            data.edge_index, 
            getattr(data, "edge_attr", None), 
            getattr(data, "batch", None)
        )
        
        # Calculate edge distances if not provided
        if ea is None:  
            with torch.no_grad():
                pos = data.pos
                src_pos, dst_pos = pos[ei[0]], pos[ei[1]]
                ea = (src_pos - dst_pos).pow(2).sum(-1, keepdim=True).sqrt()
        
        h = torch.relu(self.enc(x))
        
        # Iterate through GNN layers
        for msg_mlp, upd_mlp, att_mlp in zip(self.msg_mlps, self.upd_mlps, self.att_mlps):
            src, dst = ei[0], ei[1]
            
            # Encode distances using RBF
            rbf = self.rbf(ea)
            
            # Prepare features for message and attention
            edge_feat = torch.cat([h[src], h[dst], rbf], dim=-1)
            
            # 1. 🔥 Calculate attention scores
            att_score = att_mlp(edge_feat)  # [E, 1]
            
            # 2. 🔥 Normalize scores to get attention weights using softmax
            #    This is done per-node, so weights for all incoming edges to a node sum to 1.
            att_weights = scatter_softmax(att_score, dst, dim=0)  # [E, 1]
            
            # 3. Calculate messages
            m = msg_mlp(edge_feat)  # [E, hidden]
            
            # 4. 🔥 Aggregate messages, weighted by attention
            agg = scatter_add(m * att_weights, dst, dim=0, dim_size=h.size(0))  # [N, hidden]
            
            # 5. Update node features with a residual connection
            h = upd_mlp(torch.cat([h, agg], dim=-1)) + h
        
        # Use a batch vector of zeros if not provided (for single graph case)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        
        # Pool node features to get a graph-level representation
        g = global_mean_pool(h, batch)  # [B, hidden]
        
        # Final prediction head
        out = self.head(g)  # [B, 1]
        
        return out.squeeze(-1)  # [B]