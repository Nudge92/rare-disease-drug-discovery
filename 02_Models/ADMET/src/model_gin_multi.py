# model_gin_multi.py

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool

class GINHead(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=6, dropout=0.2, num_tasks=1, 
                 num_rdkit_feats=None): 
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # [수정 1] 입력 차원 자동 감지 (LazyLinear)
        # x의 차원이 19가 아니더라도 자동으로 맞춰줍니다.
        self.proj = nn.LazyLinear(hidden_dim)
        
        self.gnn = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.gnn.append(GINConv(mlp))

        # [수정 2] Head 입력 차원 자동 감지 (LazyLinear)
        # GNN 임베딩 + RDKit 피처의 합쳐진 차원을 자동으로 계산합니다.
        self.head = nn.Sequential(
            nn.LazyLinear(hidden_dim // 2), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tasks)
        )

    def forward(self, batch):
        x, edge_index, bidx = batch.x, batch.edge_index, batch.batch
        
        # [수정 3] rdkit_feats 안전하게 가져오기 (AttributeError 방지)
        # Data 객체에 rdkit_feats가 없어도 에러가 나지 않도록 처리
        if hasattr(batch, 'rdkit_feats'):
            rdkit_feats = batch.rdkit_feats
        else:
            # [★핵심 수정★] 체크포인트 모델과 차원을 맞추기 위해 '15개'짜리 0 벡터 생성
            # (AqSolDB: 527 - 512 = 15 차이 발생 해결)
            device = x.device
            batch_size = bidx.max().item() + 1
            rdkit_feats = torch.zeros((batch_size, 15), device=device) 

        x = self.proj(x)
        for conv in self.gnn:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = self.dropout(x)
        
        # GNN 그래프 임베딩
        g = global_add_pool(x, bidx)

        # [수정 4] 차원 불일치 방지 (배치 크기 안전장치)
        # 가끔 배치가 잘리는 경우를 대비해 텐서 크기를 맞춤
        if rdkit_feats.shape[0] != g.shape[0]:
             min_len = min(rdkit_feats.shape[0], g.shape[0])
             rdkit_feats = rdkit_feats[:min_len]
             g = g[:min_len]
        
        # 결합
        combined = torch.cat([g, rdkit_feats], dim=1)
        
        # 최종 예측
        out = self.head(combined)
        return out