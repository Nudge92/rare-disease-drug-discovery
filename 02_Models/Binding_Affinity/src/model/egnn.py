import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter
import torch.nn.functional as F

class RBF(nn.Module):
    """Radial Basis Function for distance encoding"""
    def __init__(self, num=32, low=0.0, high=8.0):
        super().__init__()
        centers = torch.linspace(low, high, num)
        gamma = 1.0 / ((high - low) / num)**2
        self.register_buffer('centers', centers)
        self.gamma = gamma
        
    def forward(self, d):
        diff = d.unsqueeze(-1) - self.centers.view(1, -1)
        return torch.exp(-self.gamma * diff.pow(2))

class EGNNLayer(nn.Module):
    """Enhanced EGNN Layer with improved stability for large molecules"""
    def __init__(self, x_dim, h_dim, alpha=0.05, dropout=0.1, use_edge_attr=False):
        super().__init__()
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.use_edge_attr = use_edge_attr
        
        # 엣지 입력 차원 계산
        edge_input_dim = 2*x_dim + 1  # h_src + h_dst + distance
        
        # 엣지 메시지 네트워크 (더 깊고 안정적)
        self.phi_e = nn.Sequential(
            nn.Linear(edge_input_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim)
        )
        
        # 좌표 업데이트 게이트 (더 보수적)
        self.phi_x = nn.Sequential(
            nn.Linear(h_dim, h_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(h_dim // 4, 1),
            nn.Tanh()  # 출력 범위 제한
        )
        
        # 노드 특징 업데이트 네트워크
        self.phi_h = nn.Sequential(
            nn.Linear(x_dim + h_dim, h_dim * 2),
            nn.LayerNorm(h_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, x_dim)
        )
        
        # 레이어 정규화
        self.layer_norm_h = nn.LayerNorm(x_dim)
        
    def forward(self, h, pos, edge_index):
        # 빈 에지 처리
        if edge_index.size(1) == 0:
            return h, pos
            
        src, dst = edge_index
        
        # 입력 검증 및 클리핑
        h = torch.clamp(h, -10, 10)
        pos = torch.clamp(pos, -100, 100)
        
        # 상대 위치 벡터 및 거리 계산 (안정화)
        rij = pos[src] - pos[dst]  # [E, 3]
        dij_sq = rij.square().sum(-1, keepdim=True)
        dij_sq = torch.clamp(dij_sq, min=1e-6, max=100.0)  # 거리 제한
        dij = dij_sq.sqrt()  # [E, 1]
        
        # 엣지 메시지 계산
        edge_feat = torch.cat([h[src], h[dst], dij], -1)  # [E, 2*x_dim + 1]
        
        # 메모리 효율을 위한 체크포인팅 (대용량 분자용)
        if edge_feat.size(0) > 50000:  # 50K 이상의 에지일 때
            m_ij = torch.utils.checkpoint.checkpoint(self.phi_e, edge_feat)
        else:
            m_ij = self.phi_e(edge_feat)  # [E, h_dim]
        
        # 그래디언트 클리핑
        m_ij = torch.clamp(m_ij, -5, 5)
        
        # 안정화된 좌표 업데이트
        gate = self.phi_x(m_ij)  # [E, 1] - 이미 tanh 적용됨
        
        # 단위 벡터 계산 (안정화)
        rij_norm = rij / (dij + 1e-8)  # [E, 3]
        coord_msg = gate * rij_norm  # [E, 3]
        
        # 좌표 업데이트 (더 보수적인 스케일)
        delta = scatter(coord_msg, dst, dim=0, dim_size=h.size(0), reduce='mean')
        delta = torch.clamp(delta, -1.0, 1.0)  # 델타 제한
        pos_new = pos + self.alpha * delta
        
        # 노드 특징 업데이트
        m_i = scatter(m_ij, dst, dim=0, dim_size=h.size(0), reduce='mean')
        m_i = torch.clamp(m_i, -5, 5)
        
        h_input = torch.cat([h, m_i], -1)
        h_delta = self.phi_h(h_input)
        h_delta = torch.clamp(h_delta, -2, 2)  # 델타 제한
        
        h_new = h + h_delta  # 잔차 연결
        h_new = self.layer_norm_h(h_new)  # 레이어 정규화
        
        return h_new, pos_new

class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network for molecular property prediction"""
    def __init__(self, in_dim, hidden=128, layers=3, dropout=0.1, pool_type='mean'):
        super().__init__()
        
        self.pool_type = pool_type
        
        # 입력 임베딩 (더 깊게)
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # EGNN 레이어들 (더 보수적인 alpha)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden, hidden, alpha=0.03, dropout=dropout) 
            for _ in range(layers)
        ])
        
        # 그래프 레벨 어텐션 (선택적)
        if pool_type == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, 1)
            )
        
        # =================================================================
        # ✅ 1. Readout Layer 단순화 (모델 붕괴 방지)
        # -----------------------------------------------------------------
        # 기존의 매우 깊은 구조 대신, 학습 신호가 잘 전달될 수 있는
        # 간단하고 안정적인 구조로 변경합니다.
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1)
        )
        # =================================================================
        
        # 가중치 초기화 적용
        self.apply(self._init_weights)

        # =================================================================
        # ✅ 2. 최종 레이어 Bias 0으로 초기화 (모델 붕괴 방지)
        # -----------------------------------------------------------------
        # 모델의 초기 예측값이 0 근처에서 시작하도록 강제하여,
        # 모든 예측을 0으로만 하는 상태에 빠지는 것을 방지합니다.
        final_layer = self.readout[-1]
        if isinstance(final_layer, nn.Linear):
            final_layer.bias.data.fill_(0.0)
        # =================================================================
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier 초기화 (안정적)
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                # bias는 0으로 초기화 (마지막 레이어 제외, 위에서 별도 처리)
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def graph_pooling(self, h, batch_idx):
        """그래프 레벨 풀링"""
        if batch_idx is None:
            # 단일 그래프
            if self.pool_type == 'mean':
                return h.mean(dim=0, keepdim=True)
            elif self.pool_type == 'max':
                return h.max(dim=0, keepdim=True)[0]
            elif self.pool_type == 'attention':
                att_weights = F.softmax(self.attention(h), dim=0)
                return (att_weights * h).sum(dim=0, keepdim=True)
        else:
            # 배치 처리 (scatter 사용으로 최적화)
            if self.pool_type == 'mean':
                return scatter(h, batch_idx, dim=0, reduce='mean')
            elif self.pool_type == 'max':
                return scatter(h, batch_idx, dim=0, reduce='max')
            elif self.pool_type == 'attention':
                att_weights = F.softmax(self.attention(h), dim=0)
                return scatter(att_weights * h, batch_idx, dim=0, reduce='sum')
            else:
                return scatter(h, batch_idx, dim=0, reduce='mean') # 기본값
        
    def forward(self, x, pos, edge_index, batch_idx=None):
        """
        Args:
            x: 노드 특징 [N, in_dim]
            pos: 노드 좌표 [N, 3] 
            edge_index: 에지 인덱스 [2, E]
            batch_idx: 배치 인덱스 [N] (옵션)
        """
        # 입력 검증 및 안전 처리
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: Invalid input features detected")
            batch_size = batch_idx.max().item() + 1 if batch_idx is not None else 1
            return torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        
        if torch.isnan(pos).any() or torch.isinf(pos).any():
            print("Warning: Invalid positions detected")
            batch_size = batch_idx.max().item() + 1 if batch_idx is not None else 1
            return torch.zeros(batch_size, 1, device=x.device, dtype=x.dtype)
        
        # 입력 클리핑
        x = torch.clamp(x, -10, 10)
        pos = torch.clamp(pos, -100, 100)
        
        # 초기 노드 특징 임베딩
        h = self.embed(x)
        
        # EGNN 레이어들 통과
        for i, layer in enumerate(self.layers):
            h_prev, pos_prev = h.clone(), pos.clone()
            h, pos = layer(h, pos, edge_index)
            
            # NaN 체크 및 복구
            if torch.isnan(h).any() or torch.isnan(pos).any():
                print(f"Warning: NaN detected in layer {i}, using previous values")
                h, pos = h_prev, pos_prev
                break
        
        # 그래프 레벨 표현
        graph_repr = self.graph_pooling(h, batch_idx)
        
        # 최종 예측
        output = self.readout(graph_repr)
        
        # 최종 NaN 체크 및 출력 클리핑
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: Invalid output detected")
            # 배치 크기에 맞는 0 텐서 반환
            batch_size = batch_idx.max().item() + 1 if batch_idx is not None else 1
            return torch.zeros(batch_size, 1, device=output.device, dtype=output.dtype)
        
        return output.squeeze(-1)

# 호환성을 위한 별칭
SafeAdvancedEGNN = EGNN