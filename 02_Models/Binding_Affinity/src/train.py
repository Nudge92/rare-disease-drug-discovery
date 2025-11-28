import os
import glob
import argparse
import yaml
import math
import random
import time
import sys
import gc
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ✅ 노트북/터미널 호환성을 위한 tqdm 설정 (끊김 방지)
try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# Mixed Precision (메모리 최적화)
from torch.cuda.amp import GradScaler, autocast

# EGNN 모델 import 경로 설정
# (경로 문제 방지를 위한 이중 체크)
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
try:
    from egnn import EGNN
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src/binding_affinity/model'))
    from egnn import EGNN


def optimize_memory():
    """
    GPU 메모리 캐시를 비우고 가비지 컬렉션을 수행하여 OOM(Out Of Memory)을 방지합니다.
    학습 루프 중간중간 호출해주면 효과적입니다.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =================================================================================
# Dataset Class (기존 로직, 주석, 안전장치 100% 복원)
# =================================================================================
class ImprovedDataset(Dataset):
    """
    개선된 데이터셋 클래스
    - Data 객체에서 직접 라벨(y)을 로딩합니다.
    - 각종 예외 처리 및 차원 검증 로직이 포함되어 있습니다.
    - 데이터 로딩 실패 시 None을 반환하여 Collate Function에서 처리하도록 합니다.
    """
    def __init__(self, cache_dir, index_path=None, max_nodes=3000, augment=False, use_dummy_labels=False):
        self.cache_dir = cache_dir
        # 파일 목록 로드
        self.paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        # 파일명에서 ID 추출 (디버깅용)
        self.ids = [os.path.splitext(os.path.basename(p))[0].upper() for p in self.paths]
        
        self.max_nodes = max_nodes
        self.augment = augment
        self.use_dummy_labels = use_dummy_labels
        
        # 라벨 처리 방식 선택 (더미 라벨 vs 실제 데이터 라벨)
        if use_dummy_labels:
            print("🔧 [Dataset] Using dummy labels for testing/debugging...")
            self.label_map = self._generate_dummy_labels()
            self.use_external_labels = True
        else:
            print("🔧 [Dataset] Using labels directly from Data objects...")
            self.label_map = {}
            self.use_external_labels = False
        
        print(f"📂 Found {len(self.paths)} data files in {cache_dir}")
        
        # 라벨 통계 수집 (정규화 파라미터 계산용)
        self._collect_label_stats()
        
    def _collect_label_stats(self):
        """
        데이터셋의 라벨 분포(평균, 표준편차)를 계산합니다.
        전체 데이터를 다 읽으면 느리므로, 앞부분 1000개만 샘플링하여 추정합니다.
        """
        if self.use_dummy_labels:
            self.label_mean = 7.5
            self.label_std = 1.5
            return
            
        labels = []
        # 속도를 위해 처음 1000개 샘플만 사용하여 통계 계산
        print("🔍 Collecting label statistics (sampling first 1000 files)...")
        for i, path in enumerate(self.paths[:1000]):
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                if hasattr(data, 'y') and data.y is not None:
                    y_val = float(data.y.item())
                    # NaN이나 무한대 값은 제외 (안전장치)
                    if not torch.isnan(torch.tensor(y_val)) and math.isfinite(y_val):
                        labels.append(y_val)
            except Exception:
                continue
        
        if labels:
            self.label_mean = np.mean(labels)
            self.label_std = np.std(labels)
            print(f"🔍 Label Statistics: Mean={self.label_mean:.2f}, Std={self.label_std:.2f}")
            print(f"🔍 Label Range: Min={min(labels):.2f}, Max={max(labels):.2f}")
        else:
            print("⚠️ Warning: Could not collect label stats. Using default values.")
            self.label_mean = 7.0
            self.label_std = 2.0
        
    def _generate_dummy_labels(self):
        """테스트용 더미 라벨 생성 (디버깅용)"""
        import random
        random.seed(42)
        dummy_labels = {}
        for id_name in self.ids:
            dummy_labels[id_name] = random.uniform(5.0, 10.0)
        return dummy_labels
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, i):
        """
        데이터 하나를 로드하고 전처리하는 핵심 함수
        모든 안전장치가 포함되어 있습니다.
        """
        try:
            path = self.paths[i]
            # 안전하게 데이터 로드
            data = torch.load(path, map_location="cpu", weights_only=False)
            cid = self.ids[i]
            
            # -------------------------------------------------------
            # 1. 라벨(Target) 로딩 및 검증
            # -------------------------------------------------------
            if self.use_external_labels:
                y = self.label_map.get(cid, float("nan"))
            else:
                if hasattr(data, 'y') and data.y is not None:
                    try:
                        y_tensor = data.y
                        if isinstance(y_tensor, torch.Tensor):
                            if y_tensor.numel() == 1:
                                y = float(y_tensor.item())
                            elif y_tensor.dim() == 1 and y_tensor.size(0) == 1:
                                y = float(y_tensor[0].item())
                            else:
                                y = float(y_tensor.view(-1)[0].item())
                        else:
                            y = float(y_tensor)
                    except Exception:
                        return None
                else:
                    return None
            
            # 라벨이 NaN이거나 무한대면 데이터 사용 불가 -> Skip
            if torch.isnan(torch.tensor(y)) or not math.isfinite(y):
                return None

            # -------------------------------------------------------
            # 2. 데이터 유효성 검사 (Validation)
            # -------------------------------------------------------
            # 노드 수가 너무 많거나(메모리 초과 위험) 너무 적으면(정보 부족) 스킵
            if data.x.size(0) > self.max_nodes or data.x.size(0) < 5:
                return None

            # 입력 특징(x)에 NaN이나 Inf가 있으면 스킵 (학습 폭발 방지)
            if torch.isnan(data.x).any() or torch.isinf(data.x).any():
                return None

            # -------------------------------------------------------
            # 3. 좌표(Coordinates) 처리
            # -------------------------------------------------------
            if hasattr(data, 'pos') and data.pos is not None:
                coords = data.pos
            else:
                # 좌표가 없으면 랜덤 노이즈로 대체 (비상시 대책)
                coords = torch.randn(data.x.size(0), 3) * 0.1

            if torch.isnan(coords).any() or torch.isinf(coords).any():
                coords = torch.randn(data.x.size(0), 3) * 0.1

            # -------------------------------------------------------
            # 4. 엣지(Edge Index) 처리
            # -------------------------------------------------------
            if hasattr(data, 'edge_index') and data.edge_index.numel() > 0:
                edge_index = data.edge_index
                # 인덱스가 노드 범위를 벗어나는지 체크 (IndexError 방지)
                if edge_index.max() >= data.x.size(0):
                    edge_index = torch.empty(2, 0, dtype=torch.long)
            else:
                edge_index = torch.empty(2, 0, dtype=torch.long)
                        
            # -------------------------------------------------------
            # 5. 특징(Feature) 차원 보정
            # -------------------------------------------------------
            x = data.x.clone()

            # (중요) 입력 차원 고정: 20차원으로 맞춤
            # 원본 데이터가 20차원이 아닐 경우를 대비한 로직
            target_dim = 20
            if x.size(1) != target_dim:
                if x.size(1) < target_dim:
                    # 차원이 부족하면 평균값으로 패딩 (0 padding보다 정보 손실 적음)
                    mean_features = x.mean(dim=0, keepdim=True)
                    padding = mean_features.repeat(x.size(0), target_dim - x.size(1))
                    x = torch.cat([x, padding], dim=1)
                else:
                    # 차원이 넘치면 앞에서부터 자름
                    x = x[:, :target_dim]
            
            # -------------------------------------------------------
            # 6. 특징 정규화 (Feature Standardization)
            # -------------------------------------------------------
            if torch.std(x) > 1e-6: # 표준편차가 0이 아닐 때만 수행
                x_mean = torch.mean(x, dim=0, keepdim=True)
                x_std = torch.std(x, dim=0, keepdim=True) + 1e-6
                x = (x - x_mean) / x_std
                # 이상치 제거를 위한 클리핑 (-3 ~ 3 범위로 제한)
                x = torch.clamp(x, -3, 3)
            
            # -------------------------------------------------------
            # 7. 데이터 증강 (Data Augmentation)
            # -------------------------------------------------------
            if self.augment:
                # 학습 시 약간의 노이즈를 추가하여 일반화 성능 향상
                noise_scale = 0.01
                x = x + torch.randn_like(x) * noise_scale
                coords = coords + torch.randn_like(coords) * noise_scale
            
            # -------------------------------------------------------
            # 8. 라벨 정규화 (Target Normalization)
            # -------------------------------------------------------
            # 예측할 값을 정규분포(0~1) 형태로 변환하여 학습 안정성 확보
            y_normalized = (y - self.label_mean) / (self.label_std + 1e-6)
            
            return {
                'x': x.float(),
                'coords': coords.float(),
                'edge_index': edge_index.long(),
                'y': torch.tensor([y_normalized], dtype=torch.float),
                'y_original': torch.tensor([y], dtype=torch.float), # 평가를 위해 원본 값도 저장
                'num_nodes': x.size(0)
            }
            
        except Exception as e:
            # 데이터 로딩 중 에러 발생 시 None을 반환하여 무시 (학습 중단 방지)
            return None


def improved_collate_fn(batch):
    """
    배치 처리 함수 (Collate Function)
    - 여러 그래프 데이터를 하나의 큰 배치로 병합합니다.
    - None 데이터(로딩 실패)를 필터링하여 안정성을 확보합니다.
    """
    # None 필터링
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    try:
        x_list = []
        coords_list = []
        edge_index_list = []
        y_list = []
        y_original_list = []
        batch_idx_list = []
        
        node_offset = 0
        valid_count = 0
        
        for i, item in enumerate(batch):
            # 최종 안전장치: 텐서에 NaN 값이 있는지 다시 확인
            if torch.isnan(item['y']).any() or torch.isnan(item['x']).any() or torch.isnan(item['coords']).any():
                continue
                
            x_list.append(item['x'])
            coords_list.append(item['coords'])
            
            # 엣지 인덱스 오프셋 조정 (배치 내 노드 인덱스가 겹치지 않도록)
            edge_index = item['edge_index']
            if edge_index.numel() > 0:
                edge_index = edge_index + node_offset
            edge_index_list.append(edge_index)
            
            y_list.append(item['y'])
            y_original_list.append(item['y_original'])
            
            # 배치 인덱스 생성 (어떤 노드가 몇 번째 그래프에 속하는지 표시)
            batch_idx = torch.full((item['num_nodes'],), valid_count, dtype=torch.long)
            batch_idx_list.append(batch_idx)
            
            node_offset += item['num_nodes']
            valid_count += 1
        
        if valid_count == 0:
            return None
        
        # 리스트를 하나의 큰 텐서로 병합 (Concatenation)
        result = {
            'x': torch.cat(x_list, dim=0),
            'coords': torch.cat(coords_list, dim=0),
            'edge_index': torch.cat(edge_index_list, dim=1) if any(e.numel() > 0 for e in edge_index_list) else torch.empty(2, 0, dtype=torch.long),
            'y': torch.cat(y_list, dim=0),
            'y_original': torch.cat(y_original_list, dim=0),
            'batch_idx': torch.cat(batch_idx_list, dim=0)
        }
        
        return result
        
    except Exception as e:
        print(f"⚠️ Batch collation error: {e}")
        return None


class EarlyStopping:
    """
    Early Stopping 클래스
    - 검증 손실(Validation Loss)이 일정 기간(patience) 동안 개선되지 않으면 
    - 과적합(Overfitting)이라 판단하고 학습을 조기 종료합니다.
    """
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.counter = 0
        
    def __call__(self, val_loss):
        if val_loss < self.best_score - self.min_delta:
            # 성능이 개선된 경우
            self.best_score = val_loss
            self.counter = 0
            return False  # 계속 학습
        else:
            # 성능이 개선되지 않은 경우 카운트 증가
            self.counter += 1
            return self.counter >= self.patience  # True면 종료


def set_seed(s):
    """재현성을 위한 난수 시드 고정"""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(s)
        torch.cuda.manual_seed_all(s)


def metrics(y, yhat, label_mean, label_std):
    """
    성능 평가 지표 계산 함수
    - 정규화된 값을 원래 스케일로 복원(Denormalization)하여 계산합니다.
    - RMSE, MAE, Pearson Correlation, Spearman Correlation 등을 반환합니다.
    """
    # 1. 역정규화 (Denormalization)
    y = y * label_std + label_mean
    yhat = yhat * label_std + label_mean
    
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    
    # 2. NaN 제거 마스크 (안전장치)
    mask = ~np.isnan(y) & ~np.isnan(yhat) & np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    
    if len(y) < 2:
        return {"rmse": float("inf"), "mae": float("inf"), "pearson": 0.0, "spearman": 0.0}
    
    # 3. 지표 계산
    rmse = float(np.sqrt(np.mean((yhat - y)**2)))
    mae = float(np.mean(np.abs(yhat - y)))
    
    try:
        # 표준편차가 0이면 상관계수 계산 불가 (예외 처리)
        if np.std(y) == 0 or np.std(yhat) == 0:
            pr = sr = 0.0
        else:
            pr = float(stats.pearsonr(yhat, y)[0])
            sr = float(stats.spearmanr(yhat, y)[0])
            
            if not np.isfinite(pr): pr = 0.0
            if not np.isfinite(sr): sr = 0.0
    except:
        pr = sr = 0.0
        
    return {"rmse": rmse, "mae": mae, "pearson": pr, "spearman": sr}


# =================================================================================
# Main Execution Function (Gradient Accumulation & New Visualization)
# =================================================================================
def main():
    # 1. 인자 파싱
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/binding_affinity_safe.yaml", help="Path to configuration file")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--debug", action="store_true", help="Enable debug mode with smaller dataset")
    ap.add_argument("--debug_size", type=int, default=1000, help="Number of samples in debug mode")
    ap.add_argument("--dummy_labels", action="store_true", help="Use dummy labels for testing")
    ap.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    ap.add_argument("--mixed_precision", action="store_true", help="Use Mixed Precision (FP16)")
    ap.add_argument("--early_stopping", action="store_true", help="Enable Early Stopping")
    ap.add_argument("--patience", type=int, default=7, help="Patience for early stopping")
    ap.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    # 🚀 [NEW] Gradient Accumulation Steps 옵션
    ap.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    args = ap.parse_args()
    
    # 2. 초기 설정
    set_seed(args.seed)
    optimize_memory()

    # 설정 파일이 없으면 기본값 생성
    if not os.path.exists(args.config):
        print(f"⚠️ Config file not found. Creating default at {args.config}")
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        default_config = {
            "data": {
                "cache_root": "data/processed/binding",
                "pdbbind_index": "data/raw/pdbbind_refined/index/INDEX_refined_data.2016"
            },
            "model": {
                "hidden": 128,
                "layers": 4,
                "dropout": 0.0
            },
            "train": {
                "batch_size": 8,
                "lr": 1e-6,
                "epochs": 30,
                "weight_decay": 0,
                "scheduler_type": "plateau"
            }
        }
        with open(args.config, 'w') as f:
            yaml.dump(default_config, f)

    # 설정 로드
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 에포크 오버라이드
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
        print(f"🔧 Overriding epochs to: {args.epochs}")

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # 3. 데이터셋 로드
    cache_root = cfg["data"].get("cache_root", "data/processed/binding")
    train_dir = os.path.join(cache_root, "train")
    valid_dir = os.path.join(cache_root, "valid")
    index_path = cfg["data"].get("pdbbind_index")

    print("🔄 Initializing Datasets...")
    train_ds = ImprovedDataset(
        train_dir, 
        index_path=index_path, 
        max_nodes=3000,
        augment=False,
        use_dummy_labels=args.dummy_labels
    )
    valid_ds = ImprovedDataset(
        valid_dir, 
        index_path=index_path, 
        max_nodes=3000, 
        augment=False,
        use_dummy_labels=args.dummy_labels
    )
    
    print(f"✅ Train dataset size: {len(train_ds)}")
    print(f"✅ Valid dataset size: {len(valid_ds)}")
    
    # 디버그 모드
    if args.debug:
        print(f"🔧 Debug mode enabled. Using {args.debug_size} samples.")
        train_ds.paths = train_ds.paths[:args.debug_size]
        train_ds.ids = train_ds.ids[:args.debug_size]
        valid_ds.paths = valid_ds.paths[:args.debug_size//5]
        valid_ds.ids = valid_ds.ids[:args.debug_size//5]

    # 4. 입력 차원 자동 감지 및 모델 초기화
    sample = None
    for i in range(min(50, len(train_ds))):
        sample = train_ds[i]
        if sample is not None:
            break
            
    if sample is None:
        print("❌ Error: No valid samples found in dataset.")
        return
    
    in_dim = sample['x'].size(-1)
    print(f"ℹ️ Input feature dimension: {in_dim}")

    model = EGNN(
        in_dim=in_dim,
        hidden=cfg["model"]["hidden"],
        layers=cfg["model"]["layers"],
        dropout=cfg["model"]["dropout"]
    ).to(device)
    
    # 가중치 초기화
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.01)

    model.apply(init_weights)
    print(f"🧠 Model initialized. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 5. 학습 설정 (Optimizer, Scheduler)
    batch_size = cfg["train"]["batch_size"]
    initial_lr = 1e-7
    target_lr = float(cfg["train"].get("lr", 1e-3))
    weight_decay = float(cfg["train"].get("weight_decay", 0))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    scheduler_type = cfg["train"].get("scheduler_type", "plateau")
    epochs = cfg["train"]["epochs"]
    print(f"🔧 Scheduler: {scheduler_type} | Epochs: {epochs}")

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3, verbose=True, min_lr=1e-6
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=15, gamma=0.5
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=3
        )
    
    warmup_epochs = 3
    criterion = nn.MSELoss()
    
    # Mixed Precision Scaler
    scaler = None
    if args.mixed_precision and device.type == 'cuda':
        try:
            scaler = GradScaler()
            print("🚀 Mixed Precision Training Enabled")
        except Exception as e:
            print(f"⚠️ Mixed Precision unavailable: {e}")
            args.mixed_precision = False
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=args.patience) if args.early_stopping else None
    if early_stopping:
        print(f"🛑 Early Stopping enabled (Patience: {args.patience})")
    
    # DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=improved_collate_fn, 
        num_workers=2, 
        pin_memory=True,
        persistent_workers=False
    )
    
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=improved_collate_fn, 
        num_workers=2, 
        pin_memory=True,
        persistent_workers=False
    ) if len(valid_ds) > 0 else None
    
    os.makedirs("checkpoints", exist_ok=True)
    best_rmse = float("inf")
    
    # =================================================================================
    # 6. 학습 루프 (Gradient Accumulation + Total Progress Bar)
    # =================================================================================
    accum_steps = args.accum_steps
    effective_batch_size = batch_size * accum_steps
    print(f"🚀 Effective Batch Size: {effective_batch_size} (Physical: {batch_size} x Accum: {accum_steps})")
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    
    # 전체 진행률 바 (Total Progress)
    outer_pbar = tqdm(range(1, epochs + 1), desc="🚀 Total Progress", position=0, leave=True, dynamic_ncols=True)
    
    optimizer.zero_grad() # 루프 시작 전 그라디언트 초기화

    for epoch in outer_pbar:
        epoch_start = time.time()
        model.train()
        
        train_losses = []
        
        # 에포크 내부 배치 진행률 (Epoch Progress)
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", position=1, leave=False, dynamic_ncols=True)
        
        for batch_idx, batch in enumerate(batch_pbar):
            try:
                if batch is None:
                    continue
                
                # Warmup Logic
                if epoch <= warmup_epochs:
                    total_warmup_steps = warmup_epochs * len(train_loader)
                    current_warmup_step = (epoch - 1) * len(train_loader) + batch_idx + 1
                    warmup_factor = current_warmup_step / total_warmup_steps
                    current_lr = initial_lr + (target_lr - initial_lr) * warmup_factor
                    
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                
                # 데이터 이동
                x = batch['x'].to(device, non_blocking=True)
                coords = batch['coords'].to(device, non_blocking=True)
                edge_index = batch['edge_index'].to(device, non_blocking=True)
                y = batch['y'].to(device, non_blocking=True)
                batch_idx_tensor = batch['batch_idx'].to(device, non_blocking=True)
                
                # Forward & Backward (Gradient Accumulation 적용)
                if scaler:
                    with autocast():
                        pred = model(x, coords, edge_index, batch_idx_tensor)
                        loss = criterion(pred, y)
                        # Accumulation: 배치를 쪼개서 loss를 계산하므로 나눔
                        loss = loss / accum_steps
                    
                    scaler.scale(loss).backward()
                    
                    # 지정된 스텝마다 업데이트
                    if (batch_idx + 1) % accum_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    pred = model(x, coords, edge_index, batch_idx_tensor)
                    loss = criterion(pred, y)
                    loss = loss / accum_steps
                    
                    loss.backward()
                    
                    if (batch_idx + 1) % accum_steps == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                
                # 로깅용 Loss는 다시 복원
                loss_val = loss.item() * accum_steps
                train_losses.append(loss_val)
                
                # 배치 바에 실시간 Loss 표시
                batch_pbar.set_postfix({"loss": f"{loss_val:.4f}"})

            except Exception:
                continue

        # 에포크 종료 후 처리
        if not train_losses:
            tqdm.write(f"❌ Epoch {epoch} failed: No valid batches.")
            continue
        
        # 7. 검증 (Validation)
        val_metrics = {"rmse": float("inf"), "mae": float("inf"), "pearson": 0.0, "spearman": 0.0}
        
        if valid_loader is not None:
            model.eval()
            y_list, yhat_list = [], []
            
            with torch.no_grad():
                for batch in valid_loader:
                    try:
                        if batch is None: continue
                            
                        x = batch['x'].to(device, non_blocking=True)
                        coords = batch['coords'].to(device, non_blocking=True)
                        edge_index = batch['edge_index'].to(device, non_blocking=True)
                        y = batch['y'].to(device, non_blocking=True)
                        batch_idx_tensor = batch['batch_idx'].to(device, non_blocking=True)
                        
                        if scaler:
                            with autocast():
                                pred = model(x, coords, edge_index, batch_idx_tensor)
                        else:
                            pred = model(x, coords, edge_index, batch_idx_tensor)
                        
                        if not torch.isnan(pred).any() and not torch.isnan(y).any():
                            y_list.append(y.cpu().numpy())
                            yhat_list.append(pred.cpu().numpy())
                        
                    except Exception as e:
                        continue
            
            if y_list and yhat_list:
                y_val = np.concatenate(y_list)
                yhat_val = np.concatenate(yhat_list)
                val_metrics = metrics(y_val, yhat_val, train_ds.label_mean, train_ds.label_std)

        avg_loss = np.mean(train_losses)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # 🚀 [로그 출력] tqdm.write 사용하여 진행바 깨짐 방지
        log_msg = (f"[Epoch {epoch:03d}/{epochs}] "
                   f"Loss: {avg_loss:.4f} | "
                   f"Val RMSE: {val_metrics['rmse']:.4f} | "
                   f"Pearson: {val_metrics['pearson']:.3f} | "
                   f"LR: {current_lr:.2e} | "
                   f"Time: {epoch_time:.1f}s")
        
        tqdm.write(log_msg)
        
        # 전체 진행바 업데이트
        outer_pbar.set_postfix({
            "Best RMSE": f"{best_rmse:.4f}", 
            "Current RMSE": f"{val_metrics['rmse']:.4f}"
        })
        
        # 스케줄러 Step
        if epoch > warmup_epochs:
            if scheduler_type == "plateau":
                scheduler.step(val_metrics["rmse"])
            else:
                scheduler.step()
        
        # Early Stopping
        if early_stopping and early_stopping(val_metrics["rmse"]):
            tqdm.write(f"🛑 Early stopping triggered at epoch {epoch}")
            break
        
        # Best Model Saving
        if math.isfinite(val_metrics["rmse"]) and val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
                "in_dim": in_dim,
                "config": cfg,
                "epoch": epoch,
                "best_rmse": best_rmse,
                "metrics": val_metrics,
                "label_mean": train_ds.label_mean,
                "label_std": train_ds.label_std
            }
            torch.save(checkpoint, "checkpoints/egnn_best.pt")
            tqdm.write(f"  → Best model saved! (RMSE: {val_metrics['rmse']:.4f})")

        optimize_memory()

    print("\n" + "="*50)
    print(f"🏁 Training Completed!")
    print(f"🏆 Final Best RMSE: {best_rmse:.4f}")
    print(f"💾 Model saved at: checkpoints/egnn_best.pt")
    print("="*50)

if __name__ == "__main__":
    main()