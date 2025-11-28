import os
import glob
import argparse
import math
import random
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import optuna
from torch.utils.data import Dataset, DataLoader
from contextlib import redirect_stdout

# EGNN 모델 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
try:
    from egnn import EGNN
except ImportError:
    # 실행 위치에 따른 경로 호환성 확보
    sys.path.append(os.path.join(os.getcwd(), 'src/binding_affinity/model'))
    from egnn import EGNN

def optimize_memory():
    """GPU 메모리 정리"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ==============================================================================
# Dataset Class (Scaffold Split 호환 & 안전장치 포함)
# ==============================================================================
class ImprovedDataset(Dataset):
    def __init__(self, cache_dir, max_nodes=3000, subset_ratio=1.0):
        self.paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        
        # 데이터 서브샘플링 (속도 향상용)
        if subset_ratio < 1.0:
            random.seed(42)
            # 튜닝 단계에서는 속도가 우선이므로 일부만 샘플링 (shuffle 후 slice)
            random.shuffle(self.paths) 
            n_samples = int(len(self.paths) * subset_ratio)
            self.paths = self.paths[:n_samples]
            
        self.max_nodes = max_nodes
        self._collect_label_stats()
        
    def _collect_label_stats(self):
        labels = []
        sample_paths = self.paths[:500] if len(self.paths) > 500 else self.paths
        for path in sample_paths:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                if hasattr(data, 'y') and data.y is not None:
                    labels.append(float(data.y.item()))
            except:
                continue
        
        self.label_mean = np.mean(labels) if labels else 7.0
        self.label_std = np.std(labels) if labels else 2.0
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, i):
        try:
            data = torch.load(self.paths[i], map_location="cpu", weights_only=False)
            
            if data.x.size(0) > self.max_nodes or data.x.size(0) < 5: return None
            if torch.isnan(data.x).any(): return None

            # 좌표 정보
            coords = data.pos if hasattr(data, 'pos') else torch.randn(data.x.size(0), 3)
            
            # Edge Index
            edge_index = data.edge_index if hasattr(data, 'edge_index') else torch.empty(2, 0, dtype=torch.long)
            
            # Feature 차원 보정 (20차원 고정)
            x = data.x.clone()
            if x.size(1) != 20:
                if x.size(1) < 20:
                    padding = torch.zeros(x.size(0), 20 - x.size(1))
                    x = torch.cat([x, padding], dim=1)
                else:
                    x = x[:, :20]
            
            # 라벨 정규화
            y_val = float(data.y.item())
            y_normalized = (y_val - self.label_mean) / (self.label_std + 1e-6)
            
            return {
                'x': x.float(), 
                'coords': coords.float(), 
                'edge_index': edge_index.long(),
                'y': torch.tensor([y_normalized], dtype=torch.float),
                'num_nodes': x.size(0)
            }
        except Exception:
            return None

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    
    x_list, coords_list, edge_index_list, y_list, batch_idx_list = [], [], [], [], []
    node_offset = 0
    
    for i, item in enumerate(batch):
        num_nodes = item['num_nodes']
        x_list.append(item['x'])
        coords_list.append(item['coords'])
        
        if item['edge_index'].numel() > 0:
            edge_index_list.append(item['edge_index'] + node_offset)
        
        y_list.append(item['y'])
        batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))
        
        node_offset += num_nodes
        
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.empty(2, 0, dtype=torch.long)
    
    return {
        'x': torch.cat(x_list, dim=0),
        'coords': torch.cat(coords_list, dim=0),
        'edge_index': edge_index,
        'y': torch.cat(y_list, dim=0),
        'batch_idx': torch.cat(batch_idx_list, dim=0)
    }

# ==============================================================================
# Objective Function (Gradient Accumulation 적용)
# ==============================================================================
def objective(trial, train_ds, valid_ds, epochs, accum_steps):
    optimize_memory()
    
    # 1. 하이퍼파라미터 Search Space (메모리 제약 반영)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    # hidden dim은 128을 넘지 않도록 제한
    hidden = trial.suggest_categorical("hidden", [64, 128])
    layers = trial.suggest_int("layers", 3, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.4, step=0.1)
    # Batch size는 작게 유지하되 Accumulation으로 보완
    batch_size = trial.suggest_categorical("batch_size", [8, 16]) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화
    model = EGNN(in_dim=20, hidden=hidden, layers=layers, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # 데이터 로더 (num_workers=2로 로딩 속도 확보)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    best_rmse = float("inf")
    
    # 학습 루프
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        for i, batch in enumerate(train_loader):
            if batch is None: continue
            
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward
                pred = model(batch['x'], batch['coords'], batch['edge_index'], batch['batch_idx'])
                loss = criterion(pred, batch['y'])
                
                # ✅ Gradient Accumulation 핵심 로직
                # Loss를 스텝 수로 나누어 스케일링 (평균 효과)
                loss = loss / accum_steps
                loss.backward()
                
                # 지정된 스텝마다 가중치 업데이트
                if (i + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    optimize_memory()
                    continue # OOM 발생 시 해당 배치 건너뛰기
                else:
                    raise e
            except Exception:
                continue
                
        # 검증 루프
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in valid_loader:
                if batch is None: continue
                try:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    pred = model(batch['x'], batch['coords'], batch['edge_index'], batch['batch_idx'])
                    
                    y_true.append(batch['y'].cpu())
                    y_pred.append(pred.cpu())
                except: continue
        
        if y_true:
            # 역정규화 (Denormalization) 후 RMSE 계산
            yt = torch.cat(y_true).numpy() * train_ds.label_std + train_ds.label_mean
            yp = torch.cat(y_pred).numpy() * train_ds.label_std + train_ds.label_mean
            rmse = np.sqrt(np.mean((yt - yp)**2))
            
            # Optuna Pruning (가망 없는 Trial 조기 종료)
            trial.report(rmse, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            best_rmse = min(best_rmse, rmse)
            
    return best_rmse

# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials")
    parser.add_argument("--study_name", type=str, default="egnn_scaffold_opt")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per trial")
    parser.add_argument("--subset", type=float, default=0.3, help="Data subset ratio for faster tuning")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps to save memory")
    args = parser.parse_args()

    print(f"🚀 튜닝 시작 | Memory Saving Mode (Accumulation: {args.accum_steps})")
    print(f"   Trials: {args.n_trials} | Epochs: {args.epochs} | Subset: {args.subset*100:.0f}%")

    cache_root = "data/processed/binding"
    train_ds = ImprovedDataset(os.path.join(cache_root, "train"), subset_ratio=args.subset)
    valid_ds = ImprovedDataset(os.path.join(cache_root, "valid"), subset_ratio=args.subset)

    # SQLite DB 사용
    storage_url = f"sqlite:///{args.study_name}.db"
    study = optuna.create_study(
        direction="minimize", 
        study_name=args.study_name, 
        storage=storage_url, 
        load_if_exists=True
    )
    
    try:
        # ✅ show_progress_bar=True 옵션 추가
        # 이것이 전체 Trial 진행 상황과 예상 남은 시간을 표시해줍니다.
        study.optimize(
            lambda t: objective(t, train_ds, valid_ds, args.epochs, args.accum_steps), 
            n_trials=args.n_trials,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("🛑 튜닝이 중단되었습니다.")

    print(f"🏆 Best RMSE: {study.best_value:.4f}")
    print("🔧 Best Params:", study.best_params)

if __name__ == "__main__":
    main()