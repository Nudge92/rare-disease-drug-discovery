import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, recall_score,
    accuracy_score, precision_score
)
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader
from .datasets import TdcGraphSet
from .model_gin_multi import GINHead

# -----------------------------
# 보조 함수
# -----------------------------
def calibration_curve(y_true, y_prob, bins=10):
    """간단한 reliability diagram용 calibration curve 계산"""
    bins_edges = np.linspace(0, 1, bins + 1)
    binids = np.digitize(y_prob, bins_edges) - 1
    probs = np.zeros(bins)
    frac_pos = np.zeros(bins)
    for i in range(bins):
        mask = binids == i
        if np.sum(mask) > 0:
            probs[i] = np.mean(y_prob[mask])
            frac_pos[i] = np.mean(y_true[mask])
    return probs, frac_pos

# -----------------------------
# 태스크별 평가
# -----------------------------
# --- [ 💡 핵심 수정 1 ] ---
# Optuna로 튜닝된 하이퍼파라미터를 인자로 받을 수 있도록 추가
def eval_task(
    task: str, 
    root: str, 
    ckpt_path: str, 
    batch: int = 128, 
    device: str = "cuda",
    hidden_dim: int = 384,  # (기본값, Baseline 평가용)
    num_layers: int = 8,    # (기본값, Baseline 평가용)
    dropout: float = 0.3    # (기본값, Baseline 평가용)
):
    """학습된 모델 ckpt를 로드하여 test set 전면 평가"""
    test_path = Path(root) / f"{task}_test.parquet"
    
    # [수정] 대소문자 문제 해결
    task_lower = task.lower()
    is_cls = "herg" in task_lower or "cyp" in task_lower
    task_type = "cls" if is_cls else "reg"
    
    ds_test = TdcGraphSet(str(test_path), task_type=task_type)
    
    # --- [ 수정 ] ---
    # GeoLoader의 collate_fn이 누락되어 있었습니다. 
    # train.py의 collate 함수를 가져오거나, 여기서 간단히 정의해야 하지만,
    # TdcGraphSet이 collate를 내장하고 있지 않다면 DataLoader로 충분합니다.
    # Baseline과 동일하게 GeoLoader를 사용하려면 collate_fn이 필요하지만,
    # 여기서는 표준 DataLoader를 사용하겠습니다. (TdcGraphSet 호환성 가정)
    # *만약 TdcGraphSet이 표준 DataLoader와 호환되지 않으면, 
    # train.py의 collate 함수를 이 파일로 가져와야 합니다.*
    # *업데이트: TdcGraphSet은 Dataset이므로 DataLoader와 호환됩니다.*
    dl_test = DataLoader(ds_test, batch_size=batch, shuffle=False)
    
    # --- [ 💡 핵심 수정 2 ] ---
    # 하드코딩된 값(384, 8) 대신, 함수 인자로 받은 파라미터로 모델을 생성합니다.
    print(f"[{task}] Loading model architecture: hidden={hidden_dim}, layers={num_layers}, dropout={dropout}")
    model = GINHead(
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        dropout=dropout, 
        num_tasks=1
    )
    # --- [ 수정 완료 ] ---
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for g, y in dl_test: # TdcGraphSet이 (g, y) 튜플을 반환한다고 가정
            g, y = g.to(device), y.to(device)
            out = model(g)
            y_true.append(y.cpu())
            y_pred.append(out.cpu())
            
    y_true = torch.cat(y_true).numpy().ravel()
    y_pred = torch.cat(y_pred).numpy().ravel()

    # -----------------------------
    # 분류 / 회귀 분기
    # -----------------------------
    results = {}
    if is_cls:
        # Binary Classification
        y_prob = torch.sigmoid(torch.tensor(y_pred)).numpy()
        y_hat = (y_prob >= 0.5).astype(int)

        results["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        results["PR-AUC"]  = average_precision_score(y_true, y_prob)
        results["F1"]      = f1_score(y_true, y_hat)
        results["Recall@0.5"] = recall_score(y_true, y_hat)
        
        # [ 👈 추가 ]
        try:
            results["Accuracy"] = accuracy_score(y_true, y_hat)
        except Exception:
            results["Accuracy"] = np.nan
        try:
            results["Precision@0.5"] = precision_score(y_true, y_hat)
        except Exception:
            results["Precision@0.5"] = np.nan
        
        # Calibration
        probs, fracs = calibration_curve(y_true, y_prob, bins=10)
        results["ECE"] = np.mean(np.abs(probs - fracs))

    else:
        # Regression
        
        # [ 수정: 역정규화 로직 ]
        mu, std = 0.0, 1.0
        # train.py가 저장한 stats.json 파일 경로 찾기
        stats_file = Path(ckpt_path).parent / f"admet_{task}.stats.json"
        
        if stats_file.exists():
            try:
                with open(stats_file) as f:
                    s = json.load(f)
                mu, std = float(s.get("mu", 0.0)), float(s.get("std", 1.0))
                if std == 0: std = 1.0 # (std가 0으로 저장되는 경우 방지)
                print(f"[{task}] Loaded stats for denorm: mu={mu:.4f}, std={std:.4f}")
            except Exception as e:
                print(f"[{task}] Warning: Could not load/parse {stats_file} ({e})")
        else:
             print(f"[{task}] Warning: stats file not found, using mu=0, std=1. ({stats_file})")
        
        # 예측값 역정규화
        y_pred = y_pred * std + mu
        # [ 수정: 역정규화 로직 끝 ]

        results["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
        results["MAE"]  = mean_absolute_error(y_true, y_pred)
        try:
            results["R^2"]  = r2_score(y_true, y_pred)
        except Exception:
            results["R^2"] = np.nan
        try:
            results["Spearman"] = spearmanr(y_true, y_pred)[0]
        except Exception:
            results["Spearman"] = np.nan

    # -----------------------------
    # 결과 저장
    # -----------------------------
    # [ 💡 수정 3 ]
    # 셀 1의 REPORTS_DIR와 경로 일치 (root = .../data/tdc_admet)
    out_dir = Path(root).parent.parent / "reports" 
    out_dir.mkdir(exist_ok=True, parents=True)
    
    out_csv_path = out_dir / f"{task}_metrics.csv"

    df = pd.DataFrame([results], index=[0]) # index=[0] 추가
    df.to_csv(out_csv_path, index=False)
    print(f"[{task}] metrics:")
    for k, v in results.items():
        print(f"  {k:<12s}: {v:.4f}")
    return results