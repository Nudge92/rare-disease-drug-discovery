# === src/admet_multi/train.py (Scheduler V4 - 완전 수정 버전) ===
import argparse, os, yaml, random, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import amp
# --- 스케줄러 임포트 ---
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR
)
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GeoLoader 

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)
from scipy.stats import spearmanr

from .datasets import TdcGraphSet, ensure_scaffold_parquets, collate, NUM_RDKIT_FEATS
from .model_gin_multi import GINHead

# ---------------- Utils (변경 없음) ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infer_task_type(task: str, cfg_types: dict | None = None) -> str:
    if cfg_types and task in cfg_types:
        t = cfg_types[task].lower()
        return "cls" if t.startswith("c") else "reg"
    name = task.lower()
    cls_like = ["bbbp", "tox", "clintox", "ames", "herg", "cyp", "hiv", "bace"] 
    return "cls" if any(k in name for k in cls_like) else "reg" 

# ---------------- 🎯 핵심 수정: train_one 함수 ----------------
def train_one(
    task: str, root: str, epochs: int, bs: int, lr: float,
    device: str, amp_enabled: bool, task_type: str,
    hidden_dim: int = 384,
    num_layers: int = 8,
    dropout: float = 0.3,
    patience: int = 10,
    # --- 스케줄러 인자 추가 (기본값 = 비활성화) ---
    scheduler_type: str = 'none',
    warmup_epochs: int = 0
):
    print(f"\n[Train] Task={task} | type={task_type} | device={device}")
    print(f"  Params: hidden={hidden_dim}, layers={num_layers}, dropout={dropout:.3f}, lr={lr:g}, patience={patience}")
    print(f"  Scheduler: type={scheduler_type}, warmup_epochs={warmup_epochs}")
    
    if task_type == "reg":
        print(f"  📈 회귀 태스크: MAE 최소화를 목표로 학습합니다 (TDC 공식 지표)")
    else:
        print(f"  📊 분류 태스크: ROC-AUC 최대화를 목표로 학습합니다")
    
    ds_tr = TdcGraphSet(os.path.join(root, f"{task}_train.parquet"), task_type) 
    ds_va = TdcGraphSet(os.path.join(root, f"{task}_valid.parquet"), task_type) 
    
    ld_tr = GeoLoader(ds_tr, batch_size=bs, shuffle=True,  collate_fn=collate) 
    ld_va = GeoLoader(ds_va, batch_size=bs, shuffle=False, collate_fn=collate) 

    model = GINHead(
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        dropout=dropout, 
        num_tasks=1,
        num_rdkit_feats=NUM_RDKIT_FEATS 
    ).to(device) 
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) 

    # --- 👇 [완전 수정] 스케줄러 정의 로직 ---
    scheduler = None
    
    # 1. 메인 스케줄러 설정
    main_scheduler = None
    if scheduler_type == "cosine":
        main_scheduler = CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr*0.01
        )
        print(f"  [Scheduler] Cosine Annealing: T_max={epochs}, eta_min={lr*0.01:.2e}")
    elif scheduler_type == "plateau":
        mode = 'max' if task_type == 'cls' else 'min'
        main_scheduler = ReduceLROnPlateau(
            opt, mode=mode, patience=10, factor=0.5, verbose=True
        )
        print(f"  [Scheduler] ReduceLROnPlateau: mode={mode}, patience=10, factor=0.5")
    
    # 2. 웜업 스케줄러 적용 (Plateau는 SequentialLR 호환 불가능)
    if warmup_epochs > 0 and scheduler_type != 'plateau':
        print(f"  [Scheduler] Applying Linear Warmup for {warmup_epochs} epochs.")
        warmup_scheduler = LinearLR(
            opt, 
            start_factor=1e-8,
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        if main_scheduler is not None:
            # 웜업 + 메인 스케줄러 조합
            scheduler = SequentialLR(
                opt, 
                schedulers=[warmup_scheduler, main_scheduler], 
                milestones=[warmup_epochs]
            )
            print(f"  [Scheduler] Sequential: Warmup({warmup_epochs}) → {scheduler_type}")
        else:
            # 웜업만 사용
            scheduler = warmup_scheduler
            print(f"  [Scheduler] Warmup only ({warmup_epochs} epochs)")
            
    elif main_scheduler is not None:
        # 웜업 없음 또는 Plateau 스케줄러
        scheduler = main_scheduler
        print(f"  [Scheduler] Main only: {scheduler_type}")
    
    if scheduler is None:
        print(f"  [Scheduler] None (Fixed LR: {lr:.2e})")
    # --- 👆 [수정 완료] ---

    scaler = amp.GradScaler(device="cuda", enabled=amp_enabled) 

    ckpt_dir = Path("models"); ckpt_dir.mkdir(parents=True, exist_ok=True) 
    stats_path = ckpt_dir / f"admet_{task}.stats.json" 

    if task_type == "cls": 
        pos = float(min(max(ds_tr.y.mean().item(), 1e-4), 1 - 1e-4)) 
        pos_weight = torch.tensor([(1 - pos) / pos], device=device) 
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
        norm = denorm = lambda x: x 
    else:
        mu = float(ds_tr.y.mean()); std = float(ds_tr.y.std() + 1e-8) 
        with open(stats_path, "w") as f: 
            json.dump({"mu": mu, "std": std}, f) 
        print(f"[{task}] saved stats → {stats_path}") 

        norm   = lambda y: (y - mu) / std 
        denorm = lambda y: y * std + mu 
        loss_fn = torch.nn.SmoothL1Loss(beta=1.0) 

    best_score = -float('inf') 
    best_epoch = 0
    patience_counter = 0
    
    ckpt_path = ckpt_dir / f"admet_{task}.pt" 

    for ep in range(1, epochs + 1): 
        model.train() 
        run_loss = 0.0
        for g, y in ld_tr: 
            g, y = g.to(device), y.to(device) 
            opt.zero_grad(set_to_none=True) 
            with amp.autocast(device_type="cuda", enabled=amp_enabled): 
                out = model(g).squeeze(-1)  
                tgt = norm(y) if task_type == "reg" else y 
                loss = loss_fn(out, tgt) 
            scaler.scale(loss).backward() 
            scaler.step(opt) 
            scaler.update() 
            run_loss += float(loss) 

        model.eval() 
        ys, ps = [], []
        with torch.no_grad(): 
            for g, y in ld_va: 
                g, y = g.to(device), y.to(device) 
                out = model(g).squeeze(-1) 
                ys.append(y.float().cpu()) 
                if task_type == "cls":
                    ps.append(torch.sigmoid(out).float().cpu()) 
                else:
                    ps.append(out.float().cpu())  
        y = torch.cat(ys).numpy().ravel() 
        p = torch.cat(ps).numpy().ravel() 

        if task_type == "cls": 
            try: roc = roc_auc_score(y, p) 
            except Exception: roc = float("nan")
            try: pr  = average_precision_score(y, p) 
            except Exception: pr = float("nan")
            monitor = roc # 분류는 ROC-AUC 최대화
            msg = f"AUC={roc:.4f} PR-AUC={pr:.4f}" 
        else:
            p_den = denorm(p) 
            mae  = mean_absolute_error(y, p_den) 
            rmse = float(np.sqrt(mean_squared_error(y, p_den))) 
            try:
                r2   = r2_score(y, p_den) 
            except Exception:
                r2 = float("nan")
            try:
                sp   = float(spearmanr(y, p_den)[0]) 
            except Exception:
                sp = float("nan")
            
            monitor = -mae # MAE 최소화 = -MAE 최대화
            msg = f"MAE={mae:.4f} (주요) | RMSE={rmse:.4f} R²={r2:.4f} Spearman={sp:.4f}" 

        # --- 👇 [수정] 스케줄러 스텝 (Early Stopping 직전) ---
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(monitor) # Plateau는 모니터링 지표가 필요
            else:
                scheduler.step() # Cosine, Linear 등은 epoch 기반
            
            # 현재 LR 표시
            current_lr = opt.param_groups[0]['lr']
            lr_msg = f" | LR={current_lr:.2e}"
        else:
            lr_msg = ""
        # --- 👆 [수정 완료] ---

        # Early Stopping 로직
        is_best = monitor > best_score
        
        if is_best:
            best_score = monitor
            best_epoch = ep
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            
            if task_type == "reg":
                actual_mae = -monitor 
                status_msg = f"🎯 New Best MAE: {actual_mae:.4f}!"
            else:
                status_msg = f"🎯 New Best AUC: {monitor:.4f}!"
        else:
            patience_counter += 1
            status_msg = f"(Patience: {patience_counter}/{patience})"

        print(f"[{task}] ep{ep:03d} loss={run_loss/len(ld_tr):.4f} | {msg}{lr_msg} {status_msg}")

        if patience_counter >= patience:
            print(f"[{task}] --- Early stopping triggered at epoch {ep} (best was epoch {best_epoch}) ---")
            break  
    
    if task_type == "reg":
        final_mae = -best_score 
        print(f"[{task}] ✅ 최종 Best MAE: {final_mae:.4f} (from epoch {best_epoch})")
    else:
        print(f"[{task}] ✅ 최종 Best AUC: {best_score:.4f} (from epoch {best_epoch})")
    
    return best_score, str(ckpt_path) 

# ---------------- Main 함수 (변경 없음) ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/admet_multi.yaml") 
    ap.add_argument("--seed", type=int, default=42) 
    ap.add_argument("--epochs", type=int, default=50) 
    ap.add_argument("--batch_size", "--batch", type=int, default=128) 
    ap.add_argument("--lr", type=float, default=5e-4) 
    ap.add_argument("--split", choices=["scaffold", "random"], default="scaffold") 
    ap.add_argument("--val_ratio", type=float, default=0.15) 
    ap.add_argument("--test_ratio", type=float, default=0.15) 
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto") 
    ap.add_argument("--patience", type=int, default=10) 
    
    args = ap.parse_args() 

    set_seed(args.seed) 

    with open(args.config) as f: 
        cfg = yaml.safe_load(f)
    root  = cfg["data"]["root"] 
    tasks = cfg["data"]["tasks"] 
    cfg_types = cfg.get("data", {}).get("task_types", None) 

    if args.device == "auto": 
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device 
    amp_enabled = (device == "cuda") 

    if args.split == "scaffold": 
        for t in tasks: 
            ensure_scaffold_parquets( 
                task=t, root=root, seed=args.seed,
                tr=1.0 - args.val_ratio - args.test_ratio, 
                va=args.val_ratio, te=args.test_ratio, min_cluster=1 
            )

    rows = []
    for t in tasks: 
        ttype = infer_task_type(t, cfg_types) 
        
        # main()에서 호출 시 스케줄러/웜업 인자를 전달하지 않으므로 기본값(none, 0) 사용
        best_score, ckpt = train_one(
            task=t, root=root,
            epochs=args.epochs, bs=args.batch_size, lr=args.lr,
            device=device, amp_enabled=amp_enabled, task_type=ttype,
            patience=args.patience  
        ) 
        rows.append({"task": t, "ckpt": ckpt, "seed": args.seed, "val_score": best_score}) 

    out_tbl = "reports/tables/admet_ckpts.csv" 
    Path(out_tbl).parent.mkdir(parents=True, exist_ok=True) 
    pd.DataFrame(rows).to_csv(out_tbl, index=False) 
    print(f"\n✅ Saved ckpts: {out_tbl}") 

if __name__ == "__main__":
    main()