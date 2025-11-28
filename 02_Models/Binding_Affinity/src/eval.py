# path: src/binding_affinity/eval.py
import os, argparse, yaml, numpy as np, torch, time
from torch_geometric.loader import DataLoader

# -----------------------------
# 로컬 메트릭(의존성 최소화)
# -----------------------------
def _metrics(y, yhat):
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    valid = ~np.isnan(y)
    y = y[valid]; yhat = yhat[valid]
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2))) if len(y) else float("nan")
    mae  = float(np.mean(np.abs(y - yhat))) if len(y) else float("nan")
    if len(y) > 1:
        pearson = float(np.corrcoef(y, yhat)[0, 1])
        try:
            from scipy.stats import spearmanr
            spearman = float(spearmanr(y, yhat, nan_policy="omit")[0])
        except Exception:
            spearman = float("nan")
    else:
        pearson = float("nan"); spearman = float("nan")
    return {"rmse": rmse, "mae": mae, "pearson": pearson, "spearman": spearman}

# -----------------------------
# 모델 빌드 (compile 없이)
# -----------------------------
def build_model_no_compile(cfg, in_dim, device):
    m = cfg.get("model", {})
    mtype = m.get("type", "baseline").lower()
    cutoff = float(cfg.get("data", {}).get("cutoff", 6.0))

    if mtype == "egnn":
        from .model.egnn import EGNN
        model = EGNN(
            in_dim=in_dim,
            hidden=int(m.get("hidden", 224)),
            layers=int(m.get("layers", 8)),
            rbf_dim=int(m.get("rbf_dim", 64)),
            cutoff=cutoff,
        ).to(device)
    else:
        from .model.baseline import EdgeGNN
        model = EdgeGNN(
            in_dim=in_dim,
            hidden=int(m.get("hidden", 128)),
            layers=int(m.get("layers", 4)),
        ).to(device)
    return model

# -----------------------------
# 체크포인트 로더
# -----------------------------
def load_model_from_checkpoint(checkpoint_path, cfg, in_dim, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ck_cfg = ckpt.get("cfg", cfg)
    ck_in  = ckpt.get("in_dim", in_dim)
    state  = ckpt.get("state_dict", ckpt)

    model = build_model_no_compile(ck_cfg, ck_in, device)

    # torch.compile 저장물 키 보정 (_orig_mod. 제거)
    if isinstance(state, dict) and any(k.startswith("_orig_mod.") for k in state.keys()):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# -----------------------------
# split 평가
# -----------------------------
def run_split(split, cfg, device):
    # 지연 임포트로 결합도↓
    from .train import CacheDataset

    cache_root = cfg["data"].get("cache_root", "data/processed/binding")
    index_path = cfg["data"].get("pdbbind_index", "data/raw/pdbbind_refined/v2016/index/INDEX_refined_data.2016")
    split_dir = os.path.join(cache_root, split)

    if not os.path.isdir(split_dir):
        print(f"[warn] skip {split}: no dir {split_dir}")
        return

    ds = CacheDataset(split_dir, index_path=index_path)
    if len(ds) == 0:
        print(f"[warn] empty {split}")
        return

    checkpoint_path = "checkpoints/baseline_best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"[error] checkpoint not found: {checkpoint_path}")
        return

    model = load_model_from_checkpoint(checkpoint_path, cfg, None, device)

    batch_size = min(64, int(cfg.get("train", {}).get("batch", 16)) * 2)
    ncpu = os.cpu_count() or 4
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=min(6, ncpu, 8),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )

    y_all, yhat_all = [], []
    print(f"[{split}] Evaluating {len(ds)} samples (bs={batch_size})...")
    t0 = time.time()

    with torch.no_grad():
        for i, b in enumerate(loader):
            b = b.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                pred = model(b)
            yhat_all.append(pred.detach().cpu().numpy())
            y_all.append(b.y.view(-1).detach().cpu().numpy())
            if (i + 1) % 5 == 0 or (i + 1) == len(loader):
                print(f"  {i+1}/{len(loader)} batches", end="\r")

    el = time.time() - t0
    print(f"\n  done in {el:.1f}s, {len(ds)/max(el,1e-6):.1f} samples/s")

    y = np.concatenate(y_all, 0)
    yhat = np.concatenate(yhat_all, 0)
    m = _metrics(y, yhat)

    print(f"[{split}] RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  r={m['pearson']:.3f}  ρ={m['spearman']:.3f}  N={(~np.isnan(y)).sum()}/{len(y)}")
    return m

# -----------------------------
# 모든 split 평가 + 상세/벤치마크
# -----------------------------
def evaluate_all_splits(cfg, device):
    results = {}
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    for split in ("valid", "test"):
        print(f"\nEvaluating {split}...")
        res = run_split(split, cfg, device)
        if res is not None:
            results[split] = res

    if results:
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"{'Split':<8} {'RMSE':<8} {'MAE':<8} {'Pearson':<8} {'Spearman':<8}")
        print("-"*60)
        for s, m in results.items():
            print(f"{s:<8} {m['rmse']:<8.3f} {m['mae']:<8.3f} {m['pearson']:<8.3f} {m['spearman']:<8.3f}")
    else:
        print("[warn] no results")
    return results

def detailed_analysis(cfg, device):
    from .train import CacheDataset
    cache_root = cfg["data"].get("cache_root", "data/processed/binding")
    index_path = cfg["data"].get("pdbbind_index", "data/raw/pdbbind_refined/v2016/index/INDEX_refined_data.2016")
    ckpt = "checkpoints/baseline_best.pt"
    if not os.path.exists(ckpt):
        print(f"[error] checkpoint not found: {ckpt}"); return
    model = load_model_from_checkpoint(ckpt, cfg, None, device)

    for split in ("valid", "test"):
        split_dir = os.path.join(cache_root, split)
        if not os.path.isdir(split_dir): continue
        ds = CacheDataset(split_dir, index_path=index_path)
        if len(ds) == 0: continue
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=min(4, os.cpu_count() or 2),
                            pin_memory=torch.cuda.is_available(), persistent_workers=True)

        all_y, all_pred = [], []
        t0 = time.time()
        with torch.no_grad():
            for i, b in enumerate(loader):
                b = b.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                    pred = model(b)
                all_pred.append(pred.detach().cpu().numpy())
                all_y.append(b.y.view(-1).detach().cpu().numpy())
                if (i + 1) % 5 == 0 or (i + 1) == len(loader):
                    print(f"  {i+1}/{len(loader)} batches", end="\r")
        el = time.time() - t0
        print(f"\n[{split}] analysis in {el:.1f}s")

        y = np.concatenate(all_y, 0); yhat = np.concatenate(all_pred, 0)
        v = ~np.isnan(y)
        if v.sum() == 0: continue
        yv, ph = y[v], yhat[v]
        corr = float(np.corrcoef(yv, ph)[0,1])
        r2 = 1 - float(np.sum((yv - ph)**2) / np.sum((yv - yv.mean())**2))
        print(f"  Actual mean={yv.mean():.2f}±{yv.std():.2f}  Pred mean={ph.mean():.2f}±{ph.std():.2f}  corr={corr:.3f}  R²={r2:.3f}")

def benchmark_inference(cfg, device):
    from .train import CacheDataset
    cache_root = cfg["data"].get("cache_root", "data/processed/binding")
    index_path = cfg["data"].get("pdbbind_index", "data/raw/pdbbind_refined/v2016/index/INDEX_refined_data.2016")
    valid_dir = os.path.join(cache_root, "valid")
    if not os.path.isdir(valid_dir):
        print("[warn] No valid set"); return
    ds = CacheDataset(valid_dir, index_path=index_path)
    if len(ds) < 50:
        print("[warn] small set"); return

    ckpt = "checkpoints/baseline_best.pt"
    if not os.path.exists(ckpt):
        print(f"[error] checkpoint not found: {ckpt}"); return
    model = load_model_from_checkpoint(ckpt, cfg, None, device)

    print("\n" + "="*50); print("Inference Speed Benchmark"); print("="*50)
    for bs in [1, 8, 16, 32, 64]:
        if bs > len(ds): continue
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
        # warmup
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= 3: break
                b = b.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    _ = model(b)
        # measure
        t0 = time.time(); total = 0
        with torch.no_grad():
            for i, b in enumerate(loader):
                if i >= 20: break
                b = b.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    _ = model(b)
                total += len(b.y)
        el = time.time() - t0
        print(f"bs {bs:2d}: {total/max(el,1e-6):6.1f} samples/s")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/binding_affinity.yaml")
    ap.add_argument("--detailed", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"[error] Config file not found: {args.config}"); return
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # optimize_memory는 train.py 안에 있으므로 지연 임포트
    try:
        from .train import optimize_memory
        optimize_memory()
    except Exception:
        pass

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"TF32={torch.backends.cuda.matmul.allow_tf32}, cudnn.benchmark={torch.backends.cudnn.benchmark}")

    results = evaluate_all_splits(cfg, device)
    if args.detailed:   detailed_analysis(cfg, device)
    if args.benchmark:  benchmark_inference(cfg, device)
    return results

if __name__ == "__main__":
    main()
