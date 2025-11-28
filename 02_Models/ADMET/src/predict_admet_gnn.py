#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

# 프로젝트 내부 유틸
try:
    from .datasets import mol_to_graph
except ImportError:
    print("[오류] 프로젝트 루트에서 실행하세요. 예: python -m src.admet_multi.predict_admet_gnn ...")
    raise SystemExit(1)


# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles_file", required=True)
    p.add_argument("--ckpt_dir", required=True)
    
    # [추가] 사용자가 지정한 파일명 리스트
    p.add_argument("--ckpt_files", nargs="+", default=None, 
                   help="Specific checkpoint filenames to use (space separated).")
    
    p.add_argument("--out_csv", required=True)
    p.add_argument("--out_md", default=None)

    p.add_argument("--model_type", choices=["custom"], default="custom")
    p.add_argument("--model_module", default="src.admet_multi.model_gin_multi")
    p.add_argument("--model_class", default="GINHead")
    p.add_argument("--model_kwargs", default=None, help="JSON string for constructor kwargs")

    # 기본값 (실제로는 main함수의 tasks 설정이 우선됨)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    p.add_argument("--herg_thr", type=float, default=0.5)
    p.add_argument("--cyp_thr", type=float, default=0.5)

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def choose_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


# ------------------------- I/O -------------------------
def split_smiles_line(line: str):
    line = line.strip()
    if not line:
        return None, None
    if "\t" in line:
        a, b = line.split("\t", 1); return a.strip(), (b.strip() or None)
    if "," in line:
        a, b = line.split(",", 1);  return a.strip(), (b.strip() or None)
    return line, None


def read_smiles(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            smi, iid = split_smiles_line(line)
            if smi is None: continue
            rows.append({"smiles": smi, "id": iid or f"mol_{i}"})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No SMILES found in the input file.")
    return df


# ------------------------- Model Build -------------------------
def build_model(custom_module: str,
                custom_class: str,
                hidden_dim: int,
                num_layers: int,
                dropout: float,
                custom_kwargs: Optional[Dict],
                verbose: bool):
    import importlib
    mod = importlib.import_module(custom_module)
    cls = getattr(mod, custom_class)

    if custom_kwargs is None:
        kw = {"hidden_dim": hidden_dim, "num_layers": num_layers, "dropout": dropout, "num_tasks": 1}
    else:
        kw = custom_kwargs
    if verbose:
        print(f"[model] {custom_module}.{custom_class} kwargs={kw}")
    return cls(**kw)


def find_ckpt(ckpt_dir: Path, patterns: List[str]) -> Optional[Path]:
    """패턴 매칭으로 가장 최신 파일 찾기"""
    cands = []
    for pat in patterns:
        cands += list(ckpt_dir.glob(pat))
        cands += list(ckpt_dir.glob(pat.lower()))
        cands += list(ckpt_dir.glob(pat.upper()))
    cands = sorted(set(cands), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


# ------------------------- Inference -------------------------
@torch.no_grad()
def infer_task(graphs, model, device: str, batch_size: int, is_cls: bool):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    outs = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch).view(-1)
        outs.append(torch.sigmoid(out) if is_cls else out)
    return torch.cat(outs).cpu().numpy()


# ------------------------- Main -------------------------
def main():
    args = parse_args()
    device = choose_device(args.device)
    verbose = args.verbose

    df = read_smiles(Path(args.smiles_file))

    # Graphize
    data_list = []
    for smi in df["smiles"].tolist():
        g = mol_to_graph(smi)
        if g is None:
            # 실패 시 더미 데이터
            g = Data(
                x=torch.zeros((1, 19), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 6), dtype=torch.float32),
            )
        data_list.append(g)

    ckpt_dir = Path(args.ckpt_dir)
    
    # [핵심 수정] 모델별 하이퍼파라미터 하드코딩 (최적화된 모델 기준)
    # hERG: 256/3, CYP3A4: 128/3, AqSolDB: 512/5
    tasks = {
        "hERG":   {
            "cls": True,  "keywords": ["herg"], 
            "config": {"hidden_dim": 256, "num_layers": 3, "dropout": 0.2}
        },
        "CYP3A4": {
            "cls": True,  "keywords": ["cyp3a4", "cyp"], 
            "config": {"hidden_dim": 128, "num_layers": 3, "dropout": 0.2}
        },
        "AqSolDB":{
            "cls": False, "keywords": ["aqsoldb", "solub", "logs"], 
            "config": {"hidden_dim": 512, "num_layers": 5, "dropout": 0.2}
        },
    }
    
    # 자동 검색용 패턴
    auto_patterns = {
        "hERG":   ["admet_herg.pt", "*herg*.pt"],
        "CYP3A4": ["admet_cyp3a4.pt", "*cyp3a4*.pt", "*cyp*.pt"],
        "AqSolDB":["admet_aqsoldb.pt", "*aqsoldb*.pt", "*solub*pt", "*logs*.pt"]
    }

    outputs: Dict[str, np.ndarray] = {}
    
    for name, spec in tasks.items():
        ckpt = None
        
        # 1. 사용자가 지정한 파일(--ckpt_files) 중 키워드 매칭 확인
        if args.ckpt_files:
            for user_f in args.ckpt_files:
                if any(k in user_f.lower() for k in spec["keywords"]):
                    potential_path = ckpt_dir / user_f
                    if potential_path.exists():
                        ckpt = potential_path
                        break
        
        # 2. 없으면 자동 검색
        if ckpt is None:
            ckpt = find_ckpt(ckpt_dir, auto_patterns[name])

        if verbose:
            print(f"[ckpt] {name:7s}: {ckpt if ckpt else '(not found)'}")
        
        if ckpt is None:
            # 파일이 없으면 건너뜀
            continue

        # Build model with specific config
        cfg = spec["config"]
        custom_kwargs = json.loads(args.model_kwargs) if args.model_kwargs else None
        
        if verbose:
            print(f"[{name}] Building model: hidden={cfg['hidden_dim']}, layers={cfg['num_layers']}")
        
        model = build_model(
            custom_module=args.model_module,
            custom_class=args.model_class,
            hidden_dim=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout'],
            custom_kwargs=custom_kwargs,
            verbose=verbose,
        )

        # LazyLinear warm-up
        warm = next((g for g in data_list if g is not None), None)
        if warm is not None:
            # [핵심 수정] 모델을 먼저 device로 이동시켜야 함
            model.to(device)
            _ = model(Batch.from_data_list([warm]).to(device))

        # Load weights
        try:
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(str(ckpt), map_location="cpu")
            
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as e:
            print(f"❌ [모델 로딩 오류] {name} 구조 불일치.")
            print(f"   - 설정: {cfg}")
            print(f"   - 메시지: {e}")
            continue

        # Inference
        preds = infer_task([Batch.from_data_list([g]) for g in data_list], model, device,
                           batch_size=args.batch_size, is_cls=spec["cls"])
        outputs[name] = preds

    # --------------------- Postprocess & Save ---------------------
    out = df.copy()

    if "hERG" in outputs:
        out["hERG_prob"] = outputs["hERG"]
        out["hERG_flag"] = (out["hERG_prob"] >= args.herg_thr).astype(int)
    if "CYP3A4" in outputs:
        out["CYP3A4_prob"] = outputs["CYP3A4"]
        out["CYP3A4_flag"] = (out["CYP3A4_prob"] >= args.cyp_thr).astype(int)

    if "AqSolDB" in outputs:
        yhat = outputs["AqSolDB"].astype(np.float64)
        
        # Stats 로드 (역정규화)
        mu, std = 0.0, 1.0
        # Stats 파일 찾기 시도
        stats_candidates = [
            ckpt_dir / "admet_aqsoldb.stats.json",
            ckpt_dir / "admet_AqSolDB.stats.json"
        ]
        for stats_file in stats_candidates:
             if stats_file.exists():
                try:
                    s = json.load(open(stats_file))
                    mu, std = float(s.get("mu", 0.0)), float(s.get("std", 1.0))
                    break
                except: pass
        
        yhat = yhat * std + mu
        
        # Calibration (옵션)
        a, b = 1.0, 0.0
        calib_candidates = [
             ckpt_dir / "admet_aqsoldb.calib.json",
             ckpt_dir / "admet_AqSolDB.calib.json"
        ]
        for calib_file in calib_candidates:
            if calib_file.exists():
                try:
                    c = json.load(open(calib_file))
                    a, b = float(c.get("a", 1.0)), float(c.get("b", 0.0))
                    break
                except: pass
        
        yhat = a * yhat + b
        out["AqSolDB_pred"] = yhat

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote predictions: {args.out_csv} (n={len(out)})")

    if args.out_md:
        lines = []
        lines.append("# ADMET GNN Screening Summary\n")
        lines.append(f"- Inputs: **{len(out)}** molecules\n")
        found = [k for k in ["hERG","CYP3A4","AqSolDB"] if k in outputs]
        lines.append(f"- Tasks predicted: **{', '.join(found) if found else '(none)'}**\n")
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()