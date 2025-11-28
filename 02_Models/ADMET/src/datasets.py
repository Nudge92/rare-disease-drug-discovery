# datasets.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch  # Batch 임포트
from rdkit import Chem
from rdkit.Chem import Descriptors       # Descriptors 임포트
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.Scaffolds import MurckoScaffold
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, List

# =========================================================
# 1) 분자 → 그래프 변환 (노드/엣지 특성)
# ---------------------------------------------------------
ATOM_LIST = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]

# --- [ 수정 1: 전역 상수 정의 ] ---
RDKIT_FEAT_NAMES = [
    'MolWt', 'NumRadicalElectrons', 'NumValenceElectrons',
    'HeavyAtomCount', 'MolLogP', 'MolMR', 'RingCount',
    'FractionCSP3', 'TPSA', 'NumHAcceptors', 'NumHDonors',
    'NumRotatableBonds', 'NumAliphaticRings', 'NumAromaticRings',
    'NumSaturatedRings',
]
NUM_RDKIT_FEATS = len(RDKIT_FEAT_NAMES)
# --- [ 수정 완료 ] ---


def _one_hot(value: int, choices: List[int]) -> List[int]:
    return [1 if value == c else 0 for c in choices]

def mol_to_graph(smiles: str):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    
    # ---------- [ 💡 RDKit 피처 계산 ] ----------
    try:
        rdkit_feats_raw = [getattr(Descriptors, name)(m) for name in RDKIT_FEAT_NAMES]
        rdkit_feats_clean = np.nan_to_num(rdkit_feats_raw, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- 👇 [핵심 수정 2: 1D -> 2D 텐서로 생성] ---
        # (1D 텐서 [F]가 아닌 2D 텐서 [1, F]로 만듭니다)
        # 이래야 Batch로 합쳐질 때 [B, F] 텐서가 됩니다.
        rdkit_feats = torch.tensor(rdkit_feats_clean, dtype=torch.float32).unsqueeze(0)
        # --- 👆 [수정 완료] ---
        
    except Exception:
        # --- 👇 [핵심 수정 2: 1D -> 2D 텐서로 생성] ---
        # (실패 시에도 2D 텐서 [1, F]를 반환해야 합니다)
        rdkit_feats = torch.zeros((1, NUM_RDKIT_FEATS), dtype=torch.float32)
        # --- 👆 [수정 완료] ---
    # ---------- [ 계산 완료 ] ----------

    # ---------- node features ----------
    x_rows = []
    for a in m.GetAtoms():
        Z = a.GetAtomicNum()
        deg = a.GetDegree()
        formal = a.GetFormalCharge()
        arom = int(a.GetIsAromatic())

        atom_oh = _one_hot(Z, ATOM_LIST)
        deg_oh = _one_hot(min(deg, 5), list(range(6)))
        x_rows.append(atom_oh + deg_oh + [formal, arom])

    x = torch.tensor(x_rows, dtype=torch.float32)

    # ---------- edge index / edge features ----------
    e_index = []
    e_attr = []
    for b in m.GetBonds():
        u = b.GetBeginAtomIdx()
        v = b.GetEndAtomIdx()

        bt = b.GetBondType()
        bond_oh = [
            int(bt == BondType.SINGLE),
            int(bt == BondType.DOUBLE),
            int(bt == BondType.TRIPLE),
            int(bt == BondType.AROMATIC),
        ]
        conj = int(b.GetIsConjugated())
        ring = int(b.IsInRing())
        feats = bond_oh + [conj, ring]

        e_index.append([u, v]); e_attr.append(feats)
        e_index.append([v, u]); e_attr.append(feats)

    if len(e_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float32)
    else:
        edge_index = torch.tensor(e_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(e_attr, dtype=torch.float32)

    return Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr,
        rdkit_feats=rdkit_feats  # (이제 [1, F] 2D 텐서입니다)
    )

# =========================================================
# 2) Dataset: {task}_{split}.parquet 로드
# ---------------------------------------------------------
class TdcGraphSet(Dataset):
    def __init__(self, parquet_path: str, task_type: str):
        df = pd.read_parquet(parquet_path)
        assert "smiles" in df.columns and "Y" in df.columns

        self.task_type = task_type
        self.graphs = []
        ys = []
        
        print(f"Loading and processing {parquet_path}...")
        
        valid_indices = []
        for i, (s, y) in enumerate(zip(df["smiles"].tolist(), df["Y"].tolist())):
            g = mol_to_graph(s)
            if g is not None:
                self.graphs.append(g)
                valid_indices.append(i)

        ys_raw = df["Y"].tolist()
        ys = [ys_raw[i] for i in valid_indices]

        self.y = torch.tensor(ys, dtype=torch.float32)
        
        if len(self.graphs) != len(self.y):
             raise RuntimeError("Graph list and Y list length mismatch!")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.graphs[i], self.y[i]

# =========================================================
# 3) Scaffold split 유틸
# ---------------------------------------------------------
def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core, isomericSmiles=True)

def scaffold_split_df(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_cluster: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    rng = np.random.RandomState(seed)

    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(df["smiles"].tolist()):
        scaf = bemis_murcko_scaffold(smi)
        buckets[scaf].append(i)

    clusters = list(buckets.values())
    big = [c for c in clusters if len(c) >= min_cluster]
    small = [c for c in clusters if len(c) <  min_cluster]
    big.sort(key=lambda c: -len(c))
    rng.shuffle(small)
    clusters = big + small

    N = len(df)
    tgt_tr, tgt_va = int(N * train_ratio), int(N * val_ratio)
    out = {"train": [], "valid": [], "test": []}
    cnt = {"train": 0, "valid": 0, "test": 0}

    for c in clusters:
        deficits = {
            "train": tgt_tr - cnt["train"],
            "valid": tgt_va - cnt["valid"],
            "test":  N - tgt_tr - tgt_va - cnt["test"],
        }
        bucket = max(deficits, key=deficits.get)
        out[bucket].extend(c)
        cnt[bucket] += len(c)

    tr = df.iloc[out["train"]].reset_index(drop=True)
    va = df.iloc[out["valid"]].reset_index(drop=True)
    te = df.iloc[out["test"]].reset_index(drop=True)

    return tr, va, te

def _load_source_table(task: str, root: Path) -> pd.DataFrame:
    p_parq = root / f"{task}.parquet"
    p_csv  = root / f"{task}.csv"

    if p_parq.exists():
        df = pd.read_parquet(p_parq)
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
    else:
        raise FileNotFoundError(f"No source table for {task} under {root}")

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("smiles", "drug"): col_map[c] = "smiles"
        if lc in ("y", "label", "target"): col_map[c] = "Y"
    if col_map:
        df = df.rename(columns=col_map)

    assert "smiles" in df.columns and "Y" in df.columns, \
        f"'smiles' and 'Y' required, got {df.columns.tolist()}"
    return df[["smiles", "Y"]].copy()

def ensure_scaffold_parquets(
    task: str,
    root: str,
    seed: int = 42,
    tr: float = 0.7,
    va: float = 0.15,
    te: float = 0.15,
    min_cluster: int = 1,
) -> None:
    rootp = Path(root)
    p_tr = rootp / f"{task}_train.parquet"
    p_va = rootp / f"{task}_valid.parquet"
    p_te = rootp / f"{task}_test.parquet"
    if p_tr.exists() and p_va.exists() and p_te.exists():
        print(f"Scaffold parquets for {task} already exist. Skipping split.")
        return

    print(f"Generating scaffold parquets for {task}...")
    df = _load_source_table(task, rootp)
    tr_df, va_df, te_df = scaffold_split_df(
        df, seed=seed,
        train_ratio=tr, val_ratio=va, test_ratio=te,
        min_cluster=min_cluster
    )
    tr_df.to_parquet(p_tr, index=False)
    va_df.to_parquet(p_va, index=False)
    te_df.to_parquet(p_te, index=False)
    print(f"Saved {task} splits to {rootp}")

# =========================================================
# 4) (선택) Collate helper
# ---------------------------------------------------------
# --- [ 수정 3: 'collate' 함수 정의 ] ---
def collate(batch):
    """
    DataLoader(GeoLoader)에서 사용할 collate_fn.
    (graph, y_scalar) 리스트를 (Batch_of_graphs, y_tensor)로 합칩니다.
    """
    graphs, ys = zip(*batch)
    
    # 1. 그래프 리스트를 하나의 'Batch' 객체로 합칩니다.
    graph_batch = Batch.from_data_list(list(graphs))
    
    # 2. Y값 (스칼라) 리스트를 하나의 텐서로 합칩니다.
    y_tensor = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in ys], dim=0)
    
    return graph_batch, y_tensor
# --- [ 수정 완료 ] ---