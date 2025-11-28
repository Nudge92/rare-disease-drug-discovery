import os
import glob
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
# ✅ [NEW] Scaffold Split을 위한 모듈 추가
from rdkit.Chem.Scaffolds import MurckoScaffold

# --- Helper Functions: Logging and Seed ---

def log(msg: str):
    """Prints a message to stdout."""
    print(msg, flush=True)

def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@dataclass
class PrepConfig:
    """Configuration class for data preprocessing."""
    pdbbind_root: str
    cache_root: str
    index_path: str
    pocket_radius: float = 12.0
    seed: int = 42
    train_split: float = 0.8 # Scaffold split에서는 내부 비율 사용 (7:1.5:1.5)
    max_samples: Optional[int] = None

# --- Molecule Processing Logic ---

def safe_sanitize_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """더 관대하게 분자 정규화"""
    if mol is None: 
        return None
    try:
        # 기본 sanitization 시도
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        try:
            # 실패 시 더 관대한 옵션으로 재시도
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return mol
        except Exception:
            try:
                # 마지막 시도: 수소 제거 후 재시도
                mol = Chem.RemoveHs(mol)
                Chem.SanitizeMol(mol)
                return mol
            except Exception:
                return None

def load_ligand(complex_dir: str) -> Optional[Chem.Mol]:
    """Flexibly finds and loads a ligand file from a directory."""
    ligand_patterns = [
        '*_ligand.sdf', '*.sdf', 
        '*_ligand.mol2', '*.mol2',
        '*_ligand.mol', '*.mol'
    ]
    
    for pattern in ligand_patterns:
        path = next(glob.iglob(os.path.join(complex_dir, pattern)), None)
        if path:
            mol = None
            try:
                if path.endswith('.sdf'):
                    suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
                    if suppl and len(suppl) > 0: 
                        mol = suppl[0]
                elif path.endswith('.mol2'):
                    mol = Chem.MolFromMol2File(path, removeHs=False, sanitize=False)
                elif path.endswith('.mol'):
                    mol = Chem.MolFromMolFile(path, removeHs=False, sanitize=False)
                
                if mol:
                    return safe_sanitize_mol(mol)
            except Exception:
                continue
    
    return None

def load_protein(complex_dir: str) -> Optional[Chem.Mol]:
    """Flexibly finds and loads a protein PDB file from a directory."""
    protein_patterns = ['*_protein.pdb', '*_pocket.pdb', '*.pdb']
    
    for pattern in protein_patterns:
        path = next(glob.iglob(os.path.join(complex_dir, pattern)), None)
        if path:
            try:
                mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False, proximityBonding=False)
                if mol and mol.GetNumAtoms() > 0:
                    return mol
            except Exception:
                continue
    
    return None

def extract_pocket(protein: Chem.Mol, ligand: Chem.Mol, radius: float) -> Optional[Chem.Mol]:
    """결합 포켓 추출 (원자와 결합 정보를 모두 포함하여 안전하게 생성)"""
    if ligand.GetNumConformers() == 0 or protein.GetNumConformers() == 0:
        return None

    try:
        lig_centroid = ligand.GetConformer().GetPositions().mean(axis=0)

        pocket_indices = set() # 빠른 조회를 위해 set 사용
        for i, atom in enumerate(protein.GetAtoms()):
            try:
                pos = protein.GetConformer().GetAtomPosition(i)
                dist = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - lig_centroid)
                if dist <= radius:
                    pocket_indices.add(i)
            except Exception:
                continue

        if not pocket_indices:
            return None

        # RWMol을 사용하여 분자를 편집 가능한 상태로 생성
        new_mol = Chem.RWMol()
        atom_map = {} # key: old_idx, value: new_idx

        # 1. 원자(Atom) 복사
        for old_idx in pocket_indices:
            atom = protein.GetAtomWithIdx(old_idx)
            new_idx = new_mol.AddAtom(atom) # 원자 객체 자체를 추가
            atom_map[old_idx] = new_idx

        # 2. 결합(Bond) 복사
        for old_idx in pocket_indices:
            atom = protein.GetAtomWithIdx(old_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_old_idx = neighbor.GetIdx()
                if neighbor_old_idx in pocket_indices and old_idx < neighbor_old_idx:
                    bond = protein.GetBondBetweenAtoms(old_idx, neighbor_old_idx)
                    new_mol.AddBond(atom_map[old_idx], atom_map[neighbor_old_idx], bond.GetBondType())

        # 3. 3D 좌표(Conformer) 복사
        conf = Chem.Conformer(new_mol.GetNumAtoms())
        for old_idx, new_idx in atom_map.items():
            pos = protein.GetConformer().GetAtomPosition(old_idx)
            conf.SetAtomPosition(new_idx, pos)
        new_mol.AddConformer(conf)

        final_mol = new_mol.GetMol()
        if final_mol:
            Chem.SanitizeMol(final_mol)
            return final_mol
        return None

    except Exception:
        return None

# --- Feature Generation Logic ---

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'Unknown']
HYBRIDIZATION_TYPES = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, 'Unknown']

def get_atom_features(atom: Chem.Atom) -> torch.Tensor:
    """Generates a 20-dimensional feature vector for an atom."""
    try:
        symbol = atom.GetSymbol()
        atom_type_idx = ATOM_TYPES.index(symbol) if symbol in ATOM_TYPES else len(ATOM_TYPES)-1
        atom_type = F.one_hot(torch.tensor(atom_type_idx), num_classes=len(ATOM_TYPES))
        
        hybrid = atom.GetHybridization()
        hybrid_idx = HYBRIDIZATION_TYPES.index(hybrid) if hybrid in HYBRIDIZATION_TYPES else len(HYBRIDIZATION_TYPES)-1
        hybrid_type = F.one_hot(torch.tensor(hybrid_idx), num_classes=len(HYBRIDIZATION_TYPES))

        scalar_features = torch.tensor([
            float(atom.GetDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalNumHs()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
        ], dtype=torch.float)

        return torch.cat([atom_type.float(), hybrid_type.float(), scalar_features])
    except Exception as e:
        # 실패 시 기본 특징 반환
        return torch.zeros(20, dtype=torch.float)

def build_complex_graph(pocket: Chem.Mol, ligand: Chem.Mol) -> Optional[Data]:
    """Builds a PyTorch Geometric Data object for the complex."""
    try:
        if pocket.GetNumAtoms() == 0 or ligand.GetNumAtoms() == 0:
            return None
        
        x_p = torch.stack([get_atom_features(a) for a in pocket.GetAtoms()])
        x_l = torch.stack([get_atom_features(a) for a in ligand.GetAtoms()])
        pos_p = torch.tensor(pocket.GetConformer().GetPositions(), dtype=torch.float)
        pos_l = torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float)
        
        x = torch.cat([x_p, x_l], dim=0)
        pos = torch.cat([pos_p, pos_l], dim=0)
        
        if torch.isnan(x).any() or torch.isnan(pos).any():
            return None
        
        dist_matrix = torch.cdist(pos, pos)
        edge_index = (dist_matrix <= 5.0).nonzero(as_tuple=False).t().contiguous()
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]

        return Data(x=x, pos=pos, edge_index=edge_index)
    except Exception:
        return None

# --- ✅ [NEW] Scaffold Splitting Logic ---

def scaffold_split(data_list: List[Tuple], frac_train=0.7, frac_valid=0.15, frac_test=0.15, seed=42):
    """
    Murcko Scaffold 기반 데이터 분할
    - 동일한 Scaffold를 가진 분자들은 모두 같은 세트(Train/Valid/Test)에 속하게 함.
    """
    log(f"\n🧪 Starting Scaffold Split (Ratio: {frac_train:.2f}/{frac_valid:.2f}/{frac_test:.2f})")
    
    scaffold_sets = defaultdict(list)
    
    # 1. Scaffold 생성 및 그룹핑
    for item in tqdm(data_list, desc="Generating Scaffolds"):
        cid, data, smiles = item
        # Scaffold가 없거나 실패하면 빈 문자열 그룹(generic)으로 처리
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=True)
        except:
            scaffold = ""
        scaffold_sets[scaffold].append(item)
    
    # 2. Scaffold 크기순 정렬 (큰 그룹부터 배치하여 균형 맞춤)
    scaffolds = sorted(scaffold_sets.items(), key=lambda x: len(x[1]), reverse=True)
    
    train_set, valid_set, test_set = [], [], []
    train_cutoff = len(data_list) * frac_train
    valid_cutoff = len(data_list) * (frac_train + frac_valid)
    
    # 3. 데이터셋 할당
    for scaffold, items in scaffolds:
        if len(train_set) + len(items) <= train_cutoff:
            train_set.extend(items)
        elif len(train_set) + len(valid_set) + len(items) <= valid_cutoff:
            valid_set.extend(items)
        else:
            test_set.extend(items)
    
    log(f"   - Scaffolds found: {len(scaffolds)}")
    return train_set, valid_set, test_set

# --- Main Execution Pipeline ---

# ✅ [FIX] 반환 타입에 SMILES(str) 추가
def process_one(complex_dir: str, cfg: PrepConfig) -> Tuple[Optional[Tuple[str, Data, str]], str]:
    """단일 복합체 처리 with 탈락 이유 반환"""
    complex_id = os.path.basename(complex_dir)
    
    try:
        # 1. 리간드 로드
        ligand = load_ligand(complex_dir)
        if not ligand:
            return None, "ligand_load_failed"
            
        # ✅ [NEW] SMILES 추출 (Scaffold Split용)
        smiles = Chem.MolToSmiles(ligand)

        # 2. 단백질 로드
        protein = load_protein(complex_dir)
        if not protein:
            return None, "protein_load_failed"

        # 3. 포켓 추출
        pocket = extract_pocket(protein, ligand, cfg.pocket_radius)
        if not pocket or pocket.GetNumAtoms() == 0:
            return None, "pocket_extraction_failed"

        # 4. 그래프 생성
        data = build_complex_graph(pocket, ligand)
        if data is None:
            return None, "graph_construction_failed"

        # ✅ [NEW] SMILES도 함께 반환
        return (complex_id, data, smiles), "success"
        
    except Exception as e:
        if "Submol" in str(e):
            return None, "rdkit_submol_error"
        elif "valence" in str(e).lower():
            return None, "valence_error"
        elif "sanitize" in str(e).lower():
            return None, "sanitization_error"
        else:
            return None, f"unexpected_error: {str(e)[:50]}"

def run(cfg: PrepConfig):
    log(f"🚀 Starting robust data preprocessing (Pocket Radius: {cfg.pocket_radius}Å)")
    set_seed(cfg.seed)

    log(f"📖 Reading complex IDs from index file: {cfg.index_path}")
    all_ids = []
    try:
        with open(cfg.index_path, 'r', errors='ignore') as f:
            for line in f:
                if not line.startswith('#'):
                    all_ids.append(line.strip().split()[0])
    except FileNotFoundError:
        log(f"❌ ERROR: Index file not found! Check path: {cfg.index_path}")
        return

    # 경로 찾기 로직
    all_dirs = []
    possible_bases = [
        os.path.dirname(os.path.dirname(cfg.index_path)),
        "data/raw/CASF-2016/coreset",
        "data/raw/casf2016_core/CASF-2016/coreset",
        "data/raw/pdbbind_refined/v2016",
    ]
    
    for cid in all_ids:
        for base in possible_bases:
            potential_dir = os.path.join(base, cid)
            if os.path.isdir(potential_dir):
                all_dirs.append(potential_dir)
                break
    
    if cfg.max_samples:
        random.shuffle(all_dirs)
        all_dirs = all_dirs[:cfg.max_samples]
    
    log(f"📁 Processing {len(all_dirs)} complex directories.")
    if not all_dirs:
        log("⚠️ No complex folders found. Please verify data paths.")
        return

    # 통계 수집
    processed_data = []
    failure_stats = defaultdict(int)
    
    for complex_dir in tqdm(all_dirs, desc="Processing Complexes"):
        result, reason = process_one(complex_dir, cfg)
        if result:
            processed_data.append(result)
            failure_stats["success"] += 1
        else:
            failure_stats[reason] += 1

    # 통계 출력
    total = len(all_dirs)
    success_count = len(processed_data)
    failure_count = total - success_count
    
    log(f"\n📊 **처리 결과 상세 통계** (총 {total}개)")
    log(f"✅ 성공: {success_count}개 ({success_count/total*100:.1f}%)")
    log(f"❌ 실패: {failure_count}개 ({failure_count/total*100:.1f}%)")
    
    if failure_count > 0:
        log(f"\n🔍 **실패 이유별 분석:**")
        for reason, count in sorted(failure_stats.items()):
            if reason != "success":
                log(f"   • {reason}: {count}개")

    if success_count == 0:
        log("❌ 처리된 데이터가 없어서 종료합니다.")
        return

    log("🏷️ Labeling data...")
    id_to_label = {}
    with open(cfg.index_path, 'r', errors='ignore') as f:
        for line in f:
            if not line.startswith('#'):
                try: 
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        id_to_label[parts[0]] = float(parts[3])
                except (ValueError, IndexError): 
                    continue
    
    final_data = []
    # ✅ [NEW] SMILES 포함해서 unpacking
    for cid, data, smiles in processed_data:
        if cid in id_to_label and not np.isnan(id_to_label[cid]):
            data.y = torch.tensor([id_to_label[cid]], dtype=torch.float)
            # Scaffold Split을 위해 smiles도 튜플에 유지
            final_data.append((cid, data, smiles))

    # ✅ [NEW] Scaffold Split 적용 (Random Shuffle 대체)
    train_data, valid_data, test_data = scaffold_split(final_data, frac_train=0.7, frac_valid=0.15, frac_test=0.15)

    # 저장 디렉토리
    train_dir = os.path.join(cfg.cache_root, "train")
    valid_dir = os.path.join(cfg.cache_root, "valid")
    test_dir = os.path.join(cfg.cache_root, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 저장 (SMILES는 저장하지 않고 Data 객체만 저장)
    # ✅ [FIX] unpacking 시 smiles 무시하고 저장
    for cid, data, _ in tqdm(train_data, desc="Saving train data"): 
        torch.save(data, os.path.join(train_dir, f"{cid}.pt"))
    for cid, data, _ in tqdm(valid_data, desc="Saving valid data"): 
        torch.save(data, os.path.join(valid_dir, f"{cid}.pt"))
    for cid, data, _ in tqdm(test_data, desc="Saving test data"): 
        torch.save(data, os.path.join(test_dir, f"{cid}.pt"))

    log(f"\n🎉 **전체 데이터 전처리 완료 (Scaffold Split)!**")
    log(f"   📁 훈련 데이터: {len(train_data)}개 → {train_dir}")
    log(f"   📁 검증 데이터: {len(valid_data)}개 → {valid_dir}")
    log(f"   📁 테스트 데이터: {len(test_data)}개 → {test_dir}")
    
    if train_data:
        sample_data = train_data[0][1]
        log(f"   🔍 특징 차원: {sample_data.x.size(1)}차원")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Preprocess PDBbind with Scaffold Split.")
    p.add_argument('--pdbbind_root', type=str, default="data/raw/pdbbind_refined")
    p.add_argument('--cache_root', type=str, default="data/processed/binding")
    p.add_argument('--index_path', type=str, default="data/raw/pdbbind_refined/v2016/index/INDEX_refined_data.2016")
    p.add_argument('--pocket_radius', type=float, default=12.0)
    p.add_argument('--max_samples', type=int, default=None)
    args = p.parse_args()
    run(PrepConfig(**vars(args)))