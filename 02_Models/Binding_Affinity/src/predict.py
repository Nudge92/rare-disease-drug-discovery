import os
import argparse
import numpy as np
import torch
from torch_geometric.data import Data

# RDKit 관련 import
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType

# 모델 설계도 import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from egnn import EGNN

# ==============================================================================
# 데이터 전처리 함수들 (data_prep.py에서 가져옴)
# ==============================================================================
ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'Unknown']
HYBRIDIZATION_TYPES = [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, 'Unknown']

def get_atom_features(atom):
    """원자의 20차원 특징 벡터를 생성합니다."""
    symbol = atom.GetSymbol()
    atom_type_idx = ATOM_TYPES.index(symbol) if symbol in ATOM_TYPES else len(ATOM_TYPES) - 1
    atom_type = torch.nn.functional.one_hot(torch.tensor(atom_type_idx), num_classes=len(ATOM_TYPES))
    
    hybrid = atom.GetHybridization()
    hybrid_idx = HYBRIDIZATION_TYPES.index(hybrid) if hybrid in HYBRIDIZATION_TYPES else len(HYBRIDIZATION_TYPES) - 1
    hybrid_type = torch.nn.functional.one_hot(torch.tensor(hybrid_idx), num_classes=len(HYBRIDIZATION_TYPES))

    scalar_features = torch.tensor([
        float(atom.GetDegree()), float(atom.GetFormalCharge()), float(atom.GetTotalNumHs()),
        float(atom.GetIsAromatic()), float(atom.IsInRing()),
    ], dtype=torch.float)

    return torch.cat([atom_type.float(), hybrid_type.float(), scalar_features])



# predict.py 파일의 extract_pocket 함수를 이 코드로 교체하세요.

def extract_pocket(protein, ligand, radius=12.0):
    """리간드 주변의 단백질 결합 주머니(pocket)를 추출합니다."""
    try:
        lig_centroid = ligand.GetConformer().GetPositions().mean(axis=0)
        
        # 포켓 원자 찾기
        pocket_atoms = []
        pocket_coords = []
        
        for i, atom in enumerate(protein.GetAtoms()):
            pos = protein.GetConformer().GetAtomPosition(i)
            coord = np.array([pos.x, pos.y, pos.z])
            dist = np.linalg.norm(coord - lig_centroid)
            
            if dist <= radius:
                pocket_atoms.append(atom)
                pocket_coords.append(coord)
        
        if not pocket_atoms:
            print(f"❌ No pocket atoms found within {radius}Å radius.")
            return None
        
        #print(f"✅ Found {len(pocket_atoms)} pocket atoms within {radius}Å")
        
        # 새로운 분자 생성 (안전한 방식)
        pocket_mol = Chem.RWMol()
        
        # 원자 추가 (방향족 속성 제거로 안전하게)
        for atom in pocket_atoms:
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            # 방향족 속성 복사하지 않음 (문제 방지)
            pocket_mol.AddAtom(new_atom)
        
        # 좌표 설정
        conf = Chem.Conformer(len(pocket_atoms))
        for i, coord in enumerate(pocket_coords):
            conf.SetAtomPosition(i, [float(coord[0]), float(coord[1]), float(coord[2])])
        
        pocket_mol.AddConformer(conf)
        
        # 안전한 sanitization (방향족 검사 제외)
        try:
            pocket = pocket_mol.GetMol()
            # 기본 sanitization만 수행 (방향족 검사 제외)
            Chem.SanitizeMol(pocket, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
            #print(f"✅ Extracted pocket with {pocket.GetNumAtoms()} atoms (basic sanitization)")
            return pocket
        except:
            # 그것도 실패하면 최소한의 sanitization만
            try:
                pocket = pocket_mol.GetMol()
                Chem.SanitizeMol(pocket, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)
                print(f"✅ Extracted pocket with {pocket.GetNumAtoms()} atoms (no sanitization)")
                return pocket
            except:
                # 마지막 대안: sanitization 없이
                pocket = pocket_mol.GetMol()
                print(f"⚠️ Extracted pocket with {pocket.GetNumAtoms()} atoms (raw)")
                return pocket
        
    except Exception as e:
        print(f"❌ Error extracting pocket: {e}")
        import traceback
        traceback.print_exc()
        return None

def build_complex_graph(pocket, ligand, smiles=None):
    """단백질-리간드 복합체 그래프를 생성합니다."""
    try:
        # 특징 벡터 생성
        x_p = torch.stack([get_atom_features(a) for a in pocket.GetAtoms()])
        x_l = torch.stack([get_atom_features(a) for a in ligand.GetAtoms()])
        
        # 좌표 정보
        pos_p = torch.tensor(pocket.GetConformer().GetPositions(), dtype=torch.float)
        pos_l = torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float)
        
        # 연결
        x = torch.cat([x_p, x_l], dim=0)
        pos = torch.cat([pos_p, pos_l], dim=0)
        
        # 차원 검증
        if x.size(1) != 20:
            print(f"⚠️ Feature dimension mismatch: {x.size(1)} (expected 20)")
            if x.size(1) < 20:
                padding = torch.zeros(x.size(0), 20 - x.size(1))
                x = torch.cat([x, padding], dim=1)
            else:
                x = x[:, :20]
        
        # 정규화 (훈련 시와 동일하게)
        if torch.std(x) > 1e-6:
            x_mean = torch.mean(x, dim=0, keepdim=True)
            x_std = torch.std(x, dim=0, keepdim=True) + 1e-6
            x = (x - x_mean) / x_std
            x = torch.clamp(x, -3, 3)
        
        # edge_index 생성 (예시: fully connected, 실제론 더 복잡할 수 있음)
        num_atoms = x.size(0)
        if num_atoms < 2:
            print("❌ Not enough atoms to build edge_index.")
            return None
        edge_index = torch.combinations(torch.arange(num_atoms), r=2).t()
        if edge_index is None or edge_index.numel() == 0:
            print("❌ No edges in edge_index.")
            return None
        
        #print(f"✅ Built complex graph: {x.size(0)} atoms, {x.size(1)} features, {edge_index.size(1)} edges")
        return Data(x=x, pos=pos, edge_index=edge_index, smiles=smiles)
        
    except Exception as e:
        print(f"❌ Error building complex graph: {e}")
        return None

# ==============================================================================
# 메인 예측 함수
# ==============================================================================
# predict.py 파일의 predict 함수를 이 코드로 교체하세요.

def predict(model_path, protein_path, smiles):
    """훈련된 모델로 새로운 분자의 결합력을 예측합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 1. 모델의 "뇌"와 "해독 키" 불러오기
    print(f"🧠 Loading model from {model_path}...")
    try:
        import warnings
        warnings.filterwarnings('ignore', category=FutureWarning)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        config = ckpt['config']
        label_mean = ckpt['label_mean']
        label_std = ckpt['label_std']
        print(f"✅ Model config: hidden={config['model']['hidden']}, layers={config['model']['layers']}")
        print(f"✅ Label normalization: mean={label_mean:.3f}, std={label_std:.3f}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. AI의 "신체 설계도"로 모델 복원하기
    try:
        model = EGNN(
            in_dim=ckpt['in_dim'],
            hidden=config['model']['hidden'],
            layers=config['model']['layers'],
            dropout=config['model']['dropout']
        ).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return

    # 3. 새로운 "문제지" 준비 (단백질과 리간드)
    print(f"🔬 Preparing protein '{os.path.basename(protein_path)}' and ligand...")
    
    # 단백질 로딩
    try:
        protein = Chem.MolFromPDBFile(protein_path, removeHs=False)
        if protein is None:
            print("❌ Failed to load protein from PDB file.")
            return

        # ✅ FIX: 단백질 분자를 먼저 정리하여 오류를 바로잡습니다.
        print("🔬 Sanitizing protein structure...")
        Chem.SanitizeMol(protein)
        print("✅ Protein sanitized.")
        
        print(f"✅ Loaded protein with {protein.GetNumAtoms()} atoms")
    except Exception as e:
        print(f"❌ Error loading or sanitizing protein: {e}")
        return
    
    # 리간드 생성 및 3D 구조 최적화
    try:
        ligand = Chem.MolFromSmiles(smiles)
        if ligand is None:
            print("❌ Invalid SMILES string.")
            return
        
        ligand = Chem.AddHs(ligand)
        
        if AllChem.EmbedMolecule(ligand, randomSeed=42) != 0:
            if AllChem.EmbedMolecule(ligand, useRandomCoords=True, randomSeed=42) != 0:
                print("❌ Failed to generate 3D coordinates for ligand.")
                return
        
        AllChem.UFFOptimizeMolecule(ligand)
        print("✅ 3D structure optimized with UFF")
        print(f"✅ Generated ligand with {ligand.GetNumAtoms()} atoms")
        
    except Exception as e:
        print(f"❌ Error processing ligand: {e}")
        return

    # 4. "번역기"로 AI가 이해할 수 있는 데이터로 변환
    pocket = extract_pocket(protein, ligand)
    if pocket is None:
        return

    data = build_complex_graph(pocket, ligand)
    if data is None:
        return
        
    data = data.to(device)

    # 5. 예측 실행
    print("🤖 Predicting binding affinity...")
    try:
        with torch.no_grad():
            edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
            batch_idx = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
            
            pred_norm = model(data.x, data.pos, edge_index, batch_idx)
            
            if pred_norm.dim() > 0:
                pred_norm = pred_norm[0] if pred_norm.size(0) == 1 else pred_norm.mean()
            
        # 6. "해독 키"로 결과 해독하기
        predicted_value = pred_norm.item() * label_std + label_mean
        
        print("\n" + "="*50)
        print(f"🎉 Prediction Complete!")
        print(f"   Protein: {os.path.basename(protein_path)}")
        print(f"   Ligand (SMILES): {smiles}")
        print(f"   Predicted Binding Affinity: {predicted_value:.4f} pKd")
        print(f"   📊 Confidence: Based on r=0.579 correlation")
        print("="*50)
        
        return predicted_value
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

def validate_inputs(protein_path, smiles):
    """입력 검증 함수"""
    if not os.path.exists(protein_path):
        print(f"❌ Protein file not found: {protein_path}")
        return False
    
    if not protein_path.lower().endswith('.pdb'):
        print("⚠️ Warning: File doesn't have .pdb extension")
    
    # SMILES 간단 검증
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ Invalid SMILES: {smiles}")
            return False
    except:
        print(f"❌ Error parsing SMILES: {smiles}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict binding affinity using a trained EGNN model.")
    parser.add_argument("--model_path", type=str, default="checkpoints/egnn_best.pt", 
                       help="Path to the trained model checkpoint.")
    parser.add_argument("--protein_path", type=str, required=True, 
                       help="Path to the target protein PDB file.")
    parser.add_argument("--smiles", type=str, required=True, 
                       help="SMILES string of the ligand molecule.")
    args = parser.parse_args()
    
    # 입력 검증
    if not validate_inputs(args.protein_path, args.smiles):
        exit(1)
    
    # 예측 실행
    result = predict(args.model_path, args.protein_path, args.smiles)
    
    if result is not None:
        print(f"\n🎯 Final Result: {result:.4f} pKd")
    else:
        print("\n❌ Prediction failed.")
        exit(1)