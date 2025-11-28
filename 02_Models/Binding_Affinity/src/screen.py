# src/binding_affinity/screen.py

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import pandas as pd # ✅ Pandas 라이브러리 추가

# predict.py에서 사용했던 Helper 함수들을 그대로 가져옵니다.
try:
    from predict import EGNN, extract_pocket, build_complex_graph, Chem, AllChem
    from rdkit.Geometry import Point3D # ◀◀◀ [수정됨] Point3D 임포트
except ImportError:
    print("❌ Error: 'predict.py' not found. Make sure it's in the same directory.")
    exit(1)

class ScreeningDataset(Dataset):
    """스크리닝을 위한 PyTorch Dataset 클래스"""
    # ◀◀◀ [수정됨] pocket_center 인자 추가
    def __init__(self, smiles_list, protein, pocket_center):
        self.smiles_list = smiles_list
        self.protein = protein
        # ◀◀◀ [수정됨] pocket_center 저장
        if pocket_center and (pocket_center[0] != 0 or pocket_center[1] != 0 or pocket_center[2] != 0):
             self.pocket_center = Point3D(pocket_center[0], pocket_center[1], pocket_center[2])
        else:
             self.pocket_center = None # (0,0,0)이면 이동 안 함

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        try:
            ligand = Chem.MolFromSmiles(smiles)
            if ligand is None: return None, smiles
            
            ligand = Chem.AddHs(ligand)
            if AllChem.EmbedMolecule(ligand, randomSeed=42) != 0:
                if AllChem.EmbedMolecule(ligand, useRandomCoords=True, randomSeed=42) != 0:
                    print(f"⚠️ Failed embedding: {smiles}")
                    return None, smiles
            AllChem.UFFOptimizeMolecule(ligand)

            # --- 🔻🔻🔻 [수정됨] 리간드 이동 코드 추가 🔻🔻🔻 ---
            if self.pocket_center:
                conf = ligand.GetConformer()
                # 모든 원자를 pocket_center 만큼 평행 이동
                for i in range(ligand.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    conf.SetAtomPosition(i, pos + self.pocket_center)
            # --- 🔺🔺🔺 코드 추가 완료 🔺🔺🔺 ---

            pocket = extract_pocket(self.protein, ligand)
            if pocket is None:
                # print(f"⚠️ Failed pocket extraction: {smiles}") # 너무 많은 메시지 방지
                return None, smiles

            data = build_complex_graph(pocket, ligand)
            if data is None:
                # print(f"⚠️ Failed graph build: {smiles}") # 너무 많은 메시지 방지
                return None, smiles
            
            # Data 객체에 smiles 정보 추가 (나중에 매핑을 위해)
            data.smiles = smiles
            return data, smiles
        except Exception as e:
            # print(f"⚠️ Error processing {smiles}: {e}") # 너무 많은 메시지 방지
            return None, smiles

def custom_collate_fn(batch):
    """배치 생성을 위한 collate 함수 (실패 샘플 처리 및 유효 SMILES 추적)"""
    from torch_geometric.data import Batch
    
    valid_data = []
    valid_smiles = [] # 성공적으로 처리된 SMILES 목록
    all_smiles = [] # 배치 내 모든 SMILES 목록 (원본 순서 유지)

    for i, item in enumerate(batch):
        data, smiles = item
        all_smiles.append(smiles)
        # edge_index 유효성 검사 강화 및 data 유효성 확인
        if data is not None and hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.numel() > 0:
             valid_data.append(data)
             valid_smiles.append(smiles) # 성공한 경우에만 추가

    if not valid_data:
        return None, all_smiles, [] # 유효 데이터 없으면 빈 리스트 반환

    try:
        batch_object = Batch.from_data_list(valid_data)
        # 배치 객체에 유효 SMILES 저장 (더 쉬운 매핑을 위해)
        batch_object.smiles = valid_smiles
        return batch_object, all_smiles, valid_smiles # 생성된 배치, 전체 SMILES, 유효 SMILES 반환
    except Exception as e:
        print(f"⚠️ Error in Batch.from_data_list: {e}")
        failed_smiles = set(all_smiles) - set(valid_smiles)
        return None, all_smiles, list(failed_smiles) # 실패한 SMILES 목록 반환

# ◀◀◀ [수정됨] pocket_center 인자 받도록 추가
def screen_batch(model_path, protein_path, library_path, output_path, batch_size=32, pocket_center=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Batch Virtual Screening on device: {device}")

    # --- 모델과 단백질 로드 ---
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        config, label_mean, label_std = ckpt['config'], ckpt['label_mean'], ckpt['label_std']
        model = EGNN(in_dim=ckpt['in_dim'], hidden=config['model']['hidden'], layers=config['model']['layers'], dropout=config['model']['dropout']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        
        protein = Chem.MolFromPDBFile(protein_path, removeHs=False)
        
        # --- 🔻🔻🔻 [수정됨] PDB 로딩 실패 검사 🔻🔻🔻 ---
        if protein is None:
            print(f"❌ Error: RDKit failed to load PDB file: {protein_path}")
            print("   (The file might be invalid or in a format RDKit cannot parse.)")
            print("   (Try opening the PDB in a text editor and removing 'CONECT' records at the end.)")
            return # 여기서 즉시 함수 종료
        # --- 🔺🔺🔺 ---

        Chem.SanitizeMol(protein)
        print("✅ Model and Protein loaded successfully!")
    except FileNotFoundError:
        print(f"❌ Error: Model or Protein file not found.")
        return
    except Exception as e:
        print(f"❌ Error during setup: {e}")
        return

    # --- 데이터셋 및 데이터로더 생성 ---
    import csv # # ◀◀ 1. csv 모듈 임포트
    try:
        smiles_list = [] # ◀◀ 2. 빈 리스트로 시작
        with open(library_path, 'r') as f:
            #◀◀ 3. CSV 리더로 파일 읽기
            reader = csv.reader(f)
            for row in reader:
                if row: # 빈 줄 아니면
                    smiles_list.append(row[0]) # ◀◀ 4. 첫 번째 열(SMILES) 추가
            
        if not smiles_list:
            print(f"❌ Error: No SMILES found in '{library_path}'")
            return
        
        # ◀◀◀ [수정됨] pocket_center를 Dataset에 전달
        dataset = ScreeningDataset(smiles_list, protein, pocket_center)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
    except FileNotFoundError:
        print(f"❌ Error: Library file not found at '{library_path}'")
        return
    except Exception as e:
         print(f"❌ Error creating dataset/loader: {e}")
         return

    # --- 스크리닝 실행 ---
    results = {}
    processed_count = 0
    error_count = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Screening Batches")
        for batch_data, all_smiles_in_batch, valid_smiles_in_batch in pbar:
            processed_count += len(all_smiles_in_batch)

            # 배치 생성 실패 또는 빈 배치 처리
            if batch_data is None or batch_data.num_graphs == 0:
                error_smiles = valid_smiles_in_batch # 이 경우 실패한 SMILES 목록일 수 있음
                for smiles in all_smiles_in_batch:
                    if smiles not in results: # 아직 결과 없으면 에러 처리
                        results[smiles] = "Preprocessing/Batching Error"
                        error_count +=1
                pbar.set_postfix({"Errors": f"{error_count}/{processed_count}"})
                continue

            # 예측 수행
            try:
                batch_data = batch_data.to(device)
                pred_norm = model(batch_data.x, batch_data.pos, batch_data.edge_index, batch_data.batch)
                predicted_values = pred_norm.cpu().numpy() * label_std + label_mean

                # ✅ FIX: 배치 객체에 저장된 SMILES를 이용해 정확하게 매핑
                predictions_map = {smiles: pred for smiles, pred in zip(batch_data.smiles, predicted_values)}

                # 배치 내 모든 SMILES에 대해 결과 할당
                for smiles in all_smiles_in_batch:
                    score = predictions_map.get(smiles)
                    if score is not None:
                        results[smiles] = score
                    elif smiles not in results:
                        results[smiles] = "Preprocessing Error"
                        error_count += 1
            except Exception as e:
                print(f"\n⚠️ Error during prediction for a batch: {e}")
                for smiles in valid_smiles_in_batch: # 예측 시도된 분자들만 에러 처리
                     if smiles not in results: results[smiles] = "Prediction Error"
                for smiles in all_smiles_in_batch: # 나머지는 전처리 에러
                     if smiles not in results: results[smiles] = "Preprocessing Error"
                error_count += len(all_smiles_in_batch) # 에러 카운트 증가


            pbar.set_postfix({"Errors": f"{error_count}/{processed_count}"})
    
    # --- 결과 저장 ---
    try:
        with open(output_path, 'w', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['SMILES', 'Predicted_pKd'])
            # smiles_list (원본 순서) 기준으로 결과 저장
            for smiles in smiles_list:
                score = results.get(smiles, "Unknown Error")
                writer.writerow([smiles, f"{score:.4f}" if isinstance(score, (float, np.floating)) else score])

        print(f"\n🎉 Batch screening complete! Found {len(results)} results.")
        print(f"   Total processed: {processed_count}, Errors: {error_count}")
        print(f"   Results saved to '{output_path}'.")

        # --- ✅ NEW: 상위 결과 출력 ---
        print("\n🏆 Top 20 Predicted Molecules:")
        try:
            # Pandas를 사용하여 CSV 읽고 정렬
            df = pd.read_csv(output_path)
            # 'Predicted_pKd'가 숫자인 행만 필터링
            df_numeric = df[pd.to_numeric(df['Predicted_pKd'], errors='coerce').notna()]
            df_numeric['Predicted_pKd'] = pd.to_numeric(df_numeric['Predicted_pKd'])
            
            # 점수(pKd) 기준으로 내림차순 정렬 후 상위 20개 선택
            top_results = df_numeric.sort_values(by='Predicted_pKd', ascending=False).head(20)
            
            # 보기 좋게 출력
            if top_results.empty:
                print("   (No valid numeric predictions found to rank)")
            else:
                print(top_results.to_string(index=False))

        except FileNotFoundError:
            print(f"   ⚠️ Could not read results file '{output_path}' to show top molecules.")
        except Exception as e:
            print(f"   ⚠️ Error processing results for top molecules: {e}")
            
    except Exception as e:
        print(f"❌ Error saving results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Virtual Screening with a trained EGNN model.")
    parser.add_argument("--model_path", type=str, default="checkpoints/egnn_best.pt")
    parser.add_argument("--protein_path", type=str, required=True)
    parser.add_argument("--library_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="screening_results_batch.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    # --- 🔻🔻🔻 [수정됨] 포켓 센터 인자 추가 🔻🔻🔻 ---
    parser.add_argument("--pocket_center", type=str, default="0,0,0", 
                        help="Center of the binding pocket, e.g., '42.6,42.9,89.6'")
    # --- 🔺🔺🔺 ---
    args = parser.parse_args()
    
    # Pandas 설치 확인
    try:
        import pandas as pd
    except ImportError:
        print("❌ Error: Pandas library is required for showing top results.")
        print("   Please install it using: pip install pandas")
        exit(1)

    # --- 🔻🔻🔻 [수정됨] pocket_center 파싱 🔻🔻🔻 ---
    try:
        # 콤마(,)로 분리된 문자열을 float 리스트로 변환
        pocket_center_coords = [float(c.strip()) for c in args.pocket_center.split(',')]
        if len(pocket_center_coords) != 3:
            raise ValueError("pocket_center must have exactly 3 coordinates (x,y,z).")
    except Exception as e:
        print(f"❌ Error parsing --pocket_center: {e}")
        print("   Expected format: 'x,y,z' (e.g., '42.63,42.89,89.61')")
        exit(1)

    # --- 🔻🔻🔻 [수정됨] pocket_center_coords 전달 🔻🔻🔻 ---
    screen_batch(args.model_path, 
                 args.protein_path, 
                 args.library_path, 
                 args.output_path, 
                 args.batch_size,
                 pocket_center=pocket_center_coords)