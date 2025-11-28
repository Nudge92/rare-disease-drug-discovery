import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import yaml
from scipy import stats # ✅ FIX: 이 줄을 추가했습니다!

# train.py에서 사용했던 Helper 함수 및 클래스들을 그대로 가져옵니다.
try:
    from train import ImprovedDataset, improved_collate_fn, metrics, set_seed
except ImportError:
    print("❌ Error: 'train.py' not found. Make sure it's in the same directory.")
    exit(1)

# 모델 설계도 import
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from egnn import EGNN

def test(model_path, test_data_dir):
    """훈련된 모델을 테스트 데이터셋으로 최종 평가합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    set_seed(42)

    print(f"🧠 Loading model and config from {model_path}...")
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint not found at '{model_path}'")
        return

    try:
        ckpt = torch.load(model_path, map_location=device)
        config = ckpt['config']
        label_mean = ckpt['label_mean']
        label_std = ckpt['label_std']
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    model = EGNN(
        in_dim=ckpt['in_dim'],
        hidden=config['model']['hidden'],
        layers=config['model']['layers'],
        dropout=config['model']['dropout']
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print("✅ Model loaded successfully!")

    print(f"📝 Loading test data from {test_data_dir}...")
    if not os.path.exists(test_data_dir):
        print(f"❌ Error: Test data directory not found at '{test_data_dir}'")
        return
        
    test_ds = ImprovedDataset(test_data_dir)
    test_ds.label_mean = label_mean
    test_ds.label_std = label_std
    
    test_loader = DataLoader(
        test_ds,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        collate_fn=improved_collate_fn,
        num_workers=0
    )
    print(f"✅ Test data loaded: {len(test_ds)} samples")

    print("🤖 Running final evaluation on the test set...")
    y_list, yhat_list = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if batch is None: continue
            
            x = batch['x'].to(device)
            coords = batch['coords'].to(device)
            edge_index = batch['edge_index'].to(device)
            batch_idx = batch['batch_idx'].to(device)
            y_original = batch['y_original']

            pred_norm = model(x, coords, edge_index, batch_idx)
            
            pred_original = pred_norm.cpu().numpy() * label_std + label_mean
            
            yhat_list.append(pred_original)
            y_list.append(y_original.numpy())

    if not y_list:
        print("❌ No valid samples were processed in the test set.")
        return

    y_test = np.concatenate(y_list).flatten()
    yhat_test = np.concatenate(yhat_list).flatten()
    
    mask = ~np.isnan(y_test) & ~np.isnan(yhat_test)
    y_test, yhat_test = y_test[mask], yhat_test[mask]

    rmse = np.sqrt(np.mean((yhat_test - y_test)**2))
    mae = np.mean(np.abs(yhat_test - y_test))
    
    pearson_r = 0.0
    if np.std(y_test) > 0 and np.std(yhat_test) > 0:
        pearson_r, _ = stats.pearsonr(yhat_test, y_test)

    print("\n" + "="*50)
    print("🏆 Final Model Performance (Test Set)")
    print("="*50)
    print(f"   - RMSE:      {rmse:.4f}")
    print(f"   - MAE:       {mae:.4f}")
    print(f"   - Pearson r: {pearson_r:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the final EGNN model on the test set.")
    parser.add_argument("--model_path", type=str, default="checkpoints/egnn_best.pt",
                       help="Path to the best trained model checkpoint.")
    parser.add_argument("--test_data_dir", type=str, default="data/processed/binding/test",
                       help="Directory of the processed test data.")
    args = parser.parse_args()
    
    test(args.model_path, args.test_data_dir)