import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import argparse
import wandb
import numpy as np

# 既存のモジュールからインポート
# ※ファイル構成に合わせてパスは調整してください
from vla_robot_policy import VLARobotPolicy
from libero_dataset import LIBERODataset, libero_collate_fn

# ==========================================
# Loss Function Definition
# ==========================================
class HybridLoss(nn.Module):
    def __init__(self, pos_weight=10.0, rot_weight=20.0, grip_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss() 
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight
        self.grip_weight = grip_weight

    def forward(self, pred, target):
        loss_pos = self.mse(pred[:, :3], target[:, :3])
        loss_rot = self.mse(pred[:, 3:6], target[:, 3:6])
        
        grip_pred = pred[:, 6] 
        grip_target = (target[:, 6] > 0).float() 
        loss_grip = self.bce(grip_pred, grip_target)
        
        total_loss = (self.pos_weight * loss_pos) + \
                     (self.rot_weight * loss_rot) + \
                     (self.grip_weight * loss_grip)
        return total_loss

# ==========================================
# Visualization Helper
# ==========================================
def print_pred_vs_gt(pred, gt, epoch):
    """最初のサンプルの予測値と正解値をコンソールに表示して比較する"""
    print(f"\n--- Epoch {epoch} Inspection (Sample 0) ---")
    
    # Position
    p_pred = pred[0, :3].detach().cpu().numpy()
    p_gt = gt[0, :3].detach().cpu().numpy()
    print(f"Pos [x,y,z]:")
    print(f"  Pred: {np.round(p_pred, 4)}")
    print(f"  GT:   {np.round(p_gt, 4)}")
    print(f"  Diff: {np.round(np.abs(p_pred - p_gt), 4)}")
    
    # Gripper (Sigmoidかけて確率にする)
    g_logit = pred[0, 6].item()
    g_prob = torch.sigmoid(torch.tensor(g_logit)).item()
    g_gt_raw = gt[0, 6].item()
    g_gt = 1.0 if g_gt_raw > 0 else 0.0
    
    print(f"Gripper:")
    print(f"  Pred (Prob): {g_prob:.4f} (Logit: {g_logit:.4f})")
    print(f"  GT:          {g_gt:.1f} (Raw: {g_gt_raw})")
    print("-" * 40)

# ==========================================
# Main Overfit Loop
# ==========================================
def main(args):
    # WandBは無効化推奨（ローカルで見れば十分なため）だが、指定があれば有効化
    mode = "online" if not args.no_wandb else "disabled"
    wandb.init(project="vla-overfit-test", config=args, mode=mode)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset Setup
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = LIBERODataset(args.data_dir, transform=transform)
    
    # ★重要: データセットの先頭「batch_size個」だけを切り出す
    # これにより、モデルは常に同じ少数のデータを学習することになる
    #indices = list(range(args.batch_size))
    # データセットの長さ取得
    total_len = len(full_dataset)
    # ランダムなインデックスを生成 (固定シードで再現性確保)
    np.random.seed(42)
    indices = np.random.choice(total_len, args.batch_size, replace=False).tolist()

    overfit_dataset = Subset(full_dataset, indices)
    
    print(f"Overfit Test: Using only {len(overfit_dataset)} samples.")
    
    # ★重要: shuffle=False (順番すら変えない)
    train_loader = DataLoader(
        overfit_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=libero_collate_fn, 
        num_workers=0, # デバッグ時は0推奨
        pin_memory=True
    )

    # 2. Model Setup
    model = VLARobotPolicy(
        pretrained_checkpoint=args.pretrained_checkpoint,
        freeze_vision=args.freeze_vision,
        freeze_llm=args.freeze_llm
    )
    model.to(device)
    model.train() # 常にTrainモード

    # Tokenizer取得
    tokenizer = model.tokenizer

    # 3. Optimizer
    # Overfitテストではシンプルな設定にする（Schedulerなしで固定LRの方が挙動がわかりやすい場合もある）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    # 回転(rot_weight)を 0.0 に変更してください
    criterion = HybridLoss(pos_weight=10.0, rot_weight=0.0, grip_weight=1.0)

    print("\n=== Starting Overfit Test ===")
    print("目標: Lossがほぼ0になり、PredがGTと一致すること")

    # 4. Training Loop
    # データローダーから1つだけバッチを取り出して、それを永遠に使い回す方法もあるが、
    # ここではLoaderを回す形にする（Subsetなので中身は同じ）
    
    progress_bar = tqdm(range(1, args.epochs + 1))
    
    for epoch in progress_bar:
        total_loss = 0.0
        
        for batch in train_loader:
            images = batch['images'].to(device)
            actions_gt = batch['actions'].to(device)
            instructions = batch['instructions']
            
            tokenized = tokenizer(
                instructions, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            
            optimizer.zero_grad()
            
            # Forward
            actions_pred = model(images, tokenized.input_ids, tokenized.attention_mask)
            
            loss = criterion(actions_pred, actions_gt)
            loss.backward()
            
            # Clip Grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        # ログと表示
        wandb.log({"loss": avg_loss})
        progress_bar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # 10エポックごとに予測値を具体的に表示して確認
        if epoch % 20 == 0 or epoch == 1:
            print_pred_vs_gt(actions_pred, actions_gt, epoch)
            
        # 早期終了条件 (Lossが十分に下がったら)
        if avg_loss < 0.001:
            print(f"\n✅ Success! Loss converged to {avg_loss:.6f} at epoch {epoch}")
            break

    print("=== Test Finished ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="8サンプルだけ学習させる")
    parser.add_argument("--epochs", type=int, default=300, help="完全に覚えるまで回す")
    parser.add_argument("--lr", type=float, default=5e-4, help="少し高めのLRでOK")
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_llm", action="store_true", default=True)
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    main(args)