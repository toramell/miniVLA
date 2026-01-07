import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from vla_robot_policy import VLARobotPolicy
from libero_dataset import LIBERODataset, libero_collate_fn

# === 設定 ===
DATA_DIR = "/home/toramoto/toramoto/vla/LIBERO/libero/datasets/libero_spatial" # パスは環境に合わせてください
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Using device: {DEVICE}")
    
    # 1. データセット読み込み
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = LIBERODataset(DATA_DIR, transform=transform)
    
    # ★たった1つのサンプルだけ取り出す
    subset = Subset(dataset, [0])
    loader = DataLoader(subset, batch_size=1, collate_fn=libero_collate_fn)
    
    # データの中身チェック
    batch = next(iter(loader))
    images = batch['images'].to(DEVICE)
    actions_gt = batch['actions'].to(DEVICE)
    instructions = batch['instructions']
    
    print("\n=== Data Inspection ===")
    print(f"Image Shape: {images.shape}")
    print(f"Image Max: {images.max().item():.4f}, Min: {images.min().item():.4f}")
    if images.max().item() == 0 and images.min().item() == 0:
        print("⚠️ 警告: 画像が真っ黒(All Zeros)です！データ読み込みに問題があります。")
        return
    
    print(f"Action GT: {actions_gt[0, :3].cpu().numpy()} (Pos only)")
    
    # 2. モデル準備
    model = VLARobotPolicy(pretrained_checkpoint=None, freeze_vision=False, freeze_llm=True)
    model.to(DEVICE)
    model.train()
    
    # Tokenizer
    tokenizer = model.tokenizer
    tokenized = tokenizer(instructions, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3) # 強めのLR
    mse = nn.MSELoss()
    
    print("\n=== Gradient Flow Check & Training ===")
    
    # 3. 学習ループ (100回だけ回す)
    for step in range(100):
        optimizer.zero_grad()
        
        # Forward
        # requires_grad=True になっているか確認するために画像に勾配追跡をつける
        images.requires_grad_(True) 
        
        actions_pred = model(images, tokenized.input_ids, tokenized.attention_mask)
        
        # PositionだけのLossをとる
        loss = mse(actions_pred[:, :3], actions_gt[:, :3])
        
        loss.backward()
        
        # ★最重要: 勾配チェック
        if step == 0:
            print(f"Initial Loss: {loss.item():.6f}")
            
            # 画像入力まで勾配が戻ってきているか？
            if images.grad is not None and images.grad.abs().sum() > 0:
                print("✅ Success: Vision Encoderまで勾配が流れています！ (Image grad detected)")
            else:
                print("❌ ERROR: Vision Encoderまで勾配が届いていません！ (No Image grad)")
                print("考えられる原因: freeze_vision=Trueのまま, またはモデル内部で .detach() されている")
            
            # Vision Encoderの重みに勾配があるか？
            # param名に合わせて調整してください
            found_grad = False
            for name, param in model.vision_encoder.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    print(f"✅ Success: Vision Encoderのパラメータ({name})が更新されようとしています。")
                    found_grad = True
                    break
            if not found_grad:
                print("❌ ERROR: Vision Encoderのパラメータに勾配がありません！")

        optimizer.step()
        
        if (step+1) % 20 == 0:
            print(f"Step {step+1}: Loss = {loss.item():.6f}")

    print(f"\nFinal Pred: {actions_pred[0, :3].detach().cpu().numpy()}")
    print(f"Target GT : {actions_gt[0, :3].cpu().numpy()}")

if __name__ == "__main__":
    main()