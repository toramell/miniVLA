import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import argparse
import wandb # 追加
from transformers import get_cosine_schedule_with_warmup # 追加
from transformers import GPT2Tokenizer

from vla_robot_policy import VLARobotPolicy
from libero_dataset import LIBERODataset, libero_collate_fn

# 1. Tokenizerをmain関数の最初の方で定義して、train_one_epoch等に渡す必要があります
# あるいは、モデルの中から取り出してもOKです。ここではモデルから取り出す方式で書きます。

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    
    # DataParallelの場合、元のモデルは .module の中にある
    if isinstance(model, nn.DataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['images'].to(device, non_blocking=True)
        actions_gt = batch['actions'].to(device, non_blocking=True)
        instructions = batch['instructions'] # これはリスト
        
        # ★ここでトークン化してTensorにする（GPUへの転送もここで行う）
        tokenized = tokenizer(
            instructions, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask
        
        optimizer.zero_grad()
        
        # ★Tensorとして渡す (attention_maskも渡す)
        actions_pred = model(images, input_ids, attention_mask)
        
        loss = criterion(actions_pred, actions_gt)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        wandb.log({
            "train_step_loss": loss.item(),
            "lr": scheduler.get_last_lr()[0]
        })
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    
    if isinstance(model, nn.DataParallel):
        tokenizer = model.module.tokenizer
    else:
        tokenizer = model.tokenizer
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch['images'].to(device)
        actions_gt = batch['actions'].to(device)
        instructions = batch['instructions']
        
        # ★評価時も同様にトークン化
        tokenized = tokenizer(
            instructions, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        actions_pred = model(images, tokenized.input_ids, tokenized.attention_mask)
        
        loss = criterion(actions_pred, actions_gt)
        mae = torch.abs(actions_pred - actions_gt).mean()
        
        total_loss += loss.item()
        total_mae += mae.item()
        
    return total_loss / len(dataloader), total_mae / len(dataloader)

class HybridLoss(nn.Module):
    """MSEとL1を組み合わせたLoss"""
    def __init__(self, mse_weight=0.7):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mse_weight = mse_weight
    
    def forward(self, pred, target):
        return self.mse_weight * self.mse(pred, target) + (1 - self.mse_weight) * self.l1(pred, target)

def main(args):
    # WandB初期化
    wandb.init(project="vla-libero", config=args, mode="online" if not args.no_wandb else "disabled")
    
    # Device Setup
    if args.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices if args.cuda_devices else "0,1"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = LIBERODataset(args.data_dir, transform=transform)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Multi-GPUの場合はバッチサイズを調整
    num_gpus = torch.cuda.device_count()
    batch_size = args.batch_size * num_gpus if args.multi_gpu else args.batch_size
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=libero_collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=libero_collate_fn, num_workers=4, pin_memory=True
    )

    # Model Setup
    model = VLARobotPolicy(
        pretrained_checkpoint=args.pretrained_checkpoint,
        freeze_vision=args.freeze_vision,
        freeze_llm=args.freeze_llm
    )
    # --- ここから追加 ---
    print("\n" + "="*30)
    print("【学習パラメータの凍結状況チェック】")
    
    # チェックしたい主要なコンポーネントの名前
    # チェックしたい主要なコンポーネントの名前（修正版）
    # action_head を arm_head と gripper_head に変更
    components = ["vision_encoder", "qformer", "llm", "arm_head", "gripper_head"]
    
    for component in components:
        print(f"\n--- {component} ---")
        # そのコンポーネントに含まれるパラメータを1つだけ取り出して確認
        found = False
        for name, param in model.named_parameters():
            if component in name:
                status = "✅ 学習可能 (Trainable)" if param.requires_grad else "❄️ 凍結 (Frozen)"
                print(f"  {name} ... {status}")
                # 全部表示すると長すぎるので、各コンポーネントの最初の一部だけ表示してループを抜ける
                # もし詳細を見たいなら break を外してください
                found = True
                break 
        if not found:
            print("  (このコンポーネントは見つかりませんでした)")
            
    print("="*30 + "\n")
    # --- ここまで追加 ---
    
    if args.multi_gpu and num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # Optimizer (パラメータグループ分け)
    if isinstance(model, nn.DataParallel):
        raw_model = model.module
    else:
        raw_model = model
        
    param_groups = [
        {'params': raw_model.fusion_proj.parameters(), 'lr': args.lr * 10}, # 新規層は学習率高く
        {'params': raw_model.arm_head.parameters(), 'lr': args.lr * 10},
        {'params': raw_model.gripper_head.parameters(), 'lr': args.lr * 10},
        {'params': raw_model.qformer.parameters(), 'lr': args.lr * 5},      # Q-Former
        {'params': [p for n, p in raw_model.vision_encoder.named_parameters() if p.requires_grad], 'lr': args.lr},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Scheduler with Warmup (重要!)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1) # 最初の10%はWarmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    criterion = HybridLoss()
    best_loss = float('inf')

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, MAE={val_mae:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "val_loss": val_loss,
            "val_mae": val_mae,
            "train_loss_epoch": train_loss
        })
        
        # Save Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = f"libero_best_loss_{best_loss:.4f}.pt"
            torch.save(raw_model.state_dict(), save_path) # DataParallelのラッパーを外して保存
            print(f"Saved best model to {save_path}")
            
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrained_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--freeze_vision", action="store_true")
    parser.add_argument("--freeze_llm", action="store_true", default=True)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--cuda_devices", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    
    args = parser.parse_args()
    main(args)