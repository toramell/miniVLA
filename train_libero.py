import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import argparse
try:
    import wandb
except ImportError:
    wandb = None
from transformers import get_cosine_schedule_with_warmup

from vla_robot_policy import VLARobotPolicy
from libero_dataset import LIBERODataset, libero_collate_fn

# =========================================================================
# ★ 修正ポイント: 10次元アクションに対応したLoss関数
# =========================================================================
class RobotTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 10) -> [pos(3), rot6d(6), gripper(1)]
            target: (B, 10) -> [pos(3), rot6d(6), gripper(1)]
        """
        # 1. 座標 (x, y, z) [Index 0-2]
        pos_pred = pred[:, :3]
        pos_target = target[:, :3]
        loss_pos = self.mse(pos_pred, pos_target)
        
        # 2. 6D回転 (rot6d) [Index 3-8]
        # 6D回転は連続値なのでMSEで学習可能
        rot_pred = pred[:, 3:9]
        rot_target = target[:, 3:9]
        loss_rot = self.mse(rot_pred, rot_target)
        
        # 3. グリッパー (gripper) [Index 9]
        grip_pred = pred[:, 9] # (B,)
        
        # 正解データも 0 or 1 に変換して BCE で学習
        # targetのグリッパーが > 0 なら 1.0 (閉), <= 0 なら 0.0 (開)
        grip_target = (target[:, 9] > 0).float()
        loss_grip = self.bce(grip_pred, grip_target)
        
        # 重み付け: 座標:10, 回転:1, グリッパー:1
        # (6D回転は値が小さくないので、極端な重み付けは不要)
        total_loss = 10.0 * loss_pos + 1.0 * loss_rot + 10.0 * loss_grip
        
        return total_loss


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    
    # モデルのラッパーを剥がしてtokenizerを取得
    unwrapped_model = model
    # 1. torch.compile (_orig_mod) を剥がす
    if hasattr(unwrapped_model, "_orig_mod"):
        unwrapped_model = unwrapped_model._orig_mod
    # 2. DataParallel (module) を剥がす
    if hasattr(unwrapped_model, "module"):
        unwrapped_model = unwrapped_model.module
        
    tokenizer = unwrapped_model.tokenizer

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch['images'].to(device, non_blocking=True)
        actions_gt = batch['actions'].to(device, non_blocking=True)
        instructions = batch['instructions'] # List[str]
        
        # テキストをトークン化
        tokenized = tokenizer(
            instructions, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        optimizer.zero_grad()
        
        # Forward (画像, input_ids, mask)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            actions_pred = model(
                images, 
                tokenized.input_ids, 
                tokenized.attention_mask
            )
            loss = criterion(actions_pred, actions_gt)        


        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if wandb and wandb.run:
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
    
    # モデルのラッパーを剥がしてtokenizerを取得
    unwrapped_model = model
    if hasattr(unwrapped_model, "_orig_mod"):
        unwrapped_model = unwrapped_model._orig_mod
    if hasattr(unwrapped_model, "module"):
        unwrapped_model = unwrapped_model.module
        
    tokenizer = unwrapped_model.tokenizer
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch['images'].to(device)
        actions_gt = batch['actions'].to(device)
        instructions = batch['instructions']
        
        tokenized = tokenizer(
            instructions, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        actions_pred = model(
            images, 
            tokenized.input_ids, 
            tokenized.attention_mask
        )
        
        loss = criterion(actions_pred, actions_gt)
        mae = torch.abs(actions_pred - actions_gt).mean()
        
        total_loss += loss.item()
        total_mae += mae.item()
        
    return total_loss / len(dataloader), total_mae / len(dataloader)

def main(args):
    # WandB初期化
    if args.use_wandb and wandb:
        wandb.init(project="vla-libero-6d", config=args)
    
    # Device Setup
    if args.multi_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices if args.cuda_devices else "0,1"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # =========================================================================
    # ★ 修正: データ拡張 (Augmentation) の定義
    # =========================================================================
    
    # 1. 学習用: 難易度を上げる「特訓メニュー」
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 少し大きめにリサイズしてから...
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)), # ランダムに切り抜く (位置ズレ対策)
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05), # 色を激しく変える (照明対策)
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), # たまにぼかす (ノイズ対策)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 検証用: そのままの綺麗な画像
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Loading dataset from {args.data_dir}")
    
    # ★ ポイント: 同じデータを2回ロードして、それぞれ別のTransformを適用する
    # (メモリ効率は少し悪いですが、実装が一番確実でバグりません)
    full_train_dataset = LIBERODataset(args.data_dir, transform=train_transform)
    full_val_dataset   = LIBERODataset(args.data_dir, transform=val_transform)
    
    # インデックスを固定して分割
    total_size = len(full_train_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    
    # 再現性のためシード固定して、同じように分割する
    generator = torch.Generator().manual_seed(42)
    train_dataset, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    _, val_dataset   = random_split(full_val_dataset,   [train_size, val_size], generator=generator)
    
    print(f"Train samples (Augmented): {len(train_dataset)}")
    print(f"Val samples (Clean):       {len(val_dataset)}")
    
    num_gpus = torch.cuda.device_count()
    batch_size = args.batch_size * num_gpus if args.multi_gpu else args.batch_size
    
    print(f"Batch size: {batch_size} (per GPU: {args.batch_size})")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=libero_collate_fn, num_workers=16, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=libero_collate_fn, num_workers=16, pin_memory=True
    )

    # Model Setup
    # 以前のチェックポイントは次元が違うため使えません。新規学習推奨。
    print("Initializing NEW model with 6D rotation architecture...")
    model = VLARobotPolicy(
        pretrained_checkpoint=None,
        freeze_vision=False, # Visionも学習
        freeze_llm=True
    )

    if args.pretrained_checkpoint:
        print(f"\nLoading pretrained weights from {args.pretrained_checkpoint} ...")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location='cpu')
        
        # DataParallelやtorch.compileのプレフィックスを削除してロード
        state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            # torch.compileの削除
            if k.startswith('_orig_mod.'):
                k = k[10:]
            # DataParallelの削除
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print("✓ Pretrained weights loaded successfully! Starting fine-tuning.")
    
    # もしUCF101の重みだけロードしたい場合はここでVLARobotPolicyの内部で処理されますが、
    # 今回は省略（スクラッチまたはUCF101パスを指定してロード）
    
    if args.multi_gpu and num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    try:
        model = torch.compile(model)
        print("✅ Model compiled with torch.compile!")
    except Exception as e:
        print(f"⚠️ Could not compile model: {e}")

    # Optimizer (学習率調整: Visionは低めに)
    # モデルのラッパーを順番に剥がして、中身(VLARobotPolicy)を取り出す
    raw_model = model
    
    # 1. torch.compile (_orig_mod) があれば剥がす
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
        
    # 2. DataParallel (module) があれば剥がす
    if hasattr(raw_model, "module"):
        raw_model = raw_model.module

    # 念のため確認（デバッグ用）
    print(f"Optimizer model type: {type(raw_model)}")
        
    base_lr = args.lr
    
    param_groups = [
        {'params': raw_model.fusion_proj.parameters(), 'lr': base_lr * 10},
        {'params': raw_model.arm_head.parameters(),    'lr': base_lr * 10},
        {'params': raw_model.gripper_head.parameters(),'lr': base_lr * 10},
        {'params': raw_model.qformer.parameters(),     'lr': base_lr},
        # Vision Encoderは壊れやすいので学習率を低くする
        {'params': [p for n, p in raw_model.vision_encoder.named_parameters() if p.requires_grad], 'lr': base_lr * 0.1},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    criterion = RobotTaskLoss()
    best_loss = float('inf')

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        
        if wandb and wandb.run:
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
            # DataParallelのラッパーを外して保存
            state_dict = raw_model.state_dict()
            torch.save(state_dict, save_path)
            print(f"✓ Saved best model to {save_path}")

    if wandb and wandb.run:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30) # 少し多めに
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--cuda_devices", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Path to pretrained model for fine-tuning")
    
    args = parser.parse_args()
    main(args)