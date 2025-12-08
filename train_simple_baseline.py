# train_simple_baseline.py
"""
VLA準備段階: Vision Encoder + 簡単な分類器でベースライン性能を確認
目的: CLIP Vision Encoderが正しく画像特徴を抽出できているか検証
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from collections import Counter, defaultdict
import pickle
import os
from pathlib import Path

from ssv2_dataset import UCF101MiniVLADataset
from collate_ucf101 import collate_ucf101
from sample_vla import CLIPVisionEncoder
from transformers import GPT2Tokenizer

# 簡易分類モデル（Vision Encoder + MLP）
class SimpleVisionClassifier(nn.Module):
    def __init__(self, num_classes=101, clip_name="openai/clip-vit-base-patch16"):
        super().__init__()
        
        # Vision Encoder（学習可能）
        self.vision_encoder = CLIPVisionEncoder(model_name=clip_name, trainable=True)
        vision_dim = self.vision_encoder.embed_dim
        
        # Global Average Pooling でパッチトークンを集約
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # 分類ヘッド（深めのMLP）
        self.classifier = nn.Sequential(
            nn.Linear(vision_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Tokenizerは互換性のため保持
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    def forward(self, images, input_ids=None, attention_mask=None):
        # Vision tokens: (B, N_patches, D)
        vision_tokens = self.vision_encoder(images)
        
        # Global pooling: (B, N_patches, D) -> (B, D)
        # Transpose for pooling: (B, D, N_patches) -> (B, D, 1) -> (B, D)
        pooled = self.pool(vision_tokens.transpose(1, 2)).squeeze(-1)
        
        # Classification
        logits = self.classifier(pooled)
        return logits
    
    def get_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"学習可能パラメータ: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
        return trainable, total


def fast_stratified_sampling(dataset, target_size, seed=42, cache_dir="./cache"):
    np.random.seed(seed)
    Path(cache_dir).mkdir(exist_ok=True)
    cache_file = f"{cache_dir}/ucf101_labels_{len(dataset)}_{seed}.pkl"
    
    if os.path.exists(cache_file):
        print(f"✓ キャッシュから読み込み中: {cache_file}")
        with open(cache_file, 'rb') as f:
            class_indices = pickle.load(f)
        print(f"✓ キャッシュ読み込み完了")
    else:
        print("キャッシュが見つかりません。ラベル情報を構築中...")
        all_labels = np.array(dataset['label'])
        print(f"  総データ数: {len(all_labels)}")
        
        class_indices = defaultdict(list)
        for idx, label in enumerate(all_labels):
            class_indices[label].append(idx)
        
        class_indices = {k: np.array(v) for k, v in class_indices.items()}
        
        print(f"  キャッシュに保存中: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(class_indices, f)
        print(f"✓ キャッシュ保存完了")
    
    num_classes = len(class_indices)
    samples_per_class = target_size // num_classes
    print(f"各クラスから{samples_per_class}サンプルを取得 (総計: {samples_per_class * num_classes})")
    
    selected_indices = []
    for class_id, indices in class_indices.items():
        available_samples = len(indices)
        take_samples = min(samples_per_class, available_samples)
        selected = np.random.choice(indices, size=take_samples, replace=False)
        selected_indices.extend(selected.tolist())
    
    print(f"✓ サンプリング完了: {len(selected_indices)}サンプル選択")
    return selected_indices


class SimpleTrainer:
    def __init__(self, model, optimizer, device, use_amp=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and (device == "cuda")
        
        if self.use_amp:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda')
            print("Mixed Precision Training enabled")
        else:
            self.scaler = None
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        from tqdm import tqdm
        for batch in tqdm(dataloader, desc="Train"):
            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            if self.use_amp:
                from torch.amp import autocast
                with autocast('cuda'):
                    logits = self.model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        accuracy = correct / total
        return total_loss / len(dataloader), accuracy
    
    def eval(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        from tqdm import tqdm
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Eval"):
                images = batch["images"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                if self.use_amp:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        logits = self.model(images)
                        loss = nn.functional.cross_entropy(logits, labels)
                else:
                    logits = self.model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        return total_loss / len(dataloader), accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        print("=== A100 GPU最適化設定 ===")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ CuDNN benchmark enabled")
        print("✓ TF32 enabled for Tensor Cores")

    print("\n=== Stage 2: Vision Encoder Baseline Test ===")
    print("目的: CLIP Vision Encoderが画像特徴を正しく抽出できるか検証")
    print("構成: CLIP ViT + Simple MLP Classifier (LLM無し)\n")

    print("Loading UCF101...")
    ds_full_train = load_dataset("flwrlabs/ucf101", split="train")
    ds_full_test = load_dataset("flwrlabs/ucf101", split="test")
    
    print("\n=== 高速層別サンプリング実行中 ===")
    train_indices = fast_stratified_sampling(ds_full_train, target_size=20000, seed=42)
    test_indices = fast_stratified_sampling(ds_full_test, target_size=5000, seed=42)
    
    ds_train = ds_full_train.select(train_indices)
    ds_val = ds_full_test.select(test_indices)
    
    print("\n=== サンプリング結果 ===")
    train_labels = np.array(ds_train['label'])
    val_labels = np.array(ds_val['label'])
    
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    
    print(f"訓練データのユニークラベル数: {len(set(train_labels))}")
    print(f"検証データのユニークラベル数: {len(set(val_labels))}")

    print("\nLoading Simple Vision Classifier...")
    model = SimpleVisionClassifier(num_classes=101).to(device)
    model.get_trainable_params()

    train_set = UCF101MiniVLADataset(ds_train, model.tokenizer)
    val_set = UCF101MiniVLADataset(ds_val, model.tokenizer)

    batch_size = 64
    print(f"\n=== Training設定 ===")
    print(f"Batch size: {batch_size}")
    
    # より積極的な学習率（事前学習済みモデルのfine-tuning）
    optimizer = torch.optim.AdamW([
        {'params': model.vision_encoder.parameters(), 'lr': 1e-5, 'weight_decay': 0.01},
        {'params': model.classifier.parameters(), 'lr': 1e-3, 'weight_decay': 0.01}
    ])
    
    print(f"Optimizer: AdamW")
    print(f"  Vision Encoder: lr=1e-5 (fine-tuning)")
    print(f"  Classifier: lr=1e-3 (from scratch)")
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    num_epochs = 10
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    print(f"Scheduler: CosineAnnealingLR ({num_epochs} epochs)")

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: collate_ucf101(batch, model.tokenizer),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: collate_ucf101(batch, model.tokenizer),
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    trainer = SimpleTrainer(model, optimizer, device, use_amp=True)

    print(f"\n=== Training開始 ===")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.eval(val_loader)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = f'baseline_best_acc_{val_acc:.4f}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  ✓ Best model saved: {checkpoint_path}")
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'baseline_checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print(f"Baseline Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}")
    
    if best_val_acc > 0.20:
        print("\n✅ Vision Encoder is working correctly!")
        print("   次のステップ: LLMを統合してVLAモデルを構築")
    else:
        print("\n⚠️  Vision Encoderに問題がある可能性があります")
        print("   データセットまたは前処理を確認してください")

if __name__ == "__main__":
    main()
