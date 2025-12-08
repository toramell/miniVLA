#train_ssv2.py
import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import numpy as np
from collections import Counter, defaultdict
import pickle
import os
from pathlib import Path

from sample_vla import MiniVLA
from ssv2_dataset import UCF101MiniVLADataset
from collate_ucf101 import collate_ucf101
from Trainer import VLA_Trainer

def fast_stratified_sampling(dataset, target_size, seed=42, cache_dir="./cache"):
    """高速層別サンプリング：キャッシュとNumPy配列で最適化"""
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # A100 GPU用の最適化設定
    if device == "cuda":
        print("=== A100 GPU最適化設定 ===")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ CuDNN benchmark enabled")
        print("✓ TF32 enabled for Tensor Cores")

    print("\n=== Stage 3: Vision + Q-Former + LLM (簡易統合) ===")
    print("目的: LLM統合後も学習できるか検証")
    print("戦略: Vision特徴を優先的に使用、テキスト影響を最小化\n")

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

    print("\nLoading VLA Model with LLM...")
    # LLMを有効化してVLAモデルを構築
    model = MiniVLA(
        num_actions=101,
        use_llm=True,  # LLMを使用
        vision_trainable=True,
        llm_trainable=True
    ).to(device)
    
    model.get_trainable_params()

    train_set = UCF101MiniVLADataset(ds_train, model.tokenizer)
    val_set = UCF101MiniVLADataset(ds_val, model.tokenizer)

    batch_size = 64
    print(f"\n=== Training設定 ===")
    print(f"Batch size: {batch_size}")
    
    # 層ごとの学習率（ベースラインの成功を参考に調整）
    vision_params = list(model.vision_encoder.parameters())
    qformer_params = list(model.qformer.parameters())
    llm_params = [p for p in model.llm.parameters() if p.requires_grad]
    action_params = list(model.action_head.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': vision_params, 'lr': 1e-5, 'weight_decay': 0.01},  # ベースラインと同じ
        {'params': qformer_params, 'lr': 5e-5, 'weight_decay': 0.01},  # 慎重に
        {'params': llm_params, 'lr': 1e-5, 'weight_decay': 0.01},  # 慎重に
        {'params': action_params, 'lr': 1e-3, 'weight_decay': 0.01}  # ベースラインと同じ
    ])
    
    print(f"Optimizer: AdamW with layer-wise learning rates")
    print(f"  Vision Encoder: lr=1e-5")
    print(f"  Q-Former: lr=5e-5")
    print(f"  LLM: lr=1e-5")
    print(f"  Action Head: lr=1e-3")
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    num_epochs = 15  # ベースラインより長めに
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

    trainer = VLA_Trainer(model, optimizer, device, use_amp=True)

    print(f"\n=== Training開始 ===")
    print("目標: ベースライン88.46%に近い性能を達成")
    print("期待: 初期エポックで20%以上、最終的に60%以上\n")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.eval(val_loader)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f} (ベースライン: 0.8846)")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 進捗評価
        if epoch == 0 and val_acc < 0.05:
            print("  ⚠️  警告: 初期エポックで精度が5%未満です")
        elif val_acc > 0.50:
            print(f"  ✅ 良好: ベースラインの{val_acc/0.8846*100:.1f}%に到達")
        
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = f'vla_best_acc_{val_acc:.4f}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Best model saved: {checkpoint_path}")
        
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'vla_checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    print(f"\n{'='*60}")
    print(f"VLA Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Baseline Accuracy: 0.8846")
    print(f"Performance Ratio: {best_val_acc/0.8846*100:.1f}%")
    print(f"{'='*60}")
    
    if best_val_acc > 0.70:
        print("\n🎉 成功！VLAモデルがベースラインに近い性能を達成しました")
        print("   次のステップ: より複雑なタスクやロボット制御への応用")
    elif best_val_acc > 0.40:
        print("\n✅ 改善！LLM統合後も学習できています")
        print("   さらなる改善の余地があります：")
        print("   - エポック数を増やす")
        print("   - Q-Formerの構造を最適化")
        print("   - データ拡張を追加")
    else:
        print("\n⚠️  LLM統合にまだ問題があります")
        print("   考えられる原因：")
        print("   - LLMがVision情報を上書きしている")
        print("   - Q-Formerの圧縮が過度")
        print("   - 学習率が不適切")

if __name__ == "__main__":
    main()
