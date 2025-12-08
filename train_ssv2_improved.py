import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset
import numpy as np
from collections import Counter
import os
from tqdm import tqdm

# 必要な外部モジュール
from transformers import get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

# ユーザー定義モジュール (ディレクトリにあると想定)
from sample_vla import MiniVLA
from ssv2_dataset import UCF101MiniVLADataset
from collate_ucf101 import collate_ucf101

def get_weighted_sampler(dataset):
    """クラス不均衡を解消するためのWeightedRandomSamplerを作成"""
    labels = np.array(dataset['label'])
    class_counts = Counter(labels)
    num_samples = len(labels)
    
    # クラスごとの重み (出現頻度の逆数)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # 各サンプルに対する重み配列を作成
    sample_weights = [class_weights[label] for label in labels]
    
    # サンプラー作成
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    return sampler

def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler, epoch):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]")
    
    for batch in progress_bar:
        # データのデバイス転送
        # collate_ucf101の返り値に合わせて調整してください
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16 if device=="cuda" else torch.float32)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        action_labels = batch["action_labels"].to(device) # もしあれば
        
        optimizer.zero_grad()
        
        # AMP (Automatic Mixed Precision) Context
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            # modelのforward定義に合わせて引数を渡す
            # 想定: model(pixel_values, input_ids, attention_mask, labels=...)
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=action_labels # 損失計算に必要な場合
            )
            
            # モデルがlossを返す場合
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                # もしモデル内でloss計算しない場合はここで計算 (例: CrossEntropy)
                # 今回はMiniVLAの実装次第ですが、lossが返ると仮定します
                loss = outputs[0] if isinstance(outputs, tuple) else outputs

        # Backward & Optimization
        scaler.scale(loss).backward()
        
        # === 改善点: Gradient Clipping ===
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # === 改善点: Stepごとのスケジューリング ===
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})
        
    return total_loss / len(dataloader)

@torch.no_grad()
def eval_one_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="[Eval]")
    
    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        action_labels = batch["action_labels"].to(device)
        
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=action_labels
            )
            
            # Loss取得
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
                logits = outputs["logits"] # もしあれば
            else:
                loss = outputs[0]
                logits = outputs[1]

        total_loss += loss.item()
        
        # 精度計算 (Action Headの出力が logits であると仮定)
        # logits形状: [Batch, Num_Actions]
        preds = torch.argmax(logits, dim=1)
        correct += (preds == action_labels).sum().item()
        total += action_labels.size(0)
        
    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # A100/H100 GPU用の最適化
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("\n=== Stage 3+: VLA Improved Training (LoRA + WeightedSampling) ===")

    # 1. データセットのロード
    print("Loading UCF101...")
    ds_full_train = load_dataset("flwrlabs/ucf101", split="train")
    ds_full_test = load_dataset("flwrlabs/ucf101", split="test")
    
    # 全データを使用するが、Samplerでバランスを取る
    print(f"Train samples: {len(ds_full_train)}")
    print(f"Test samples: {len(ds_full_test)}")

    # 2. モデル構築
    print("\nInitializing MiniVLA...")
    model = MiniVLA(
        num_actions=101,
        use_llm=True,
        vision_trainable=True,
        llm_trainable=True 
    ).to(device)

    # === 改善点: LoRAの適用 ===
    print("Applying LoRA to LLM...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,           # ランク (パラメータ数と表現力のトレードオフ)
        lora_alpha=32,
        lora_dropout=0.1,
        # 対象モジュールはモデル構造による。
        # 一般的なLLMなら ["q_proj", "v_proj"] などが自動ターゲットになる場合が多い
    )
    # model.llm が HuggingFaceのモデルであることを想定
    model.llm = get_peft_model(model.llm, peft_config)
    
    # 訓練可能パラメータの確認
    model.llm.print_trainable_parameters()
    
    # Dataset & Loader
    train_set = UCF101MiniVLADataset(ds_full_train, model.tokenizer)
    val_set = UCF101MiniVLADataset(ds_full_test, model.tokenizer)

    batch_size = 32 # LoRAのおかげで少し増やせるかも、VRAM次第で調整
    
    # === 改善点: WeightedRandomSampler ===
    train_sampler = get_weighted_sampler(ds_full_train)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size,
        sampler=train_sampler, # shuffle=Trueとは排他利用
        collate_fn=lambda batch: collate_ucf101(batch, model.tokenizer),
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=lambda batch: collate_ucf101(batch, model.tokenizer),
        num_workers=8,
        pin_memory=True
    )

    # 3. Optimizer 設定
    # LoRAパラメータのみを取得
    lora_params = [p for p in model.llm.parameters() if p.requires_grad]
    vision_params = list(model.vision_encoder.parameters())
    qformer_params = list(model.qformer.parameters())
    action_params = list(model.action_head.parameters())

    optimizer = torch.optim.AdamW([
        {'params': vision_params, 'lr': 2e-5, 'weight_decay': 0.05},
        {'params': qformer_params, 'lr': 5e-5, 'weight_decay': 0.05},
        {'params': lora_params, 'lr': 2e-4, 'weight_decay': 0.01}, # LoRAは少し高めのLRでOK
        {'params': action_params, 'lr': 1e-3, 'weight_decay': 0.01}
    ])

    # 4. Scheduler 設定 (Warmup + Cosine)
    num_epochs = 15
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps) # 10% Warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    # 5. Training Loop
    best_val_acc = 0.0
    print(f"\nStart Training for {num_epochs} epochs...")
    print(f"Warmup steps: {num_warmup_steps}, Total steps: {num_training_steps}")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler, epoch
        )
        val_loss, val_acc = eval_one_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} Result:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_acc:.4f}")
        
        # チェックポイント保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"vla_best_lora_acc_{best_val_acc:.4f}.pt"
            
            # LoRAモデルの保存は少し特殊 (state_dictの扱い)
            # 全体を保存するか、adapterだけ保存するか選べますが、
            # 簡便のため全体を保存するコードにします（容量は食いますが復元が楽です）
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(), # LoRA込みのstate
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc
            }, save_path)
            print(f"  ✓ Model Saved: {save_path}")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()