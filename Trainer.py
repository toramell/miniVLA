import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp import GradScaler, autocast

class VLA_Trainer:
    def __init__(self, model, optimizer, device, use_amp=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp and (device == "cuda")
        
        # 混合精度学習用のスケーラー
        if self.use_amp:
            self.scaler = GradScaler('cuda')
            print("Mixed Precision Training (AMP) enabled for A100")
        else:
            self.scaler = None

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Train"):
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            if self.use_amp:
                # 混合精度学習
                with autocast('cuda'):
                    logits = self.model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss = F.cross_entropy(logits, labels)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # グラディエントクリッピング
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 通常の学習
                logits = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = F.cross_entropy(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def eval(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        first_batch = True

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Eval"):
                images = batch["images"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # 評価時も混合精度を使用
                if self.use_amp:
                    with autocast('cuda'):
                        logits = self.model(
                            images=images,
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        loss = F.cross_entropy(logits, labels)
                else:
                    logits = self.model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                if first_batch:
                    print(f"\nデバッグ情報（最初のバッチ）:")
                    print(f"  Logitsの形状: {logits.shape}")
                    print(f"  Logitsの最大値: {logits.max().item():.4f}")
                    print(f"  Logitsの最小値: {logits.min().item():.4f}")
                    print(f"  予測値（最初の5個）: {preds[:5].cpu().tolist()}")
                    print(f"  正解ラベル（最初の5個）: {labels[:5].cpu().tolist()}")
                    print(f"  損失値: {loss.item():.6f}")
                    print(f"  一致数: {(preds == labels).sum().item()} / {labels.size(0)}")
                    first_batch = False
    
        accuracy = correct / total
        print(f"\n評価結果の詳細:")
        print(f"  総正解数: {correct}")
        print(f"  総サンプル数: {total}")
        print(f"  正答率: {accuracy:.6f}")
        return total_loss / len(dataloader), accuracy