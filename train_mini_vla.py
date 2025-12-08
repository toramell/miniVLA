import torch
from torch.utils.data import DataLoader,random_split

from dataset import VLAClassificationDataset
from collate import vla_collate
from sample_vla import MiniVLA
from Trainer import VLA_Trainer

from PIL import Image
import numpy as np 

def random_image():
    # ランダムな画像を生成（ここでは単にランダムノイズ画像を使用）
    arr  = np.random.rand(224,224,3)*255
    return Image.fromarray(arr.astype('uint8'))

samples = [
    {"image":random_image(),"text":"pick the block","action":0},
    {"image":random_image(),"text":"move right","action":1},
    {"image":random_image(),"text":"move left","action":2},
] * 500  # サンプル数を増やすために繰り返し

# =============================================================
# 2. Dataset & DataLoader
# =============================================================
device = "cuda" if torch.cuda.is_available() else "mps"

model = MiniVLA().to(device)
tokenizer = model.tokenizer

dataset = VLAClassificationDataset(samples,tokenizer)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=vla_collate)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=vla_collate)

# =============================================================    
# 3. Optimizer＆Trainer
# =============================================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

trainer = VLA_Trainer(model, optimizer, device)

# =============================================================
# 4. Training Loop
# =============================================================
num_epochs = 5
for epoch in range(num_epochs):
    loss = trainer.train_epoch(train_loader)
    eval_loss, acc = trainer.eval(test_loader)  # タプルを正しく受け取る

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {loss:.4f} - Eval Loss: {eval_loss:.4f} - Test Acc: {acc:.4f}")