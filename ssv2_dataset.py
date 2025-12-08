import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class UCF101MiniVLADataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_size=224):
        self.ds = hf_dataset
        self.tokenizer = tokenizer

        # 画像前処理（ViT/CLIP 標準）
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),  # [0,1]
        ])

        # ラベル名のマッピング（ClassLabel が付いてる）
        if hasattr(self.ds.features["label"], "int2str"):
            self.id2label = self.ds.features["label"].int2str
        else:
            self.id2label = lambda x: str(x)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        # 1. 画像（PIL）→ tensor
        img = item["image"]
        img = self.transform(img)  # (3,224,224)

        # 2. テキスト生成（ラベル情報を含めない汎用的なプロンプト）
        # モデルは画像から動作を推論する必要がある
        text = "What action is being performed in this video?"

        # 3. Tokenize（collate関数でまとめて行うため、ここでは不要）
        # tok は collate_ucf101 で処理される

        return {
            "images": img,
            "actions": int(item["label"])  # 101クラス分類
        }
