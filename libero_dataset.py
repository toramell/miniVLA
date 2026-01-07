"""
libero_dataset.py

LIBERO HDF5データセットのためのPyTorch Datasetクラス
- HDF5形式のロボットデモンストレーションを読み込み
- 画像（agentview）と言語指示
- アクションを修正（3D回転 -> 6D回転）して返す
"""
import os
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple

# 作成したrotation_utilsから関数をインポート
from rotation_utils import euler_to_rot6d


class LIBERODataset(Dataset):
    """
    LIBERO HDF5データセットからロボットデモンストレーションを読み込む
    """
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        # すべてのHDF5ファイルを読み込み
        self._load_all_demos()
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_all_demos(self):
        """ディレクトリ内のすべてのHDF5ファイルからデモを読み込む"""
        hdf5_files = sorted(self.data_dir.glob("*_demo.hdf5"))
        
        for hdf5_path in hdf5_files:
            # ファイル名からタスク名を抽出
            task_name = hdf5_path.stem.replace("_demo", "")
            task_description = self._task_name_to_description(task_name)
            
            # HDF5ファイルを開く
            with h5py.File(hdf5_path, 'r') as f:
                data_group = f['data']
                
                # 各デモンストレーションを処理
                for demo_key in data_group.keys():
                    demo = data_group[demo_key]
                    
                    # 観測データを取得
                    if 'obs' in demo:
                        # Agentviewの画像を使用
                        images = demo['obs']['agentview_rgb'][()]
                    else:
                        images = demo['agentview_image'][()]
                    
                    actions = demo['actions'][()]
                    
                    # 各タイムステップをサンプルとして追加
                    for t in range(len(actions)):
                        self.samples.append({
                            'image': images[t],
                            'action': actions[t],
                            'instruction': task_description,
                            'task_name': task_name
                        })
    
    def _task_name_to_description(self, task_name: str) -> str:
        """タスク名を自然言語の指示に変換"""
        return task_name.replace("_", " ")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # --- 画像処理 ---
        image = sample['image']
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        if self.transform is not None:
            image = self.transform(image)
        
        # --- アクション処理 (ここを修正) ---
        raw_action = sample['action']
        
        # エラー修正: すでにTensorの場合の対策
        if isinstance(raw_action, torch.Tensor):
            action_tensor = raw_action.float()
        else:
            action_tensor = torch.from_numpy(raw_action).float()

        # 分解: [x, y, z, roll, pitch, yaw, gripper]
        pos = action_tensor[:3]      # 位置 (3)
        euler = action_tensor[3:6]   # 回転 (3)
        gripper = action_tensor[6:]  # グリッパー (1)

        # 回転を Euler(3) -> 6D(6) に変換
        # euler_to_rot6d は (B, 3) を期待するので次元を追加して戻す
        rot6d = euler_to_rot6d(euler.unsqueeze(0)).squeeze(0) # (6,)

        # 新しいアクションベクトル: [pos(3), rot6d(6), gripper(1)] = 10次元
        new_action = torch.cat([pos, rot6d, gripper], dim=0)

        # --- テキスト処理 ---
        instruction = sample['instruction']
        task_name = sample['task_name']

        # キー名は train_libero.py で期待されるものに合わせる ('images' 複数形など)
        return {
            'images': image,
            'actions': new_action,  # 10次元アクション
            'instructions': instruction,
            'task_names': task_name
        }


def libero_collate_fn(batch: List[Dict]) -> Dict:
    """
    DataLoader用の結合関数
    """
    # Tensorはスタックする
    images = torch.stack([item['images'] for item in batch])
    actions = torch.stack([item['actions'] for item in batch])
    
    # 文字列はリストにする
    instructions = [item['instructions'] for item in batch]
    task_names = [item['task_names'] for item in batch]
    
    return {
        'images': images,
        'actions': actions,
        'instructions': instructions,
        'task_names': task_names
    }


if __name__ == "__main__":
    # 動作確認用
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # パスは環境に合わせて書き換えてください
    data_dir = "/home/toramoto/toramoto/vla/LIBERO/libero/datasets/libero_spatial"
    
    if os.path.exists(data_dir):
        dataset = LIBERODataset(data_dir, transform=transform)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Image shape: {sample['images'].shape}")
        print(f"Action shape: {sample['actions'].shape}") # 10次元になっているはず
        print(f"Instruction: {sample['instructions']}")
    else:
        print("Data directory not found.")