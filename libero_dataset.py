"""
libero_dataset.py

LIBERO HDF5データセットのためのPyTorch Datasetクラス
- HDF5形式のロボットデモンストレーションを読み込み
- 画像（agentview）と言語指示、7次元アクションを返す
"""
import os
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple


class LIBERODataset(Dataset):
    """
    LIBERO HDF5データセットからロボットデモンストレーションを読み込む
    
    Args:
        data_dir: LIBERO HDF5ファイルが格納されているディレクトリ
        transform: 画像に適用する変換（オプション）
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
                        # 新しいフォーマット（regenerateされたデータ）
                        images = demo['obs']['agentview_rgb'][()]
                    else:
                        # オリジナルフォーマット
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
        # アンダースコアをスペースに置換
        description = task_name.replace("_", " ")
        return description
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 画像を処理
        image = sample['image']
        if isinstance(image, np.ndarray):
            # NumPy配列からPIL Imageに変換
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # 画像変換を適用
        if self.transform is not None:
            image = self.transform(image)
        
        # アクションを取得（7次元: x, y, z, roll, pitch, yaw, gripper）
        action = torch.from_numpy(sample['action'].astype(np.float32))
        
        return {
            'image': image,
            'action': action,
            'instruction': sample['instruction'],
            'task_name': sample['task_name']
        }


def libero_collate_fn(batch: List[Dict]) -> Dict:
    """
    LIBEROデータセット用のcollate関数
    """
    images = torch.stack([item['image'] for item in batch])
    actions = torch.stack([item['action'] for item in batch])
    instructions = [item['instruction'] for item in batch]
    task_names = [item['task_name'] for item in batch]
    
    return {
        'images': images,
        'actions': actions,
        'instructions': instructions,
        'task_names': task_names
    }


if __name__ == "__main__":
    # テスト
    from torchvision import transforms
    
    # 画像変換
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # データセットをロード
    data_dir = "/home/toramoto/toramoto/vla/LIBERO/libero/datasets/libero_spatial"
    dataset = LIBERODataset(data_dir, transform=transform)
    
    print(f"Dataset size: {len(dataset)}")
    
    # 最初のサンプルを確認
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Action shape: {sample['action'].shape}")
    print(f"Instruction: {sample['instruction']}")
    print(f"Action: {sample['action']}")
