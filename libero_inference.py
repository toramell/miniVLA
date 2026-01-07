"""
libero_inference.py

学習済みVLAモデルを使ったリアルタイムロボット制御推論
- 単一画像からアクションを予測
- バッチ推論にも対応
- LIBEROシミュレーション環境での使用を想定
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Union, List
import argparse
import sys
from rotation_utils import rot6d_to_euler

# vla_robot_policy.py と libero_dataset.py があるディレクトリにパスを通す
sys.path.append('.') 

from vla_robot_policy import VLARobotPolicy

class VLAInference:
    """VLAモデルの推論クラス"""
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Args:
            checkpoint_path: LIBERO学習済みモデルのパス
            device: 推論デバイス ("cuda" or "cpu")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 画像前処理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # モデルを初期化（重みはNone）
        # ※ 最新のVLARobotPolicyは引数で自動的に構造が決まる
        print("Initializing model architecture...")
        self.model = VLARobotPolicy(
            pretrained_checkpoint=None,  # ここで余計なロードをしない
            freeze_vision=False,
            freeze_llm=True
        )
        
        # LIBERO学習済み重みをロード
        print(f"Loading trained weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # state_dictの取り出しと修正
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # DataParallel ('module.') のプレフィックス削除
        new_state_dict = {}

        for k, v in state_dict.items():
            # まず torch.compile の接頭辞を削除
            if k.startswith('_orig_mod.'):
                k = k[10:]  # '_orig_mod.' は10文字
                
            # 次に DataParallel の接頭辞を削除
            if k.startswith('module.'):
                k = k[7:]   # 'module.' は7文字
            
            new_state_dict[k] = v
        
        # 重みをロード
        try:
            self.model.load_state_dict(new_state_dict)
            print("✓ Model weights loaded successfully")
        except RuntimeError as e:
            print(f"⚠ Warning during loading weights: {e}")
            # 形状が合う部分だけロードを試みる（緊急避難）
            model_dict = self.model.state_dict()
            valid_dict = {k: v for k, v in new_state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(valid_dict)
            self.model.load_state_dict(model_dict, strict=False)
            print(f"✓ Partially loaded {len(valid_dict)}/{len(new_state_dict)} layers")

        self.model.to(self.device)
        self.model.eval()
        
        # トークナイザーの取得（モデル内部のものを使用）
        self.tokenizer = self.model.tokenizer

    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """画像を前処理してTensorに変換"""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                # 0-1のfloatなら255倍
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            image = Image.fromarray(image)
        
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0)  # (1, 3, H, W)

    def predict_action(
        self, 
        image: Union[np.ndarray, Image.Image],
        instruction: str,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        単一画像からアクションを予測
        """
        # 1. 画像の前処理
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # 2. テキストのトークン化
        tokenized = self.tokenizer(
            [instruction], 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # 3. 推論
        with torch.no_grad():
            # forward(images, input_ids, attention_mask)
            raw_output = self.model(
                image_tensor, 
                tokenized.input_ids, 
                tokenized.attention_mask
            )
            # --- ★ここを修正: 出力のデコード ---
            
            # 1. アーム出力の分解 (合計9次元)
            # 0-2: 位置 (3)
            # 3-8: 回転6D (6)
            arm_output = raw_output[:, :9] 
            
            pos_pred = arm_output[:, :3]     # (B, 3)
            rot6d_pred = arm_output[:, 3:9]  # (B, 6)
            
            # 2. 6D回転 -> オイラー角(3) に変換
            rot_euler = rot6d_to_euler(rot6d_pred) # (B, 3)
            
            # 3. アームアクションの再構築 (x, y, z, roll, pitch, yaw)
            # Tanhは位置(x,y,z)だけにかけるべきだが、VLAモデル内ですでにTanhかかっている場合
            # 回転6DはTanhかけてもかけなくても良いが、正規化されるのでそのままでOK
            # ★重要: 前回の学習で pos に Tanh をかけていた場合は、ここでも範囲に注意
            
            # 今回はモデル内で pos のみに Tanh をかけ、rot6d は生のまま出すように
            # vla_robot_policy.py の forward も少し直すのがベストですが、
            # とりあえず推論側では変換するだけです。

            arm_action = torch.cat([pos_pred, rot_euler], dim=-1) # (B, 6)
            #arm_action[:, 2] -= 0.02  # z座標を少し下げる補正

            # 4. グリッパー処理 (前回と同じ)
            gripper_logit = raw_output[:, 9:] # 9番目がグリッパー
            gripper_prob = torch.sigmoid(gripper_logit)            
            final_action = torch.cat([arm_action, gripper_prob], dim=-1) # (B, 7) に戻る

        # 4. 返却
        if return_numpy:
            return final_action.squeeze(0).cpu().numpy()
        else:
            return final_action.squeeze(0)

    def interpret_action(self, action: np.ndarray) -> dict:
        """アクション配列を解釈"""
        return {
            'position': {
                'x': float(action[0]), 'y': float(action[1]), 'z': float(action[2])
            },
            'orientation': {
                'roll': float(action[3]), 'pitch': float(action[4]), 'yaw': float(action[5])
            },
            'gripper': "CLOSED (1.0)" if action[6] > 0 else "OPEN (-1.0)",
            'gripper_raw': float(action[6])
        }

def demo_inference(checkpoint_path: str):
    """デモ実行"""
    print("\n=== VLA Inference Demo ===")
    
    try:
        inference = VLAInference(checkpoint_path)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # ダミーデータ
    dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    instructions = [
        "pick up the red block",
        "open the gripper",
        "move carefully to the left"
    ]
    
    print("\n--- Running Predictions ---")
    for instr in instructions:
        print(f"\nInstruction: '{instr}'")
        action = inference.predict_action(dummy_image, instr)
        
        result = inference.interpret_action(action)
        print(f"  Position:    (x={result['position']['x']:.3f}, y={result['position']['y']:.3f}, z={result['position']['z']:.3f})")
        print(f"  Orientation: (r={result['orientation']['roll']:.3f}, p={result['orientation']['pitch']:.3f}, y={result['orientation']['yaw']:.3f})")
        print(f"  Gripper:     {result['gripper']}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo"])
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo_inference(args.checkpoint)

if __name__ == "__main__":
    main()