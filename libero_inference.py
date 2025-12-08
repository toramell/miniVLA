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
import h5py
from pathlib import Path

from vla_robot_policy import VLARobotPolicy


class VLAInference:
    """VLAモデルの推論クラス"""
    def __init__(self, checkpoint_path: str, device: str = "cuda", ucf101_checkpoint_path: str = "/home/toramoto/toramoto/miniVLA/vla_best_acc_0.8573.pt"):
        """
        Args:
            checkpoint_path: LIBERO微調整済みモデルのチェックポイントパス
            device: 推論デバイス ("cuda" or "cpu")
            ucf101_checkpoint_path: UCF101事前学習済みモデルのパス
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 画像前処理
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # モデルを初期化（UCF101事前学習済み重みをロード）
        print(f"Initializing model with UCF101 pretrained weights from {ucf101_checkpoint_path}")
        self.model = VLARobotPolicy(
            pretrained_checkpoint=ucf101_checkpoint_path,  # UCF101事前学習済み重みをロード
            freeze_vision=False,
            freeze_llm=True,
            action_dim=7
        )
        
        # LIBERO微調整済み重みをロード
        print(f"Loading LIBERO fine-tuned weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # LIBERO学習済み重みを適用（部分的なロード）
        if 'model_state_dict' in checkpoint:
            libero_state_dict = checkpoint['model_state_dict']
        else:
            libero_state_dict = checkpoint
        
        # モデルの現在のstate_dict
        model_state_dict = self.model.state_dict()
        
        # LIBEROで学習された部分のみを更新
        updated_keys = []
        for key, value in libero_state_dict.items():
            # DataParallelによる'module.'プレフィックスを除去
            clean_key = key.replace('module.', '') if key.startswith('module.') else key
            
            if clean_key in model_state_dict:
                if model_state_dict[clean_key].shape == value.shape:
                    model_state_dict[clean_key] = value
                    updated_keys.append(clean_key)
                else:
                    print(f"  ⚠ Shape mismatch for {clean_key}: model={model_state_dict[clean_key].shape}, checkpoint={value.shape}")
            else:
                print(f"  ⚠ Key not found in model: {clean_key}")
        
        # 更新されたstate_dictをロード
        self.model.load_state_dict(model_state_dict)
        
        print(f"Updated {len(updated_keys)} parameters from LIBERO checkpoint")
        print("Key components loaded:")
        
        # どの部分が更新されたかを表示
        action_head_keys = [k for k in updated_keys if 'action_head' in k]
        qformer_keys = [k for k in updated_keys if 'qformer' in k]
        other_keys = [k for k in updated_keys if 'action_head' not in k and 'qformer' not in k]
        
        print(f"  Action Head: {len(action_head_keys)} parameters")
        print(f"  Q-Former: {len(qformer_keys)} parameters")
        print(f"  Other: {len(other_keys)} parameters")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully")
        if 'epoch' in checkpoint:
            print(f"  LIBERO Epoch: {checkpoint['epoch']}")
        if 'best_loss' in checkpoint:
            print(f"  LIBERO Best Loss: {checkpoint['best_loss']:.4f}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        画像を前処理
        
        Args:
            image: (H, W, 3) numpy array or PIL Image
        
        Returns:
            preprocessed image tensor (1, 3, 128, 128)
        """
        if isinstance(image, np.ndarray):
            # NumPy配列からPIL Imageに変換
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # 前処理を適用
        image_tensor = self.transform(image)
        
        # バッチ次元を追加
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor
    
    def predict_action(
        self, 
        image: Union[np.ndarray, Image.Image],
        instruction: str,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        単一画像からアクションを予測
        
        Args:
            image: ロボットカメラ画像
            instruction: 言語指示 (例: "pick up the black bowl")
            return_numpy: numpy配列として返すか
        
        Returns:
            action: 7次元アクション [x, y, z, roll, pitch, yaw, gripper]
        """
        # 画像を前処理
        image_tensor = self.preprocess_image(image).to(self.device)
        
        # 推論
        with torch.no_grad():
            action = self.model(image_tensor, [instruction])
        
        # numpy配列に変換
        if return_numpy:
            action = action.squeeze(0).cpu().numpy()
        else:
            action = action.squeeze(0)
        
        return action
    
    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        instructions: List[str],
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        バッチ推論
        
        Args:
            images: 画像のリスト
            instructions: 言語指示のリスト
            return_numpy: numpy配列として返すか
        
        Returns:
            actions: (B, 7) アクション配列
        """
        # 画像を前処理してバッチ化
        image_tensors = []
        for image in images:
            image_tensor = self.preprocess_image(image)
            image_tensors.append(image_tensor)
        
        batch_images = torch.cat(image_tensors, dim=0).to(self.device)
        
        # 推論
        with torch.no_grad():
            actions = self.model(batch_images, instructions)
        
        # numpy配列に変換
        if return_numpy:
            actions = actions.cpu().numpy()
        
        return actions
    
    def interpret_action(self, action: np.ndarray) -> dict:
        """
        アクションを人間が読める形式に変換
        
        Args:
            action: 7次元アクション
        
        Returns:
            dict with action components
        """
        return {
            'position': {
                'x': float(action[0]),
                'y': float(action[1]),
                'z': float(action[2])
            },
            'orientation': {
                'roll': float(action[3]),
                'pitch': float(action[4]),
                'yaw': float(action[5])
            },
            'gripper': float(action[6])
        }


def demo_inference(checkpoint_path: str):
    """デモ推論"""
    # 推論エンジンを初期化
    inference = VLAInference(checkpoint_path)
    
    print("\n" + "="*60)
    print("VLA Inference Demo")
    print("="*60)
    
    # ダミー画像でテスト
    dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    instruction = "pick up the black bowl and place it on the plate"
    
    print(f"\nInstruction: '{instruction}'")
    print("Predicting action...")
    
    # アクション予測
    action = inference.predict_action(dummy_image, instruction)
    
    print("\nPredicted Action:")
    print(f"  Raw: {action}")
    
    # 解釈
    interpreted = inference.interpret_action(action)
    print("\nInterpreted Action:")
    print(f"  Position: x={interpreted['position']['x']:.3f}, "
          f"y={interpreted['position']['y']:.3f}, "
          f"z={interpreted['position']['z']:.3f}")
    print(f"  Orientation: roll={interpreted['orientation']['roll']:.3f}, "
          f"pitch={interpreted['orientation']['pitch']:.3f}, "
          f"yaw={interpreted['orientation']['yaw']:.3f}")
    print(f"  Gripper: {interpreted['gripper']:.3f}")
    
    # バッチ推論のテスト
    print("\n" + "="*60)
    print("Batch Inference Test")
    print("="*60)
    
    batch_images = [dummy_image for _ in range(4)]
    batch_instructions = [
        "pick up the red block",
        "move to the left",
        "place on the table",
        "open the gripper"
    ]
    
    print(f"Batch size: {len(batch_images)}")
    batch_actions = inference.predict_batch(batch_images, batch_instructions)
    
    print("\nBatch Predictions:")
    for i, (instruction, action) in enumerate(zip(batch_instructions, batch_actions)):
        print(f"\n{i+1}. '{instruction}'")
        print(f"   Action: {action}")


def interactive_inference(checkpoint_path: str):
    """インタラクティブ推論モード"""
    inference = VLAInference(checkpoint_path)
    
    print("\n" + "="*60)
    print("VLA Interactive Inference")
    print("="*60)
    print("Enter language instructions to get predicted actions")
    print("Type 'quit' to exit")
    print("="*60 + "\n")
    
    # ダミー画像（実際のロボット環境では、カメラからの画像を使用）
    dummy_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    
    while True:
        instruction = input("\nInstruction: ").strip()
        
        if instruction.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not instruction:
            continue
        
        # アクション予測
        action = inference.predict_action(dummy_image, instruction)
        
        # 結果表示
        interpreted = inference.interpret_action(action)
        print("\nPredicted Action:")
        print(f"  Position: ({interpreted['position']['x']:.3f}, "
              f"{interpreted['position']['y']:.3f}, "
              f"{interpreted['position']['z']:.3f})")
        print(f"  Orientation: ({interpreted['orientation']['roll']:.3f}, "
              f"{interpreted['orientation']['pitch']:.3f}, "
              f"{interpreted['orientation']['yaw']:.3f})")
        print(f"  Gripper: {interpreted['gripper']:.3f}")


def test_with_real_data(checkpoint_path: str, data_dir: str):
    """実際のLIBEROデータを使った推論テスト"""
    inference = VLAInference(checkpoint_path)
    
    print("\n" + "="*60)
    print("Testing with Real LIBERO Data")
    print("="*60)
    
    # LIBEROデータセットから画像を読み込む
    data_path = Path(data_dir)
    hdf5_files = sorted(data_path.glob("*_demo.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}")
        return
    
    print(f"Found {len(hdf5_files)} tasks")
    
    # 最初のファイルからサンプルを取得
    sample_file = hdf5_files[0]
    print(f"\nTesting with: {sample_file.name}")
    
    with h5py.File(sample_file, 'r') as f:
        # タスク情報を取得
        if 'data' in f:
            demo_keys = list(f['data'].keys())
            print(f"Found {len(demo_keys)} demonstrations")
            
            if demo_keys:
                demo = f[f'data/{demo_keys[0]}']
                
                # 最初の画像を取得
                if 'obs/agentview_rgb' in demo:
                    image = demo['obs/agentview_rgb'][0]  # (H, W, 3)
                    instruction = demo.attrs.get('lang', 'pick up the object')
                    
                    print(f"\nInstruction: '{instruction}'")
                    print(f"Image shape: {image.shape}")
                    print(f"Image value range: [{image.min()}, {image.max()}]")
                    
                    # アクション予測
                    action = inference.predict_action(image, instruction)
                    
                    # 実際のアクションと比較
                    if 'actions' in demo:
                        gt_action = demo['actions'][0]
                        
                        print(f"\nPredicted Action: {action}")
                        print(f"Ground Truth Action: {gt_action}")
                        
                        # 各次元の分析
                        print(f"\nDimension-wise Analysis:")
                        dims = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
                        for i, dim in enumerate(dims):
                            pred_val = action[i]
                            gt_val = gt_action[i]
                            error = abs(pred_val - gt_val)
                            print(f"  {dim:8}: pred={pred_val:8.4f}, gt={gt_val:8.4f}, error={error:8.4f}")
                        
                        l2_error = np.linalg.norm(action - gt_action)
                        print(f"\nL2 Error: {l2_error:.4f}")
                        
                        # 複数のサンプルで統計を取る
                        print(f"\n" + "="*50)
                        print("Multi-sample Statistics")
                        print("="*50)
                        
                        num_samples = min(10, len(demo['actions']))
                        errors = []
                        pred_actions = []
                        gt_actions = []
                        
                        for i in range(num_samples):
                            sample_image = demo['obs/agentview_rgb'][i]
                            sample_action_pred = inference.predict_action(sample_image, instruction)
                            sample_action_gt = demo['actions'][i]
                            
                            sample_error = np.linalg.norm(sample_action_pred - sample_action_gt)
                            errors.append(sample_error)
                            pred_actions.append(sample_action_pred)
                            gt_actions.append(sample_action_gt)
                        
                        pred_actions = np.array(pred_actions)
                        gt_actions = np.array(gt_actions)
                        errors = np.array(errors)
                        
                        print(f"Samples tested: {num_samples}")
                        print(f"Average L2 Error: {errors.mean():.4f} ± {errors.std():.4f}")
                        print(f"Min L2 Error: {errors.min():.4f}")
                        print(f"Max L2 Error: {errors.max():.4f}")
                        
                        # 次元ごとの統計
                        print(f"\nDimension-wise Statistics:")
                        for i, dim in enumerate(dims):
                            pred_mean = pred_actions[:, i].mean()
                            pred_std = pred_actions[:, i].std()
                            gt_mean = gt_actions[:, i].mean()
                            gt_std = gt_actions[:, i].std()
                            mae = np.abs(pred_actions[:, i] - gt_actions[:, i]).mean()
                            
                            print(f"  {dim:8}: pred={pred_mean:6.3f}±{pred_std:5.3f}, "
                                  f"gt={gt_mean:6.3f}±{gt_std:5.3f}, mae={mae:6.3f}")
                        
                else:
                    print("No 'obs/agentview_rgb' found in demo")
        else:
            print("No 'data' group found in HDF5 file")


def debug_model(checkpoint_path: str, data_dir: str):
    """モデルのデバッグ分析"""
    print("\n" + "="*60)
    print("VLA Model Debug Analysis")
    print("="*60)
    print("⚠️  Warning: Using modified model code with old checkpoint!")
    print("   The model architecture has been modified to use both vision and text.")
    print("   For full benefits, the model should be retrained.")
    print("="*60)
    
    inference = VLAInference(checkpoint_path)
    
    # 1. モデルの重みを確認
    print("\n1. Model Parameters Analysis:")
    total_params = sum(p.numel() for p in inference.model.parameters())
    trainable_params = sum(p.numel() for p in inference.model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # 最後の層の重みを確認
    if hasattr(inference.model, 'action_head'):
        action_head = inference.model.action_head
        print(f"  Action head type: {type(action_head)}")
        if hasattr(action_head, 'weight'):
            weight_mean = action_head.weight.mean().item()
            weight_std = action_head.weight.std().item()
            print(f"  Action head weight: mean={weight_mean:.6f}, std={weight_std:.6f}")
    
    # 2. 異なる画像での予測をテスト
    print("\n2. Prediction Variability Test:")
    test_images = []
    
    # ランダム画像
    random_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    test_images.append(("Random", random_img))
    
    # 黒画像
    black_img = np.zeros((128, 128, 3), dtype=np.uint8)
    test_images.append(("Black", black_img))
    
    # 白画像
    white_img = np.ones((128, 128, 3), dtype=np.uint8) * 255
    test_images.append(("White", white_img))
    
    # グラデーション画像
    grad_img = np.zeros((128, 128, 3), dtype=np.uint8)
    for i in range(128):
        grad_img[i, :, :] = i * 2
    test_images.append(("Gradient", grad_img))
    
    instruction = "pick up the black bowl"
    predictions = []
    
    for name, img in test_images:
        pred = inference.predict_action(img, instruction)
        predictions.append(pred)
        print(f"  {name:10}: {pred}")
    
    # 予測の分散を計算
    predictions = np.array(predictions)
    pred_std = predictions.std(axis=0)
    print(f"\n  Prediction std across images: {pred_std}")
    print(f"  Total variation: {pred_std.sum():.6f}")
    
    # 3. 異なる指示での予測をテスト
    print("\n3. Instruction Variability Test:")
    instructions = [
        "pick up the red block",
        "move to the left", 
        "place on the table",
        "open the gripper",
        "close the gripper",
        "move up",
        "move down"
    ]
    
    base_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    inst_predictions = []
    
    for inst in instructions:
        pred = inference.predict_action(base_img, inst)
        inst_predictions.append(pred)
        print(f"  '{inst}': {pred}")
    
    # 指示間の分散を計算
    inst_predictions = np.array(inst_predictions)
    inst_std = inst_predictions.std(axis=0)
    print(f"\n  Prediction std across instructions: {inst_std}")
    print(f"  Total variation: {inst_std.sum():.6f}")
    
    # 4. 再学習の提案
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("The model architecture has been fixed to properly use both vision and text.")
    print("However, the current checkpoint was trained with the old architecture.")
    print("\nTo get significantly better results, you should:")
    print("1. Retrain the model with the fixed architecture")
    print("2. Use the command: python train_libero.py --epochs 30")
    print("3. This will create a new checkpoint that properly utilizes both modalities")
    print("\nThe current model may show some improvement but will be limited")
    print("by the mismatch between training and inference architectures.")


def analyze_model_internals(checkpoint_path: str, data_dir: str):
    """モデルの内部状態を詳細に分析"""
    inference = VLAInference(checkpoint_path)
    
    print("\n" + "="*80)
    print("DETAILED MODEL INTERNAL ANALYSIS")
    print("="*80)
    
    # 1. Vision Encoderの分析
    print("\n1. Vision Encoder Analysis:")
    print("="*50)
    
    # テスト用の異なる画像を作成
    test_images = {
        "random": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
        "black": np.zeros((128, 128, 3), dtype=np.uint8),
        "white": np.ones((128, 128, 3), dtype=np.uint8) * 255,
        "red": np.zeros((128, 128, 3), dtype=np.uint8),
        "green": np.zeros((128, 128, 3), dtype=np.uint8),
        "blue": np.zeros((128, 128, 3), dtype=np.uint8)
    }
    
    # 色を設定
    test_images["red"][:, :, 0] = 255
    test_images["green"][:, :, 1] = 255
    test_images["blue"][:, :, 2] = 255
    
    vision_outputs = {}
    
    print("Vision encoder outputs for different images:")
    with torch.no_grad():
        for name, img in test_images.items():
            img_tensor = inference.preprocess_image(img).to(inference.device)
            
            # Vision encoderの出力を取得
            vision_features = inference.model.vision_encoder(img_tensor)
            
            # 統計を計算
            mean_val = vision_features.mean().item()
            std_val = vision_features.std().item()
            min_val = vision_features.min().item()
            max_val = vision_features.max().item()
            
            vision_outputs[name] = vision_features.cpu().numpy()
            
            print(f"  {name:8}: shape={vision_features.shape}, "
                  f"mean={mean_val:8.4f}, std={std_val:8.4f}, "
                  f"range=[{min_val:7.3f}, {max_val:7.3f}]")
    
    # Vision特徴量の差異を分析
    print("\nVision feature differences:")
    baseline_features = vision_outputs["random"]
    for name, features in vision_outputs.items():
        if name != "random":
            diff = np.abs(features - baseline_features).mean()
            print(f"  {name:8} vs random: mean_abs_diff = {diff:.6f}")
    
    # 2. Q-Former分析
    print("\n2. Q-Former Analysis:")
    print("="*50)
    
    qformer_outputs = {}
    print("Q-Former outputs for different images:")
    
    with torch.no_grad():
        for name, img in test_images.items():
            img_tensor = inference.preprocess_image(img).to(inference.device)
            
            # Vision -> Q-Former
            vision_features = inference.model.vision_encoder(img_tensor)
            qformer_features = inference.model.qformer(vision_features)
            
            mean_val = qformer_features.mean().item()
            std_val = qformer_features.std().item()
            
            qformer_outputs[name] = qformer_features.cpu().numpy()
            
            print(f"  {name:8}: shape={qformer_features.shape}, "
                  f"mean={mean_val:8.4f}, std={std_val:8.4f}")
    
    # Q-Former特徴量の差異を分析
    print("\nQ-Former feature differences:")
    baseline_qformer = qformer_outputs["random"]
    for name, features in qformer_outputs.items():
        if name != "random":
            diff = np.abs(features - baseline_qformer).mean()
            print(f"  {name:8} vs random: mean_abs_diff = {diff:.6f}")
    
    # 3. LLM Hidden State分析
    print("\n3. LLM Hidden State Analysis:")
    print("="*50)
    
    llm_outputs = {}
    instruction = "pick up the black bowl"
    
    print(f"LLM hidden states for instruction: '{instruction}'")
    
    with torch.no_grad():
        for name, img in test_images.items():
            img_tensor = inference.preprocess_image(img).to(inference.device)
            
            # Vision処理
            vision_tokens = inference.model.vision_encoder(img_tensor)
            qtoken = inference.model.qformer(vision_tokens)
            
            # 言語指示の処理
            encoded = inference.model.tokenizer(
                [instruction],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(inference.device)
            input_embeds = inference.model.llm.get_input_embeddings()(encoded.input_ids)
            
            # BOS token
            bos_ids = torch.tensor([inference.model.bos]).to(inference.device)
            bos_emb = inference.model.llm.get_input_embeddings()(bos_ids)
            bos_emb = bos_emb.expand(qtoken.size(0), -1, -1)
            
            # 結合
            fused = torch.cat([bos_emb, qtoken, input_embeds], dim=1)
            
            # LLM forward
            outputs = inference.model.llm(inputs_embeds=fused)
            hidden_states = outputs.last_hidden_state
            
            mean_val = hidden_states.mean().item()
            std_val = hidden_states.std().item()
            
            llm_outputs[name] = hidden_states.cpu().numpy()
            
            print(f"  {name:8}: shape={hidden_states.shape}, "
                  f"mean={mean_val:8.4f}, std={std_val:8.4f}")
    
    # LLM隠れ状態の差異を分析
    print("\nLLM hidden state differences:")
    baseline_llm = llm_outputs["random"]
    for name, features in llm_outputs.items():
        if name != "random":
            diff = np.abs(features - baseline_llm).mean()
            print(f"  {name:8} vs random: mean_abs_diff = {diff:.6f}")
    
    # 4. Action Head分析
    print("\n4. Action Head Analysis:")
    print("="*50)
    
    final_actions = {}
    
    print("Final action predictions:")
    with torch.no_grad():
        for name, img in test_images.items():
            action = inference.predict_action(img, instruction)
            final_actions[name] = action
            print(f"  {name:8}: {action}")
    
    # アクションの差異を分析
    print("\nAction differences:")
    baseline_action = final_actions["random"]
    for name, action in final_actions.items():
        if name != "random":
            diff = np.abs(action - baseline_action).mean()
            max_diff = np.abs(action - baseline_action).max()
            print(f"  {name:8} vs random: mean_abs_diff = {diff:.6f}, max_abs_diff = {max_diff:.6f}")
    
    # 5. Gradient分析（バックワードパスをチェック）
    print("\n5. Gradient Flow Analysis:")
    print("="*50)
    
    inference.model.train()  # 勾配計算のために訓練モードに
    
    img_tensor = inference.preprocess_image(test_images["random"]).to(inference.device)
    img_tensor.requires_grad_(True)
    
    # フォワードパス
    action_pred = inference.model(img_tensor, [instruction])
    
    # 損失を計算（ダミーターゲット）
    target = torch.zeros_like(action_pred)
    loss = torch.nn.functional.mse_loss(action_pred, target)
    
    # バックワード
    loss.backward()
    
    # 入力画像の勾配をチェック
    if img_tensor.grad is not None:
        grad_mean = img_tensor.grad.abs().mean().item()
        grad_max = img_tensor.grad.abs().max().item()
        print(f"  Input image gradient: mean={grad_mean:.8f}, max={grad_max:.8f}")
        
        if grad_mean < 1e-8:
            print("  ⚠️  WARNING: Very small gradients detected!")
            print("     This suggests the model is not learning from visual input")
    else:
        print("  ⚠️  ERROR: No gradient computed for input image!")
    
    # モデルのパラメータの勾配をチェック
    vision_grad_norm = 0
    action_head_grad_norm = 0
    
    for name, param in inference.model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if 'vision_encoder' in name:
                vision_grad_norm += grad_norm
            elif 'action_head' in name:
                action_head_grad_norm += grad_norm
    
    print(f"  Vision encoder gradient norm: {vision_grad_norm:.8f}")
    print(f"  Action head gradient norm: {action_head_grad_norm:.8f}")
    
    inference.model.eval()  # 評価モードに戻す
    
    # 6. 結論と推奨事項
    print("\n" + "="*80)
    print("ANALYSIS CONCLUSIONS")
    print("="*80)
    
    # Vision特徴量の変動をチェック
    vision_variations = []
    baseline_features = vision_outputs["random"]
    for name, features in vision_outputs.items():
        if name != "random":
            diff = np.abs(features - baseline_features).mean()
            vision_variations.append(diff)
    
    max_vision_var = max(vision_variations) if vision_variations else 0
    
    if max_vision_var < 1e-4:
        print("❌ CRITICAL ISSUE: Vision encoder shows minimal response to different images")
        print("   This suggests the vision encoder is not properly processing visual input")
    elif max_vision_var < 1e-2:
        print("⚠️  WARNING: Vision encoder shows low sensitivity to image changes")
    else:
        print("✅ Vision encoder shows reasonable sensitivity to image changes")
    
    # アクションの変動をチェック
    action_variations = []
    baseline_action = final_actions["random"]
    for name, action in final_actions.items():
        if name != "random":
            diff = np.abs(action - baseline_action).mean()
            action_variations.append(diff)
    
    max_action_var = max(action_variations) if action_variations else 0
    
    if max_action_var < 1e-6:
        print("❌ CRITICAL ISSUE: Actions are identical regardless of input image")
        print("   The model is completely ignoring visual input")
    elif max_action_var < 1e-3:
        print("⚠️  WARNING: Actions show very low variation with image changes")
    else:
        print("✅ Actions show reasonable variation with image changes")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="VLA Model Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "interactive", "test", "debug", "analyze"],
        default="demo",
        help="Inference mode"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to LIBERO data directory (required for 'test', 'debug', and 'analyze' modes)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demo_inference(args.checkpoint)
    elif args.mode == "interactive":
        interactive_inference(args.checkpoint)
    elif args.mode == "test":
        if args.data_dir is None:
            print("Error: --data-dir is required for 'test' mode")
            return
        test_with_real_data(args.checkpoint, args.data_dir)
    elif args.mode == "debug":
        if args.data_dir is None:
            print("Error: --data-dir is required for 'debug' mode")
            return
        debug_model(args.checkpoint, args.data_dir)
    elif args.mode == "analyze":
        if args.data_dir is None:
            print("Error: --data-dir is required for 'analyze' mode")
            return
        analyze_model_internals(args.checkpoint, args.data_dir)


if __name__ == "__main__":
    main()
