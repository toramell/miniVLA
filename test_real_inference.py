"""
test_real_inference.py

実際のLIBEROデータセットを使ってモデルの推論性能をテスト
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import h5py
from pathlib import Path
import argparse

from vla_robot_policy import VLARobotPolicy
from libero_inference import VLAInference


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
    
    # 各タスクから1つずつサンプルを取得
    test_samples = []
    for hdf5_path in hdf5_files[:5]:  # 最初の5タスクをテスト
        task_name = hdf5_path.stem.replace("_demo", "")
        task_instruction = task_name.replace("_", " ")
        
        # HDF5ファイルから画像を読み込む
        with h5py.File(hdf5_path, 'r') as f:
            data_group = f['data']
            demo_keys = list(data_group.keys())
            
            if demo_keys:
                demo = data_group[demo_keys[0]]  # 最初のデモ
                
                # 画像を取得
                if 'obs' in demo:
                    image = demo['obs']['agentview_rgb'][0]  # 最初のフレーム
                else:
                    image = demo['agentview_image'][0]
                
                # Ground truth アクション
                gt_action = demo['actions'][0]
                
                test_samples.append({
                    'image': image,
                    'instruction': task_instruction,
                    'task_name': task_name,
                    'gt_action': gt_action
                })
    
    print(f"\nTesting {len(test_samples)} samples from real data:")
    print("="*60)
    
    # 各サンプルで推論
    for i, sample in enumerate(test_samples, 1):
        print(f"\n{i}. Task: {sample['task_name']}")
        print(f"   Instruction: '{sample['instruction']}'")
        
        # 予測
        pred_action = inference.predict_action(sample['image'], sample['instruction'])
        gt_action = sample['gt_action']
        
        # 結果表示
        print(f"\n   Ground Truth Action:")
        print(f"     Position: ({gt_action[0]:.3f}, {gt_action[1]:.3f}, {gt_action[2]:.3f})")
        print(f"     Orientation: ({gt_action[3]:.3f}, {gt_action[4]:.3f}, {gt_action[5]:.3f})")
        print(f"     Gripper: {gt_action[6]:.3f}")
        
        print(f"\n   Predicted Action:")
        interpreted = inference.interpret_action(pred_action)
        print(f"     Position: ({interpreted['position']['x']:.3f}, "
              f"{interpreted['position']['y']:.3f}, "
              f"{interpreted['position']['z']:.3f})")
        print(f"     Orientation: ({interpreted['orientation']['roll']:.3f}, "
              f"{interpreted['orientation']['pitch']:.3f}, "
              f"{interpreted['orientation']['yaw']:.3f})")
        print(f"     Gripper: {interpreted['gripper']:.3f}")
        
        # 誤差計算
        error = np.abs(pred_action - gt_action)
        mae = np.mean(error)
        print(f"\n   Error (MAE): {mae:.4f}")
        print(f"     Position Error: {np.mean(error[:3]):.4f}")
        print(f"     Orientation Error: {np.mean(error[3:6]):.4f}")
        print(f"     Gripper Error: {error[6]:.4f}")
    
    print("\n" + "="*60)
    print("Real Data Test Complete!")
    print("="*60)


def test_language_conditioning(checkpoint_path: str, data_dir: str):
    """言語指示による条件付けのテスト"""
    inference = VLAInference(checkpoint_path)
    
    print("\n" + "="*60)
    print("Testing Language Conditioning")
    print("="*60)
    print("Testing if the model produces different actions for different instructions")
    
    # 1つの画像を読み込む
    data_path = Path(data_dir)
    hdf5_files = list(data_path.glob("*_demo.hdf5"))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}")
        return
    
    with h5py.File(hdf5_files[0], 'r') as f:
        data_group = f['data']
        demo = data_group[list(data_group.keys())[0]]
        
        if 'obs' in demo:
            test_image = demo['obs']['agentview_rgb'][0]
        else:
            test_image = demo['agentview_image'][0]
    
    # 異なる指示でテスト
    test_instructions = [
        "pick up the black bowl",
        "place the bowl on the plate",
        "move the bowl to the left",
        "move the bowl to the right",
        "lift the bowl up"
    ]
    
    print("\nUsing the same image with different instructions:")
    print("-"*60)
    
    actions = []
    for i, instruction in enumerate(test_instructions, 1):
        action = inference.predict_action(test_image, instruction)
        actions.append(action)
        
        print(f"\n{i}. Instruction: '{instruction}'")
        print(f"   Action: {action}")
    
    # アクションの多様性を評価
    actions_array = np.array(actions)
    std_devs = np.std(actions_array, axis=0)
    
    print("\n" + "="*60)
    print("Action Diversity Analysis:")
    print(f"  Position Std Dev: ({std_devs[0]:.4f}, {std_devs[1]:.4f}, {std_devs[2]:.4f})")
    print(f"  Orientation Std Dev: ({std_devs[3]:.4f}, {std_devs[4]:.4f}, {std_devs[5]:.4f})")
    print(f"  Gripper Std Dev: {std_devs[6]:.4f}")
    print(f"  Mean Std Dev: {np.mean(std_devs):.4f}")
    
    if np.mean(std_devs) > 0.01:
        print("\n✅ Model shows language conditioning!")
        print("   Different instructions produce different actions")
    else:
        print("\n⚠️  Model may not be properly language-conditioned")
        print("   Actions are very similar across different instructions")
    
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VLA with real LIBERO data")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="libero_best_loss_0.2217.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/toramoto/toramoto/vla/LIBERO/libero/datasets/libero_spatial",
        help="Path to LIBERO dataset directory"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["real_data", "language", "both"],
        default="both",
        help="Which test to run"
    )
    
    args = parser.parse_args()
    
    if args.test in ["real_data", "both"]:
        test_with_real_data(args.checkpoint, args.data_dir)
    
    if args.test in ["language", "both"]:
        test_language_conditioning(args.checkpoint, args.data_dir)
