import numpy as np
import torch
import argparse
import sys

# 推論クラスをインポート
from libero_inference import VLAInference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="学習済みモデルのパス")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("🤖 VLA Vision Sensitivity Test")
    print("="*50)

    # 推論エンジンの初期化
    inference = VLAInference(args.checkpoint)
    
    # 共通の指示
    instruction = "pick up the object"
    print(f"Instruction: '{instruction}'")

    # テスト画像の作成 (128x128)
    images = {
        "Black (Darkness)": np.zeros((128, 128, 3), dtype=np.uint8),
        "White (Bright)  ": np.ones((128, 128, 3), dtype=np.uint8) * 255,
        "Red Color       ": np.zeros((128, 128, 3), dtype=np.uint8),
    }
    # 赤色を作成
    images["Red Color       "][:, :, 0] = 255

    # 結果の保存用
    results = {}

    print("\n--- Predicting Actions ---")
    for name, img in images.items():
        # 推論実行
        action = inference.predict_action(img, instruction)
        results[name] = action
        
        # 結果表示（見やすいように主要な値だけ）
        # x, y, z, gripper
        print(f"Image: {name} -> Action: [x={action[0]:.4f}, y={action[1]:.4f}, z={action[2]:.4f}, grip={action[6]:.4f}]")

    # 差分の検証
    print("\n--- Analysis ---")
    black_act = results["Black (Darkness)"]
    white_act = results["White (Bright)  "]
    
    # 差分（L2ノルム）を計算
    diff = np.linalg.norm(black_act - white_act)
    
    print(f"Difference between Black vs White output: {diff:.6f}")
    
    if diff > 0.001:
        print("\n✅ SUCCESS: The model reacts to visual input!")
        print("   (The output changes depending on what the robot sees)")
    else:
        print("\n❌ WARNING: The model output is identical.")
        print("   (The vision encoder might not be affecting the decision)")

if __name__ == "__main__":
    main()