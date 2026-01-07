import argparse
import torch
import numpy as np
import os
import sys
import imageio
from tqdm import tqdm

# LIBEROのパス設定 (環境に合わせて調整してください)
# sys.path.append("/path/to/LIBERO") 
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from libero_inference import VLAInference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="学習済みモデルのパス")
    parser.add_argument("--task_suite", type=str, default="libero_spatial", help="ベンチマーク名 (libero_spatial, libero_10, etc)")
    parser.add_argument("--task_id", type=int, default=0, help="実行するタスクのID (0-9)")
    parser.add_argument("--num_episodes", type=int, default=5, help="試行回数")
    parser.add_argument("--max_steps", type=int, default=600, help="1エピソードの最大ステップ数")
    parser.add_argument("--save_video", action="store_true", help="実行結果を動画で保存")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print(f"\nLoading Benchmark: {args.task_suite}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    
    # タスクの取得
    task = task_suite.get_task(args.task_id)
    task_name = task.name
    print(f"Selected Task [{args.task_id}]: {task_name}")

    # 環境の初期化
    env_args = {
        "bddl_file_name": os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
        "render_gpu_device_id": 0,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    # 推論モデルのロード
    print("\nInitializing VLA Policy...")
    agent = VLAInference(args.checkpoint, device=args.device)

    # 言語指示（LIBEROのタスク名から取得、または手動設定）
    # LIBEROのタスク名は "pick_up_the_red_block_..." のようになっているので、
    # アンダーバーをスペースに置換して言語指示にします。
    instruction = task_name.replace("_", " ")
    print(f"Instruction: '{instruction}'")

    success_count = 0
    video_frames = []

    print("\nStarting Simulation Loop...")
    
    for episode in range(args.num_episodes):
        env.reset()
        init_states = task_suite.get_task_init_states(args.task_id)
        # ランダムな初期状態を選択
        init_state_id = np.random.randint(0, init_states.shape[0])
        env.set_init_state(init_states[init_state_id])
        
        obs = env.env._get_observations()
        done = False
        steps = 0
        
        # エピソードループ
        pbar = tqdm(total=args.max_steps, desc=f"Ep {episode+1}/{args.num_episodes}")
        
        # 平滑化係数 (0.0 ~ 1.0)
        # 0.0: 平滑化なし (今の状態)
        # 0.8: かなり滑らかになる (推奨)
        alpha = 0.8 
        
        # 前回のアクションを保存する変数
        prev_action = None

        while steps < args.max_steps:
            # 画像の取得 (LIBEROは (H,W,3) で画像を返す)
            # エージェント視点の画像を使用 (agentview_image)
            # 画像は上下逆さまの場合があるので、必要なら np.flipud 等で調整
            # LIBEROの標準は agentview_image
            image = obs["agentview_image"]
            
            # 上下反転（LIBEROの仕様による、学習データに合わせて調整）
            image = image[::-1] 

            # 動画保存用
            if args.save_video and episode == 0:
                video_frames.append(image)

            # アクション推論
            raw_action = agent.predict_action(image, instruction)

            # --- 【追加】Temporal Aggregation (EMA平滑化) ---
            if prev_action is None:
                action = raw_action
            else:
                # 前回の動きを alpha 分だけ残し、新しい動きを (1-alpha) だけ足す
                action = alpha * prev_action + (1 - alpha) * raw_action
            
            # 今回のアクションを保存
            prev_action = action
            
            # シミュレーション実行
            obs, reward, done, info = env.step(action)
            
            steps += 1
            pbar.update(1)
            
            if done:
                break
        
        pbar.close()
        
        if done:
            print(f"  ✅ Episode {episode+1}: SUCCESS!")
            success_count += 1
        else:
            print(f"  ❌ Episode {episode+1}: Failed (Timeout)")

    print("\n" + "="*30)
    print(f"Final Success Rate: {success_count}/{args.num_episodes} ({success_count/args.num_episodes*100:.1f}%)")
    print("="*30)

    # 動画の保存
    if args.save_video and video_frames:
        video_path = f"simulation_{args.task_id}.mp4"
        imageio.mimsave(video_path, video_frames, fps=20)
        print(f"Video saved to {video_path}")
    env.close()

if __name__ == "__main__":
    main()