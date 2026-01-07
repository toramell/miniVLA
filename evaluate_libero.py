"""
evaluate_libero.py

学習済みVLAモデルをLIBEROデータセットで評価
- タスク別の性能分析
- アクション次元ごとの誤差分析
- 予測結果の可視化
- WandBへの結果アップロード（オプション）
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from collections import defaultdict
from sklearn.metrics import r2_score, accuracy_score

# モデル定義ファイルのインポート（パスが通っていることを前提）
from vla_robot_policy import VLARobotPolicy
from libero_dataset import LIBERODataset, libero_collate_fn

try:
    import wandb
except ImportError:
    wandb = None


class LIBEROEvaluator:
    """LIBERO用の評価クラス"""
    def __init__(self, model, device, tokenizer):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.model.eval()
        
        # アクション次元の名前
        self.action_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    
    def evaluate_detailed(self, dataloader):
        """詳細な評価"""
        task_results = defaultdict(lambda: {
            'predictions': [],
            'ground_truths': [],
            'errors': []
        })
        
        all_predictions = []
        all_ground_truths = []
        
        print("Evaluating model...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['images'].to(self.device)
                actions_gt = batch['actions'].to(self.device)
                instructions = batch['instructions']
                task_names = batch['task_names']
                
                # --- トークン化処理（学習時と同じロジック） ---
                tokenized = self.tokenizer(
                    instructions, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                # 予測
                actions_pred = self.model(
                    images, 
                    tokenized.input_ids, 
                    tokenized.attention_mask
                )

                # 1. アーム部分 (-1~1) はそのまま
                arm_pred = actions_pred[:, :6]
                
                # 2. グリッパー部分 (Logits) を Sigmoid で 0~1 にし、さらに -1 or 1 に変換
                gripper_logits = actions_pred[:, 6:]
                gripper_prob = torch.sigmoid(gripper_logits)
                # 確率 > 0.5 なら 1 (閉じる), それ以外なら -1 (開く)
                gripper_action = torch.where(gripper_prob > 0.5, torch.tensor(1.0).to(self.device), torch.tensor(-1.0).to(self.device))
                
                # 3. 結合して元の形に戻す
                actions_pred_final = torch.cat([arm_pred, gripper_action], dim=-1)
                
                # CPUに移動 (actions_pred ではなく actions_pred_final を使う)
                actions_pred_np = actions_pred_final.cpu().numpy()
                actions_gt_np = actions_gt.cpu().numpy()
                
                # タスク別に結果を保存
                for i, task_name in enumerate(task_names):
                    task_results[task_name]['predictions'].append(actions_pred_np[i])
                    task_results[task_name]['ground_truths'].append(actions_gt_np[i])
                    task_results[task_name]['errors'].append(
                        np.abs(actions_pred_np[i] - actions_gt_np[i])
                    )
                
                all_predictions.append(actions_pred_np)
                all_ground_truths.append(actions_gt_np)
        
        # 配列に変換
        all_predictions = np.vstack(all_predictions)
        all_ground_truths = np.vstack(all_ground_truths)
        
        for task_name in task_results:
            task_results[task_name]['predictions'] = np.array(task_results[task_name]['predictions'])
            task_results[task_name]['ground_truths'] = np.array(task_results[task_name]['ground_truths'])
            task_results[task_name]['errors'] = np.array(task_results[task_name]['errors'])
        
        return all_predictions, all_ground_truths, task_results
    
    def compute_metrics(self, predictions, ground_truths):
        """メトリクスを計算"""
        mse = np.mean((predictions - ground_truths) ** 2)
        mae = np.mean(np.abs(predictions - ground_truths))
        rmse = np.sqrt(mse)
        
        # 各次元ごとのMAE
        mae_per_dim = np.mean(np.abs(predictions - ground_truths), axis=0)
        
        # 決定係数 (R2 Score) - 1に近いほど予測が完璧
        r2_per_dim = []
        for i in range(7):
            r2 = r2_score(ground_truths[:, i], predictions[:, i])
            r2_per_dim.append(r2)
            
        # グリッパーの正解率（二値化して判定: 0以上なら閉、未満なら開など）
        # LIBEROのグリッパー値域によりますが、ここでは正負の符号が一致しているかで簡易判定
        gripper_pred_binary = (predictions[:, 6] > 0).astype(int)
        gripper_gt_binary = (ground_truths[:, 6] > 0).astype(int)
        gripper_accuracy = accuracy_score(gripper_gt_binary, gripper_pred_binary)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mae_per_dim': mae_per_dim,
            'r2_per_dim': r2_per_dim,
            'gripper_accuracy': gripper_accuracy
        }
    
    def print_results(self, metrics, task_results=None):
        """結果を表示"""
        print("\n" + "="*60)
        print("Overall Evaluation Results")
        print("="*60)
        print(f"MSE:  {metrics['mse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"Gripper Accuracy: {metrics['gripper_accuracy']:.2%}")
        
        print("\nPer-Dimension Metrics:")
        print(f"{'Dim':<10} {'MAE':<10} {'R2 Score':<10}")
        print("-" * 30)
        for i, name in enumerate(self.action_names):
            print(f"{name:<10} {metrics['mae_per_dim'][i]:.4f}     {metrics['r2_per_dim'][i]:.4f}")
        
        if task_results:
            print("\n" + "="*60)
            print("Per-Task Results (Top 5 Worst MAE)")
            print("="*60)
            # MAEが悪い順にソート
            sorted_tasks = sorted(
                task_results.items(),
                key=lambda x: np.mean(x[1]['errors']),
                reverse=True
            )
            
            for task_name, results in sorted_tasks[:5]:
                task_mae = np.mean(results['errors'])
                print(f"{task_name}: MAE = {task_mae:.4f}")

    def plot_prediction_vs_gt(self, predictions, ground_truths, save_path='prediction_vs_gt.png', num_samples=2000):
        """予測値 vs 真値の散布図（改良版）"""
        if len(predictions) > num_samples:
            indices = np.random.choice(len(predictions), num_samples, replace=False)
            predictions = predictions[indices]
            ground_truths = ground_truths[indices]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, name in enumerate(self.action_names):
            ax = axes[i]
            # 散布図
            ax.scatter(ground_truths[:, i], predictions[:, i], alpha=0.2, s=5, color='blue')
            
            # 理想線 (y=x)
            min_val = min(ground_truths[:, i].min(), predictions[:, i].min())
            max_val = max(ground_truths[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Ideal')
            
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Prediction')
            ax.grid(True, alpha=0.3)
            
            # 軸の範囲を揃える
            margin = (max_val - min_val) * 0.1
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            
        axes[7].axis('off')
        
        # 全体の情報を表示
        info_text = (
            f"Overall MSE: {np.mean((predictions - ground_truths)**2):.4f}\n"
            f"Overall MAE: {np.mean(np.abs(predictions - ground_truths)):.4f}"
        )
        axes[7].text(0.1, 0.5, info_text, fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Prediction vs GT plot saved to {save_path}")
        plt.close()
        
        # WandBへのアップロード
        if wandb.run is not None:
            wandb.log({"prediction_vs_gt": wandb.Image(save_path)})


def main(args):
    # WandB初期化（オプション）
    if args.use_wandb and wandb:
        wandb.init(project="vla-libero-eval", name="evaluation_run")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 画像変換
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Loading LIBERO dataset from {args.data_dir}")
    dataset = LIBERODataset(args.data_dir, transform=transform)
    
    # 評価用なのでShuffle=False
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=libero_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Loading VLA model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # モデルの初期化
    model = VLARobotPolicy(
        pretrained_checkpoint=None,
        freeze_vision=False,
        freeze_llm=True
    )
    
    # DataParallelで保存されたモデル（module.xxx）のキーを修正してロード
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        # まず torch.compile の接頭辞を削除
        if k.startswith('_orig_mod.'):
            k = k[10:]  # '_orig_mod.' は10文字
            
        # 次に DataParallel の接頭辞を削除
        if k.startswith('module.'):
            k = k[7:]   # 'module.' は7文字
            
        new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully.")
    
    # 評価実行
    # モデルからトークナイザーを取得してEvaluatorに渡す
    evaluator = LIBEROEvaluator(model, device, model.tokenizer)
    predictions, ground_truths, task_results = evaluator.evaluate_detailed(dataloader)
    
    metrics = evaluator.compute_metrics(predictions, ground_truths)
    evaluator.print_results(metrics, task_results if args.per_task else None)
    
    # WandBにメトリクスを送信
    if args.use_wandb and wandb:
        wandb.log(metrics)
    
    if args.plot:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        evaluator.plot_prediction_vs_gt(
            predictions, ground_truths,
            save_path=os.path.join(args.output_dir, 'prediction_vs_gt.png')
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLA on LIBERO dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--per_task", action="store_true", help="Show results per task")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--use_wandb", action="store_true", help="Log results to WandB")
    
    args = parser.parse_args()
    main(args)