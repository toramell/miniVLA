"""
evaluate_libero.py

学習済みVLAモデルをLIBEROデータセットで評価
- タスク別の性能分析
- アクション次元ごとの誤差分析
- 予測結果の可視化
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

from vla_robot_policy import VLARobotPolicy
from libero_dataset import LIBERODataset, libero_collate_fn


class LIBEROEvaluator:
    """LIBERO用の評価クラス"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # アクション次元の名前
        self.action_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    
    def evaluate_detailed(self, dataloader):
        """詳細な評価"""
        # タスク別の結果を保存
        task_results = defaultdict(lambda: {
            'predictions': [],
            'ground_truths': [],
            'errors': []
        })
        
        # 全体の結果
        all_predictions = []
        all_ground_truths = []
        
        print("Evaluating model...")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['images'].to(self.device)
                actions_gt = batch['actions'].to(self.device)
                instructions = batch['instructions']
                task_names = batch['task_names']
                
                # 予測
                actions_pred = self.model(images, instructions)
                
                # CPUに移動
                actions_pred_np = actions_pred.cpu().numpy()
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
        
        # タスク別の結果を配列に変換
        for task_name in task_results:
            task_results[task_name]['predictions'] = np.array(task_results[task_name]['predictions'])
            task_results[task_name]['ground_truths'] = np.array(task_results[task_name]['ground_truths'])
            task_results[task_name]['errors'] = np.array(task_results[task_name]['errors'])
        
        return all_predictions, all_ground_truths, task_results
    
    def compute_metrics(self, predictions, ground_truths):
        """メトリクスを計算"""
        # MSE
        mse = np.mean((predictions - ground_truths) ** 2)
        
        # MAE
        mae = np.mean(np.abs(predictions - ground_truths))
        
        # 各アクション次元ごとのMAE
        mae_per_dim = np.mean(np.abs(predictions - ground_truths), axis=0)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mae_per_dim': mae_per_dim
        }
    
    def print_results(self, metrics, task_results=None):
        """結果を表示"""
        print("\n" + "="*60)
        print("Overall Evaluation Results")
        print("="*60)
        print(f"MSE:  {metrics['mse']:.4f}")
        print(f"MAE:  {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print("\nMAE per action dimension:")
        for i, name in enumerate(self.action_names):
            print(f"  {name:8s}: {metrics['mae_per_dim'][i]:.4f}")
        
        if task_results:
            print("\n" + "="*60)
            print("Per-Task Results")
            print("="*60)
            for task_name, results in task_results.items():
                task_metrics = self.compute_metrics(
                    results['predictions'],
                    results['ground_truths']
                )
                print(f"\n{task_name}:")
                print(f"  Samples: {len(results['predictions'])}")
                print(f"  MAE: {task_metrics['mae']:.4f}")
                print(f"  RMSE: {task_metrics['rmse']:.4f}")
    
    def plot_error_distribution(self, predictions, ground_truths, save_path='error_distribution.png'):
        """誤差の分布を可視化"""
        errors = np.abs(predictions - ground_truths)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, name in enumerate(self.action_names):
            ax = axes[i]
            ax.hist(errors[:, i], bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(f'{name} - MAE: {np.mean(errors[:, i]):.4f}')
            ax.set_xlabel('Absolute Error')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        
        # 全体のMAE
        axes[7].axis('off')
        overall_mae = np.mean(errors)
        axes[7].text(0.5, 0.5, f'Overall MAE:\n{overall_mae:.4f}', 
                    ha='center', va='center', fontsize=20, weight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nError distribution plot saved to {save_path}")
        plt.close()
    
    def plot_prediction_vs_gt(self, predictions, ground_truths, save_path='prediction_vs_gt.png', num_samples=1000):
        """予測値 vs 真値の散布図"""
        # サンプル数を制限
        if len(predictions) > num_samples:
            indices = np.random.choice(len(predictions), num_samples, replace=False)
            predictions = predictions[indices]
            ground_truths = ground_truths[indices]
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, name in enumerate(self.action_names):
            ax = axes[i]
            ax.scatter(ground_truths[:, i], predictions[:, i], alpha=0.3, s=10)
            
            # 理想的な予測線（y=x）
            min_val = min(ground_truths[:, i].min(), predictions[:, i].min())
            max_val = max(ground_truths[:, i].max(), predictions[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            ax.set_title(f'{name}')
            ax.set_xlabel('Ground Truth')
            ax.set_ylabel('Prediction')
            ax.grid(True, alpha=0.3)
            
            # 相関係数
            corr = np.corrcoef(ground_truths[:, i], predictions[:, i])[0, 1]
            ax.text(0.05, 0.95, f'Corr: {corr:.3f}', 
                   transform=ax.transAxes, va='top')
        
        axes[7].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Prediction vs GT plot saved to {save_path}")
        plt.close()


def main(args):
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 画像変換
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # データセットをロード
    print(f"Loading LIBERO dataset from {args.data_dir}")
    dataset = LIBERODataset(args.data_dir, transform=transform)
    
    # DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=libero_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # モデルをロード
    print(f"Loading VLA model from {args.checkpoint}")
    
    # チェックポイントをロード
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # モデルを初期化
    model = VLARobotPolicy(
        pretrained_checkpoint=None,  # 重みは後でロード
        freeze_vision=False,
        freeze_llm=True,
        action_dim=7
    )
    
    # 学習済み重みをロード
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Checkpoint info:")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_loss' in checkpoint:
        print(f"  Best Loss: {checkpoint['best_loss']:.4f}")
    
    # 評価
    evaluator = LIBEROEvaluator(model, device)
    predictions, ground_truths, task_results = evaluator.evaluate_detailed(dataloader)
    
    # メトリクス計算
    metrics = evaluator.compute_metrics(predictions, ground_truths)
    
    # 結果表示
    evaluator.print_results(metrics, task_results if args.per_task else None)
    
    # 可視化
    if args.plot:
        evaluator.plot_error_distribution(
            predictions, ground_truths, 
            save_path=args.output_dir + '/error_distribution.png'
        )
        evaluator.plot_prediction_vs_gt(
            predictions, ground_truths,
            save_path=args.output_dir + '/prediction_vs_gt.png'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLA on LIBERO dataset")
    
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
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--per_task",
        action="store_true",
        help="Show per-task results"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save plots"
    )
    
    args = parser.parse_args()
    
    main(args)
