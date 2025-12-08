import torch
from datasets import load_dataset
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def analyze_class_distribution():
    print("=== UCF101クラス分布分析 ===")
    
    # 全データセットをロード
    ds_full_train = load_dataset("flwrlabs/ucf101", split="train")
    ds_full_test = load_dataset("flwrlabs/ucf101", split="test")
    
    print(f"全訓練データサイズ: {len(ds_full_train)}")
    print(f"全テストデータサイズ: {len(ds_full_test)}")
    
    # 全データのラベル分布を確認
    all_train_labels = [item['label'] for item in ds_full_train]
    all_test_labels = [item['label'] for item in ds_full_test]
    
    train_counter = Counter(all_train_labels)
    test_counter = Counter(all_test_labels)
    
    print(f"\n=== 全データのクラス分布 ===")
    print(f"訓練データの総クラス数: {len(train_counter)}")
    print(f"テストデータの総クラス数: {len(test_counter)}")
    
    # 各クラスのサンプル数統計
    train_counts = list(train_counter.values())
    test_counts = list(test_counter.values())
    
    print(f"\n訓練データのクラス別サンプル数:")
    print(f"  最小: {min(train_counts)}")
    print(f"  最大: {max(train_counts)}")
    print(f"  平均: {np.mean(train_counts):.2f}")
    print(f"  標準偏差: {np.std(train_counts):.2f}")
    
    print(f"\nテストデータのクラス別サンプル数:")
    print(f"  最小: {min(test_counts)}")
    print(f"  最大: {max(test_counts)}")
    print(f"  平均: {np.mean(test_counts):.2f}")
    print(f"  標準偏差: {np.std(test_counts):.2f}")
    
    # ランダムサンプリングでの分布
    print(f"\n=== ランダムサンプリング後の分布 ===")
    np.random.seed(42)
    train_indices = np.random.choice(len(ds_full_train), size=min(20000, len(ds_full_train)), replace=False)
    test_indices = np.random.choice(len(ds_full_test), size=min(5000, len(ds_full_test)), replace=False)
    
    sampled_train_labels = [all_train_labels[i] for i in train_indices]
    sampled_test_labels = [all_test_labels[i] for i in test_indices]
    
    sampled_train_counter = Counter(sampled_train_labels)
    sampled_test_counter = Counter(sampled_test_labels)
    
    print(f"サンプリング後の訓練データクラス数: {len(sampled_train_counter)}")
    print(f"サンプリング後のテストデータクラス数: {len(sampled_test_counter)}")
    
    sampled_train_counts = list(sampled_train_counter.values())
    sampled_test_counts = list(sampled_test_counter.values())
    
    print(f"\nサンプリング後の訓練データ:")
    print(f"  最小: {min(sampled_train_counts)}")
    print(f"  最大: {max(sampled_train_counts)}")
    print(f"  平均: {np.mean(sampled_train_counts):.2f}")
    print(f"  標準偏差: {np.std(sampled_train_counts):.2f}")
    
    # 不均衡の度合いを計算
    imbalance_ratio = max(sampled_train_counts) / min(sampled_train_counts)
    print(f"  クラス不均衡比: {imbalance_ratio:.2f}:1")
    
    # 層別サンプリングとの比較
    print(f"\n=== 層別サンプリングのシミュレーション ===")
    samples_per_class = 20000 // len(train_counter)  # 各クラスから均等に
    print(f"各クラスから{samples_per_class}サンプルを取得する場合:")
    print(f"  総サンプル数: {samples_per_class * len(train_counter)}")
    print(f"  クラス不均衡比: 1.0:1 (完全にバランス)")

if __name__ == "__main__":
    analyze_class_distribution()