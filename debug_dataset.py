import torch
from datasets import load_dataset
from sample_vla import MiniVLA
from ssv2_dataset import UCF101MiniVLADataset
from collate_ucf101 import collate_ucf101

def debug_dataset():
    print("=== UCF101データセット構造確認 ===")
    
    # 小さなサンプルでテスト
    ds_test = load_dataset("flwrlabs/ucf101", split="test[:10]")
    print(f"データセットサイズ: {len(ds_test)}")
    
    # 最初のアイテムを確認
    first_item = ds_test[0]
    print(f"データセットのキー: {first_item.keys()}")
    print(f"ラベル値: {first_item['label']}")
    
    # ラベルの分布を確認
    labels = [item['label'] for item in ds_test]
    unique_labels = set(labels)
    print(f"ユニークなラベル数: {len(unique_labels)}")
    print(f"ラベルの範囲: {min(labels)} - {max(labels)}")
    print(f"最初の10個のラベル: {labels}")
    
    # モデルとデータセットの初期化
    model = MiniVLA(num_actions=101)
    dataset = UCF101MiniVLADataset(ds_test, model.tokenizer)
    
    # 変換後のアイテムを確認
    converted_item = dataset[0]
    print(f"変換後のキー: {converted_item.keys()}")
    print(f"変換後のアクション: {converted_item['actions']}")
    
    # バッチの確認
    batch = [dataset[i] for i in range(3)]
    collated = collate_ucf101(batch, model.tokenizer)
    print(f"バッチのキー: {collated.keys()}")
    print(f"バッチのラベル: {collated['labels']}")
    print(f"バッチサイズ: {collated['images'].shape[0]}")

if __name__ == "__main__":
    debug_dataset()