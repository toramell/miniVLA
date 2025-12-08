import torch

def vla_collate(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    actions = torch.stack([b['action'] for b in batch], dim=0)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )

    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': actions  # 'actions'を'labels'に変更
    }