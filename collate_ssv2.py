import torch

def collate_ssv2(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    actions = torch.stack([b['action'] for b in batch], dim=0)

    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)

    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'actions': actions
    }