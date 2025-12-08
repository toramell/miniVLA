import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class VLAClassificationDataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = sample["image"]
        if not torch.is_tensor(image):
            image = self.transform(image)

        sample = self.samples[idx]
        text = sample["text"]
        tok = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        action = torch.tensor(sample["action"], dtype=torch.long)

        return {
            "image":image,
            "input_ids":tok.input_ids.squeeze(0),
            "attention_mask":tok.attention_mask.squeeze(0),
            "action":action 
        }