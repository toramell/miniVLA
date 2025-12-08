from datasets import load_dataset

ds = load_dataset("flwrlabs/ucf101", split="train[:20]")
print(ds[0].keys())
for i in range(10):
    print(i, ds[i].keys())
