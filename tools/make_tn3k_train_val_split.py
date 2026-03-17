import random
from pathlib import Path

SEED = 3407
VAL_RATIO = 0.2

list_dir = Path("/home/ub/pql/Swin-Unet-main/lists/TN3K")
train_file = list_dir / "train.txt"
new_train_file = list_dir / "train_new.txt"
val_file = list_dir / "val.txt"

with open(train_file, "r", encoding="utf-8") as f:
    ids = [line.strip() for line in f if line.strip()]

random.seed(SEED)
random.shuffle(ids)

n_total = len(ids)
n_val = int(n_total * VAL_RATIO)

val_ids = sorted(ids[:n_val])
train_ids = sorted(ids[n_val:])

with open(new_train_file, "w", encoding="utf-8") as f:
    for x in train_ids:
        f.write(x + "\n")

with open(val_file, "w", encoding="utf-8") as f:
    for x in val_ids:
        f.write(x + "\n")

print(f"total={n_total}, train={len(train_ids)}, val={len(val_ids)}")
print(f"write: {new_train_file}")
print(f"write: {val_file}")