# split.py - Stratified 70/15/15 split of the Garbage Detection Dataset

# ---------- Step 1: imports ----------
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMAGES = Path("GarbageDataset/images")
LABELS = Path("GarbageDataset/labels")
OUT    = Path("GarbageSplit")

# ---------- Step 2: count boxes per class ----------
counts = Counter()
for file in LABELS.glob("*.txt"):
    for line in file.read_text().splitlines():
        if not line.strip():
            continue
        cls = int(line.split()[0])
        counts[cls] += 1
print("Boxes per class:", counts)

# ---------- Step 3: group images by their rarest class ----------
groups = defaultdict(list)
for file in LABELS.glob("*.txt"):
    image_classes = set()
    for line in file.read_text().splitlines():
        if not line.strip():
            continue
        image_classes.add(int(line.split()[0]))

    if image_classes:
        rarest = min(image_classes, key=lambda c: counts[c])
    else:
        rarest = -1                     
    groups[rarest].append(file.stem)

for g, items in sorted(groups.items()):
    print("group", g, len(items))

# ---------- Step 4: shuffle each group (seed = reproducible) ----------
random.seed(42)
for g, items in groups.items():
    random.shuffle(items)

# ---------- Step 5: split each group 70 / 15 / 15 ----------
train, val, test = [], [], []
for g, items in groups.items():
    n = len(items)
    n_train = int(n * 0.7)
    n_val   = int(n * 0.15)
    train += items[:n_train]
    val   += items[n_train:n_train + n_val]
    test  += items[n_train + n_val:]       
print("train/val/test:", len(train), len(val), len(test))

# ---------- Step 6: copy files into the YOLO folder layout ----------
splits = {"train": train, "val": val, "test": test}

for split_name, names in splits.items():
    (OUT / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (OUT / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    for name in names:
        image_file = list(IMAGES.glob(name + ".*"))[0]  
        label_file = LABELS / (name + ".txt")            

        shutil.copy(image_file, OUT / "images" / split_name)
        shutil.copy(label_file, OUT / "labels" / split_name)

    print(split_name, "copied:", len(names))