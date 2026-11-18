"""split.py - Stratified 70/15/15 split of the Garbage Detection Dataset.

Each image is assigned to the group of its rarest class (counted over the
whole dataset); every group is shuffled (seed 42) and cut 70/15/15.
Output: GarbageSplit/{images,labels}/{train,val,test} (YOLO layout).
"""
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMAGES = Path("GarbageDataset/images")
LABELS = Path("GarbageDataset/labels")
OUT = Path("GarbageSplit")
SEED = 42
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15   # test gets the rest


def read_classes(label_file):
    """Return the class id of every box in one label file."""
    return [int(line.split()[0])
            for line in label_file.read_text().splitlines() if line.strip()]


def count_boxes():
    """Count boxes per class over the whole dataset."""
    counts = Counter()
    for file in LABELS.glob("*.txt"):
        counts.update(read_classes(file))
    return counts


def group_by_rarest(counts):
    """Put each image into the group of its rarest class (-1 = no boxes)."""
    groups = defaultdict(list)
    # Note: glob order can differ between systems; sorted() would make the
    # split fully reproducible but was not used for the submitted model.
    for file in LABELS.glob("*.txt"):
        classes = set(read_classes(file))
        rarest = min(classes, key=lambda c: counts[c]) if classes else -1
        groups[rarest].append(file.stem)
    return groups


def split_groups(groups):
    """Shuffle each group and cut it 70/15/15; test takes rounding leftovers."""
    random.seed(SEED)
    train, val, test = [], [], []
    for items in groups.values():
        random.shuffle(items)
        n_train = int(len(items) * TRAIN_RATIO)
        n_val = int(len(items) * VAL_RATIO)
        train += items[:n_train]
        val += items[n_train:n_train + n_val]
        test += items[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def copy_files(splits):
    """Copy images and labels into the YOLO folder layout."""
    for split_name, names in splits.items():
        img_dir = OUT / "images" / split_name
        lbl_dir = OUT / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            image_file = next(IMAGES.glob(name + ".*"))
            shutil.copy(image_file, img_dir)
            shutil.copy(LABELS / (name + ".txt"), lbl_dir)
        print(split_name, "copied:", len(names))


def main():
    counts = count_boxes()
    print("Boxes per class:", counts)
    groups = group_by_rarest(counts)
    for g, items in sorted(groups.items()):
        print("group", g, len(items))
    splits = split_groups(groups)
    print("train/val/test:", *(len(v) for v in splits.values()))
    copy_files(splits)


if __name__ == "__main__":
    main()