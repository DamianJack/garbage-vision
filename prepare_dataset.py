from pathlib import Path
import random

# Find and shuffle the image paths in a repeatable order.
dataset = Path(__file__).resolve().parent.parent / "GarbageDataset"
images = sorted((dataset / "images").glob("*.jpg"))
random.seed(42)
random.shuffle(images)

# Divide the shuffled image paths into 70% train, 15% validation, and 15% test.
train_end = int(len(images) * 0.70)
val_end = train_end + int(len(images) * 0.15)
train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]
