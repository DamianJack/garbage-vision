# make_mosaic.py - Build one example mosaic from 4 real training images (for the documentation)

import random
from pathlib import Path

from PIL import Image, ImageDraw

# ---------- settings ----------
TRAIN_IMAGES = Path("GarbageSplit/images/train")
OUT_FILE = Path("mosaic_example.png")
NAMES = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
COLORS = ['#7f9c8c', '#b08a57', '#4c8bb5', '#8a8f98', '#c46a5a', '#7a68b0']
PICK = ['glass', 'metal', 'paper', 'biodegradable']   # one image per class, in this order
SIZE = 640                                            # final mosaic size (like training)
TILE = SIZE // 2                                      # each image becomes 320 x 320
SEED = 42


def read_boxes(img_path):
    """Return YOLO boxes (cls, x, y, w, h) as fractions 0-1."""
    label_path = Path(str(img_path).replace('/images/', '/labels/')).with_suffix('.txt')
    boxes = []
    for line in label_path.read_text().splitlines():
        if line.strip():
            c, x, y, w, h = line.split()
            boxes.append((int(c), float(x), float(y), float(w), float(h)))
    return boxes


def pick_images():
    """Pick one random training image per class in PICK, with few objects so it stays readable."""
    random.seed(SEED)
    images = sorted(TRAIN_IMAGES.glob("*"))
    random.shuffle(images)
    chosen = []
    for cls_name in PICK:
        for img_path in images:
            if img_path.name.startswith(cls_name) and 1 <= len(read_boxes(img_path)) <= 3:
                chosen.append(img_path)
                break
    return chosen


def build_mosaic(image_paths):
    """Place 4 images in a 2x2 grid and draw their (scaled) boxes."""
    mosaic = Image.new("RGB", (SIZE, SIZE), (114, 114, 114))   # grey, like YOLOv5's padding
    draw = ImageDraw.Draw(mosaic)
    offsets = [(0, 0), (TILE, 0), (0, TILE), (TILE, TILE)]

    for img_path, (ox, oy) in zip(image_paths, offsets):
        img = Image.open(img_path).convert("RGB").resize((TILE, TILE))
        mosaic.paste(img, (ox, oy))
        # boxes are fractions of the original image -> scale to the tile, then shift by the offset
        for c, x, y, w, h in read_boxes(img_path):
            x1, y1 = ox + (x - w / 2) * TILE, oy + (y - h / 2) * TILE
            x2, y2 = ox + (x + w / 2) * TILE, oy + (y + h / 2) * TILE
            draw.rectangle([x1, y1, x2, y2], outline=COLORS[c], width=3)
            draw.rectangle([x1, y1 - 16, x1 + 8 * len(NAMES[c]) + 8, y1], fill=COLORS[c])
            draw.text((x1 + 4, y1 - 14), NAMES[c], fill="white")

    # thin lines where the 4 images meet
    draw.line([(TILE, 0), (TILE, SIZE)], fill="white", width=2)
    draw.line([(0, TILE), (SIZE, TILE)], fill="white", width=2)
    return mosaic


if __name__ == "__main__":
    chosen = pick_images()
    for p in chosen:
        print("using:", p.name)
    build_mosaic(chosen).save(OUT_FILE)
    print(f"Saved {OUT_FILE}")