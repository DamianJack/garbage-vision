# visualize.py - Visualisation and error analysis (Assignment 1, Task 3)

# ---------- imports ----------
import random
import sys
import pathlib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image


# compatibility fix: best.pt was saved in Colab with Python 3.13,
# which stores a reference to 'pathlib._local' (not present in Python 3.11)
sys.modules['pathlib._local'] = pathlib

# ---------- settings ----------
ROOT = Path(__file__).resolve().parent                 # the src/ folder
MODEL_PATH = ROOT.parent / "best_model" / "best.pt"
YOLOV5 = ROOT / "yolov5"
TEST_IMAGES = ROOT / "GarbageSplit" / "images" / "test"
NAMES = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
IOU_THR = 0.5
CONF_THR = 0.25

OUT_DIR = ROOT / "examples"


# ---------- functions ----------
def load_model():
    """Load our trained YOLOv5 model (Mac GPU, CUDA GPU or CPU)."""
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = torch.hub.load(str(YOLOV5), 'custom', path=str(MODEL_PATH),
                           source='local', device=device)
    model.conf = CONF_THR
    return model


def read_true_boxes(img_path):
    """Read the label file for an image and return boxes as
    (class, xmin, ymin, xmax, ymax) in pixels."""
    W, H = Image.open(img_path).size                       # image size in pixels
    label_path = Path(str(img_path).replace('/images/', '/labels/')).with_suffix('.txt')

    boxes = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls, x, y, w, h = line.split()
        cls = int(cls)
        x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H   # fractions -> pixels
        boxes.append((cls, x - w / 2, y - h / 2, x + w / 2, y + h / 2))     # centre -> corners
    return boxes


def iou(a, b):
    """a and b are boxes: (xmin, ymin, xmax, ymax)"""
    # 1. corners of the overlap rectangle
    left   = max(a[0], b[0])
    top    = max(a[1], b[1])
    right  = min(a[2], b[2])
    bottom = min(a[3], b[3])

    # 2. overlap area (0 if they don't touch)
    overlap = max(0, right - left) * max(0, bottom - top)

    # 3. area of each box
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    # 4. union = both areas minus the overlap (so it's not counted twice)
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def match(pred_boxes, true_boxes, iou_thr=IOU_THR):
    """pred_boxes: list of (cls, conf, xmin, ymin, xmax, ymax)
       true_boxes: list of (cls, xmin, ymin, xmax, ymax)"""
    correct, wrong_class, false_alarm = [], [], []
    taken = set()                                        # true boxes already matched

    for p in sorted(pred_boxes, key=lambda p: -p[1]):    # most confident first
        best_iou, best_j = 0, None
        for j, t in enumerate(true_boxes):
            if j in taken:
                continue
            score = iou(p[2:], t[1:])
            if score > best_iou:
                best_iou, best_j = score, j

        if best_iou >= iou_thr:
            taken.add(best_j)
            if p[0] == true_boxes[best_j][0]:
                correct.append(p)                              # right place, right class
            else:
                wrong_class.append((p, true_boxes[best_j]))    # right place, wrong class
        else:
            false_alarm.append(p)                              # box on nothing

    missed = [t for j, t in enumerate(true_boxes) if j not in taken]   # never found
    return correct, wrong_class, false_alarm, missed


def predict(model, img_path):
    """Run the model on one image.
    Returns a list of (cls, conf, xmin, ymin, xmax, ymax)."""
    res = model(str(img_path))
    df = res.pandas().xyxy[0]
    preds = [(int(r['class']), r.confidence, r.xmin, r.ymin, r.xmax, r.ymax)
             for _, r in df.iterrows()]
    return preds

def draw(img_path, correct, wrong_class, false_alarm, missed, title, save_name):
    """Draw true and predicted boxes on an image and save it as a PNG."""
    img = Image.open(img_path)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img)

    def box(xmin, ymin, xmax, ymax, color, text, dashed=False):
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                                       edgecolor=color, linewidth=2,
                                       linestyle='--' if dashed else '-'))
        ax.text(xmin, ymin - 2, text, color='white', fontsize=8,
                bbox=dict(facecolor=color, alpha=0.8, pad=1))

    for p in correct:
        box(*p[2:], 'green', f"{NAMES[p[0]]} {p[1]:.2f}")
    for p, t in wrong_class:
        box(*p[2:], 'orange', f"pred: {NAMES[p[0]]} / true: {NAMES[t[0]]}")
    for p in false_alarm:
        box(*p[2:], 'red', f"FA {NAMES[p[0]]} {p[1]:.2f}")
    for t in missed:
        box(*t[1:], 'blue', f"MISSED {NAMES[t[0]]}", dashed=True)

    ax.set_title(title)
    ax.axis('off')
    OUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUT_DIR / save_name, bbox_inches='tight', dpi=150)
    plt.close(fig)

def find_examples(model, test_images, per_category=3, max_objects=5):
    """Go through test images and collect a few examples of each error type."""
    found = {'correct': [], 'missed': [], 'wrong_class': [], 'false_alarm': []}

    for img_path in test_images:
        trues = read_true_boxes(img_path)
        if len(trues) > max_objects:            # cheap check first: skip crowded images
            continue

        preds = predict(model, img_path)        # expensive step only when needed
        c, w, f, m = match(preds, trues)
        n = len(trues)

        if len(c) == n and not w and not f:
            category = 'correct'
        elif w:
            category = 'wrong_class'
        elif m and not f:
            category = 'missed'
        elif f and not m:
            category = 'false_alarm'
        else:
            continue

        if len(found[category]) < per_category:
            found[category].append((img_path, c, w, f, m))

        if all(len(v) >= per_category for v in found.values()):
            break                              
    return found

if __name__ == "__main__":
    model = load_model()
    test_images = sorted(TEST_IMAGES.glob("*"))
    random.seed(42)
    random.shuffle(test_images)
    found = find_examples(model, test_images)

    for category, examples in found.items():
        print(category, len(examples))
        for i, (img_path, c, w, f, m) in enumerate(examples, start=1):
            draw(img_path, c, w, f, m,
                 title=f"{category.replace('_', ' ')} - {img_path.name[:30]}",
                 save_name=f"{category}_{i}.png")

    print(f"Saved pictures to {OUT_DIR}/")