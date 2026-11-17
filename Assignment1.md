# Assignment 1 – Garbage Detection with YOLOv5s

Computer Vision Portfolio 1 · Task 1: dataset analysis & pipeline · Task 2: training & evaluation

Click a section to open it.

---

# Task 1 – Dataset Analysis & Training Pipeline

## Our pipeline

```
 Data  →  Analysis  →  Split  →  Augment  →  Model  →  Train  →  best.pt
10,464   imbalanced   stratified  640·flip   YOLOv5s   70 epochs  best val
images   & crowded    70/15/15    ·mosaic    pretrained patience 15 mAP
```

<details>
<summary><b>1. The data</b></summary>

- **10,464 images**, Roboflow export in YOLO format (`images/` + `labels/`)
- **6 classes:** 0 biodegradable · 1 cardboard · 2 glass · 3 metal · 4 paper · 5 plastic
- One label file per image; **one line = one object** (`class x_center y_center width height`, values 0–1)
- **74,090 labelled objects** in total
- Class names identified from the file-name prefixes (e.g. `paper1543_...txt` → lines start with `4`)
- Quality check: every image has a label file, 0 empty labels, 0 corrupt labels

</details>

<details>
<summary><b>2. What we found</b></summary>

| Class | Boxes | Share |
|---|---|---|
| biodegradable | 45,407 | 61.3 % |
| glass | 7,809 | 10.5 % |
| plastic | 5,945 | 8.0 % |
| metal | 5,841 | 7.9 % |
| cardboard | 4,698 | 6.3 % |
| paper | 4,390 | 5.9 % |

- **Strong class imbalance:** biodegradable has about **10×** more objects than paper
- **Busy scenes:** about **7 objects per image** (74,090 ÷ 10,464), many small or overlapping
- Some images are single close-ups (one object filling most of the frame)

</details>

<details>
<summary><b>3. How the findings shaped the pipeline</b></summary>

| Finding | Decision |
|---|---|
| Class imbalance | Stratified split + **per-class** evaluation |
| ~7 objects per image, many small | Image size **640** + **mosaic** augmentation |
| Colour separates materials (cardboard / paper / glass) | Only light colour augmentation, no greyscale |

</details>

<details>
<summary><b>4. Stratified split (70 / 15 / 15)</b></summary>

- **Train** 7,322 images · **Validation** 1,568 · **Test** 1,574 (sum 10,464 ✓)
- **Rarest-class rule:** each image is assigned to the pile of the rarest class it contains (rarity measured over the whole dataset: paper < cardboard < metal < plastic < glass < biodegradable)
- Each pile is shuffled (**seed 42**) and split 70/15/15; leftover images from rounding go to test
- Result: every class appears in all three splits with similar shares (biodegradable 60–64 %, others 5–11 %); paper still has 653 boxes in test
- Limitation: at box level the split is ~66/17/17, because val/test received slightly more crowded images
- Code: `split.py` (split) · `check_split.py` (per-class check)

</details>

<details>
<summary><b>5. Preprocessing & augmentation</b></summary>

| Step | Setting | Why |
|---|---|---|
| Resize | 640 × 640 (letterbox) | keeps small objects visible; divisible by 32 |
| Horizontal flip | 50 % | a mirrored object still looks realistic |
| Vertical flip | off | upside-down scenes are unrealistic |
| Colour | light brightness/saturation only | simulate lighting; keep colour as a material cue |
| Mosaic | on | 4 images → 1, teaches small and crowded objects |

Augmentation is applied to training images only.

**Mosaic augmentation:** 4 training images are shrunk and joined into one image; each box moves with its object.

```
 ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐            ┌───────┬───────┐
 │ glass │ │ metal │ │ paper │ │ bio   │    ──►     │ glass │ metal │
 └───────┘ └───────┘ └───────┘ └───────┘            ├───────┼───────┤
   4 separate training images                       │ paper │ bio   │
                                                    └───────┴───────┘
                                          1 mosaic: smaller objects, busier scene
```

Real example from our training data (made with `make_mosaic.py`):

<img src="mosaic_example.png" width="420" alt="Real mosaic of 4 training images">



</details>

<details>
<summary><b>6. Model choice</b></summary>

- **YOLOv5s** (Ultralytics), a one-stage detector covered in detail in the lecture
- Architecture: CSP backbone → FPN/PAN neck → detection head at 3 scales (80×80, 40×40, 20×20) → NMS
- Size **s**: good balance of accuracy and training time (one night on a Colab T4 GPU)
- **COCO-pretrained (transfer learning):** 343 of 349 weight tensors transferred; only the 6 detection-head tensors were re-initialised (80 COCO classes → 6 classes)
- **AutoAnchor:** 0.999 best possible recall, so the default anchors fit our boxes

</details>

<details>
<summary><b>7. Training settings</b></summary>

| Setting | Value | Why |
|---|---|---|
| Epochs | 70 | enough for a pretrained model to converge |
| Batch size | 16 | fits safely in GPU memory |
| Image size | 640 | see above |
| Early stopping | patience 15 | stop when validation mAP stops improving; avoid overfitting |
| Checkpoint | `best.pt` | epoch with the highest validation mAP |
| Seed | 42 | reproducible results |

```bash
python train.py --img 640 --batch 16 --epochs 70 --patience 15 \
  --data GarbageSplit/data.yaml --weights yolov5s.pt --seed 42
```

</details>

---

# Task 2 – Training & Evaluation

## From training to results

```
 Train  →  best.pt  →  Test  →  Metrics  →  Per class  →  mAP50 0.636
 guess ·    best      1,574     P · R ·     glass best ·   strengths &
 check ·    epoch     unseen    mAP         paper worst    limitations
 adjust
```

<details>
<summary><b>8. How training works</b></summary>

- **Guess:** the model predicts boxes and classes for a batch of photos
- **Check:** compare with the label files (the correct answers)
- **Measure the mistake:** `box_loss` (box position), `obj_loss` (object or not), `cls_loss` (class)
- **Adjust:** change the filter values a tiny bit, then repeat
- **1 epoch** = all 7,322 training photos seen once
- Learns edges and colours first, then shapes and textures
- **`best.pt`** = the learned weights (~7 M numbers) of the epoch with the best validation mAP
- Scores of every epoch are logged in `results.csv`

</details>

<details>
<summary><b>9. Testing on unseen data</b></summary>

| Split | mAP50 |
|---|---|
| Validation | 0.649 |
| **Test (1,574 images)** | **0.636** |

- Test set = the real exam: new photos, used only once
- Validation was used to pick `best.pt`, so it is slightly optimistic
- Test ≈ validation → the model generalises, **no overfitting**

```bash
python val.py --weights best.pt --data GarbageSplit/data.yaml --task test --img 640 --verbose
```

</details>

<details>
<summary><b>10. Metrics explained</b></summary>

| Metric | Test | Meaning |
|---|---|---|
| Precision | **0.735** | of the boxes drawn, how many are right |
| Recall | **0.565** | of the real objects, how many were found |
| mAP50 | **0.636** | average AP over the 6 classes, overlap ≥ 50 % |
| mAP50-95 | **0.411** | stricter: averaged over overlaps 50–95 %, boxes must fit tightly |

- **IoU** = overlap ÷ total area of the predicted and the true box
- A detection counts as correct if IoU ≥ 0.5 and the class matches
- **AP** = precision and recall combined, per class; **mAP** = mean over the classes
- Bigger problem: **missed objects** (recall 0.565), not wrong boxes

</details>

<details>
<summary><b>11. Results per class</b></summary>

| Class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| glass | 0.826 | 0.669 | **0.737** | 0.520 |
| biodegradable | 0.746 | 0.579 | 0.644 | 0.330 |
| metal | 0.774 | 0.571 | 0.627 | 0.417 |
| plastic | 0.627 | 0.587 | 0.619 | 0.373 |
| cardboard | 0.699 | 0.490 | 0.594 | 0.413 |
| paper | 0.736 | 0.492 | **0.592** | 0.413 |

- **Glass best:** consistent shapes, clear edges, transparent and reflective
- **Paper and cardboard worst:** fewest training examples, and they look alike
- **Biodegradable:** lowest mAP50-95 (0.330), irregular shapes give loose boxes
- **Plastic:** lowest precision (0.627), confused with glass and metal
- More data isn't everything: a consistent appearance matters too

</details>

<details>
<summary><b>12. Strengths & limitations</b></summary>

**Strengths**
- High precision (0.735): detections are usually correct
- Generalises well: test (0.636) ≈ validation (0.649)
- Reliable on clear, separate objects (glass 0.737)

**Limitations**
- Low recall (0.565): misses small, overlapping objects in crowded scenes
- Confuses look-alike classes: paper ↔ cardboard, plastic ↔ glass/metal
- Loose boxes on irregular shapes (biodegradable mAP50-95 = 0.330)

</details>

---

<details>
<summary><b>Files</b></summary>

- `label.py` – class counts (data analysis)
- `split.py` – stratified 70/15/15 split
- `check_split.py` – per-class distribution of each split
- `GarbageSplit/data.yaml` – dataset config for YOLOv5
- `make_mosaic.py` – builds `mosaic_example.png` from 4 real training images
- `best.pt` – trained model (best epoch)
- `task1_overview.html` – visual overview of this documentation

</details>