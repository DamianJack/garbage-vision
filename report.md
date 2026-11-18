# CV Portfolio 1 – Garbage Detection & Deep Metric Learning
Prof. Dr. Dominik Seuß · WS 2026 · **Team:** Damian, Krishna, Shravani

## Contributions
| Part | Responsible |
|---|---|
| A1 Task 1 – Data analysis & pipeline | Shravani |
| A1 Task 2 – Training & evaluation | Krishna |
| A1 Task 3 – Visualisation & error analysis | Damian |
| A2 Task 1 – Loss function | Damian |
| A2 Task 1 – Sampling | Shravani |
| A2 Task 2 – Embedding space | Shravani, Krishna |

---

# Assignment 1 – Object Detection (YOLOv5s)

## Task 1 – Data Analysis and Pipeline
- 10,464 images, 74,090 boxes, 6 classes, YOLO format
- Strong imbalance: biodegradable 61 %, paper 6 %
- Crowded scenes: ≈ 7 objects per image, many small and overlapping
- Quality: no empty or corrupt labels; 17 zero-size boxes (0.02 %), reported as label noise
- Stratified 70/15/15 split by each image's rarest class (seed 42): 7,322 / 1,568 / 1,574 images
- Training: 640 px, mosaic, horizontal flip, YOLOv5 default colour augmentation; COCO-pretrained; 70 epochs max, batch 16, early stopping (patience 15), `best.pt`

## Task 2 – Training and Evaluation
| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Validation | 0.716 | 0.583 | 0.649 | 0.417 |
| **Test** | **0.735** | **0.565** | **0.636** | **0.411** |

- Test ≈ validation → generalises, no overfitting
- Precision > recall → boxes are usually right, but many objects are missed
- Best class glass (0.737); worst paper (0.592) and cardboard (0.594): few examples, look alike
- Biodegradable has the lowest mAP50-95 (0.330): loose boxes on irregular shapes

## Task 3 – Visualisation and Error Analysis
- Matching: IoU ≥ 0.5, confidence ≥ 0.25, most confident first
- Green = correct · orange = wrong class · red = false alarm · blue dashed = missed

**Correct**

![](img/c1.jpg) ![](img/c2.jpg) ![](img/c3.jpg)

**Missed**

![](img/m1.jpg) ![](img/m2.jpg) ![](img/m3.jpg)

**Wrong class**

![](img/w1.jpg) ![](img/w2.jpg) ![](img/w3.jpg)

**False alarm**

![](img/f1.jpg) ![](img/f2.jpg)

**Counts over the test images (≤ 5 objects)**

![](img/numbers.png)

**Causes**
- Look-alike materials: paper ↔ cardboard, glass ↔ metal (reflections)
- Parts of objects boxed separately (seed, handle), mostly low confidence
- Occlusion in crowded scenes → misses
- Inconsistent labels (stems, leaves) and stock-photo domain

**Improvements (not tested)**
- Confidence threshold ≈ 0.4 · more look-alike and rare-class examples · clean labels · conveyor-belt images · larger model

---

# Assignment 2 – Deep Metric Learning (theory)
- A network maps each object crop to a 128-d embedding: same class close, different class far
- Input: crops from YOLO boxes; network: ImageNet-pretrained ResNet

## Task 1 – Loss and Sampling
**Triplet loss:** L = max(0, d(a, p) − d(a, n) + m)
- a = anchor, p = same class, n = other class, d = Euclidean distance, m = margin
- Goal: the negative is at least m farther than the positive; otherwise loss 0
- Chosen because the goal is relative: varied classes may stay spread, look-alikes are pushed apart directly

**Sampling: P × K batches + semi-hard negatives**
- P = 6 classes × K = 4 crops = 24 per batch → every class equally represented (counters imbalance)
- 24 × 24 distance matrix → pick for each anchor a negative farther than the positive but inside the margin
- Easy negatives teach nothing; hardest ones are unstable and may be mislabelled
- Result: stable training, compact and well-separated class groups

## Task 2 – Embedding Space
- **Q1 Retrieval:** embed all training crops (gallery), embed the query, return the K smallest distances
- **Q2 Unknown sample:** k-NN majority vote; if the nearest item is beyond a distance threshold → "unknown"; new classes just need gallery examples
- **Q3 Visualisation:** UMAP to 2D, coloured by class; separate groups = good, overlap = confusion, stray dots = possible label errors
- **Q4 Evaluation:** Recall@1 (is the nearest item the same class?) and k-NN accuracy (is the class vote right?), per class; test crops = queries, training crops = gallery

---

## How to run

**Not included (size):** the dataset and the YOLOv5 repository.

**Setup**
- Python 3.11 with PyTorch; install the YOLOv5 requirements: `pip install -r yolov5/requirements.txt`
- Place inside `src/`:
  - `yolov5/`: `git clone https://github.com/ultralytics/yolov5`
  - `GarbageDataset/` (`images/`, `labels/`): the original Garbage Detection Dataset
- `best.pt` is in `best_model/`

```
best_model/best.pt
src/
├── split.py  train.py  evaluate.py  visualize.py  data.yaml
├── yolov5/
├── GarbageDataset/      (input)
└── GarbageSplit/        (created by split.py)
```

**Run (from `src/`)**
| Step | Command | Output |
|---|---|---|
| 1. Split | `python split.py` | `GarbageSplit/` (7,322 / 1,568 / 1,574 images) |
| 2. Train (optional, GPU) | `python train.py` | `runs/train/yolov5s_garbage/weights/best.pt` |
| 3. Evaluate | `python evaluate.py` | test P 0.735 · R 0.565 · mAP50 0.636 · mAP50-95 0.411 |
| 4. Visualise | `python visualize.py` | example pictures in `examples/` |

Steps 3 and 4 use the submitted `best_model/best.pt`, so training is not needed to reproduce the results.