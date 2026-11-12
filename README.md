# Garbage Detection — Task 1

## Dataset analysis

The supplied dataset contains **10,464 JPEG images** and **10,464 matching YOLO text labels**. All images are 416 × 416 pixels. Labels use the normalized detection format `class_id x_center y_center width height`. There are **74,090 annotated boxes**, an average of 7.08 boxes per image (median 2; 95th percentile 32), so images can contain multiple objects and some are crowded.

The class-name mapping below is inferred from filename categories and the dominant class IDs in the annotations. Confirm it against the original dataset source if possible.

| Class ID | Class | Boxes | Share of boxes |
|---:|---|---:|---:|
| 0 | biodegradable | 45,407 | 61.3% |
| 1 | cardboard | 4,698 | 6.3% |
| 2 | glass | 7,809 | 10.5% |
| 3 | metal | 5,841 | 7.9% |
| 4 | paper | 4,390 | 5.9% |
| 5 | plastic | 5,945 | 8.0% |

Biodegradable is much more frequent than the other classes. An overall score could hide weak results for less common classes, so report precision, recall, and average precision per class as well as overall. A label audit found **17 boxes with zero width or height**. The training script does not check or correct labels; review and correct these annotations before training.

## Split and preprocessing

The saved split manifests contain **7,325 training**, **1,570 validation**, and **1,569 test** images. The YOLO configuration is `GarbageDataset/splits/data.yaml`; its manifests reference images under `GarbageDataset/images`, with corresponding labels under `GarbageDataset/labels`. The split uses about 70% of the examples for learning, while retaining validation and test subsets of about 1,570 images each. This is a practical balance, not a claim that 70/15/15 is uniquely optimal.

The images are already 416 × 416, so separate resizing or re-encoding is unnecessary. YOLO prepares images at the configured input size as it loads them. Geometric augmentations are applied together with corresponding bounding-box transformations; source files are not changed.

## Detector and training strategy

The selected approach is the **Ultralytics YOLOv8n single-stage detector**, initialized from pretrained `yolov8n.pt` weights. The nano model is a compact baseline that should be practical to fine-tune on this dataset. Pretraining starts from learned visual features instead of random initialization.

The implemented training entry point is `train.py`:

| Parameter | Value | Justification |
|---|---:|---|
| Input size | 416 | Matches the source resolution; compare a larger size only if validation shows small-object misses. |
| Maximum epochs | 100 | Allows time to learn; early stopping may end training sooner. |
| Batch size | 16 | Initial practical setting; reduce it if device memory is insufficient. |
| Early-stopping patience | 20 | Stops if validation fitness does not improve for 20 epochs. |
| Seed | 42 | Makes the training run more repeatable. |
| Optimizer and learning rate | Ultralytics defaults | Keeps the initial experiment simple; record the library version because defaults can change. |

### Augmentation

YOLO applies these augmentations online to training examples, updating their bounding boxes along with the image. Validation and test evaluation do not use training augmentation.

| Setting | Value in script | Purpose |
|---|---:|---|
| `hsv_h` | 0.015 | Small hue variation. |
| `hsv_v` | 0.2 | Brightness variation for lighting changes. |
| `degrees` | 5 | Mild rotation to vary object orientation. |
| `scale` | 0.3 | Varies apparent object size and framing. |
| `fliplr` | 0.5 | Adds left-right variation where orientation is not class-defining. |
| `flipud` | 0.0 | Avoids upside-down scenes that may be unrealistic for these photos. |
| `mosaic` | 0.5 | Sometimes combines training images to vary object counts and layouts. |
| `close_mosaic` | 10 | Turns mosaic off for the last 10 epochs so final learning uses ordinary layouts. |

Other augmentation values not specified in `train.py` use Ultralytics defaults (for example, `hsv_s=0.7` and `translate=0.1` in version 8.4.211). The `augment=True` prediction option is intentionally not set; it enables test-time augmentation, not the online training transformations listed above. Inspect augmented examples and reduce strengths if objects or material cues become unrealistic.

## Running and evaluation

From the workspace root, install Ultralytics if needed and run:

```powershell
python -m pip install ultralytics
python .\garbage-vision\train.py
```

The script saves results under `garbage-vision/runs/detect/garbage-yolov8n/`. Use validation results to select the best checkpoint. Do not tune the model using test results; after decisions are fixed, evaluate the test subset once:

```powershell
yolo detect val model=.\garbage-vision\runs\detect\garbage-yolov8n\weights\best.pt data=.\GarbageDataset\splits\data.yaml split=test imgsz=416
```

Report precision, recall, mAP@0.5, and mAP@0.5:0.95 overall and per class. Record the split sizes, image size, seed, Ultralytics version, and selected checkpoint. These are pipeline settings and justifications, not measured model results; report performance only after training and test evaluation have actually completed.
