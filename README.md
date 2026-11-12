# Garbage Detection Project

## Dataset analysis

The supplied `GarbageDataset` contains **10,464 JPEG images** and **10,464 matching YOLO text-label files**. All images are 416 × 416 pixels, decode successfully, and occupy about 200 MB. Each label row uses the YOLO format `class_id x_center y_center width height`, with normalized coordinates. The dataset has 74,090 annotated boxes, averaging 7.08 boxes per image (median 2; 95th percentile 32), so it includes multi-object and crowded images.

The six class names below are inferred from the image filename categories and the dominant class IDs in their labels. Confirm this mapping with the original dataset source if possible.

| Class ID | Class | Boxes | Share of boxes |
|---:|---|---:|---:|
| 0 | biodegradable | 45,407 | 61.3% |
| 1 | cardboard | 4,698 | 6.3% |
| 2 | glass | 7,809 | 10.5% |
| 3 | metal | 5,841 | 7.9% |
| 4 | paper | 4,390 | 5.9% |
| 5 | plastic | 5,945 | 8.0% |

Class 0 is substantially more common than the other classes. Therefore, evaluate per-class precision, recall, and average precision as well as overall metrics. A small annotation audit found **17 boxes with zero width or height**. The training script does not check or correct annotations, so review those rows in the source labels before training.

## Split and input preprocessing

The dataset is split into **7,325 training**, **1,570 validation**, and **1,569 test** images. The YOLO configuration at `GarbageDataset/splits/data.yaml` points to the `train.txt`, `val.txt`, and `test.txt` image manifests. The image paths in these manifests use the original `GarbageDataset/images` files, whose matching labels are in `GarbageDataset/labels`. Physical copies in the split subfolders are not needed by this configuration.

No separate image preprocessing or augmented copies are made. In `train.py`, `imgsz=416` tells YOLO to resize and prepare images as they are loaded. Its detection pipeline applies image transformations together with corresponding bounding-box transformations during training, keeping annotations aligned.

## Training and augmentation

The training entry point is `train.py`. It fine-tunes the pretrained **YOLOv8n** detector with:

| Setting | Value |
|---|---:|
| Input image size | 416 |
| Maximum epochs | 100 |
| Batch size | 16 |
| Early-stopping patience | 20 epochs |
| Random seed | 42 |
| HSV hue adjustment | 0.015 |
| HSV value adjustment | 0.2 |
| Rotation | up to 5 degrees |
| Scale | 0.3 |
| Horizontal flip probability | 0.5 |
| Vertical flip probability | 0.0 |
| Mosaic probability | 0.5 |
| Final epochs without mosaic | 10 |

These are online training augmentations; they do not modify the source images or labels. YOLO applies augmentation during training, not validation or test evaluation. The chosen mild geometric and color changes add variety, while avoiding vertical flips and reducing mosaic near the end of training.

Install the training package and start training from the workspace root:

```powershell
python -m pip install ultralytics
python .\garbage-vision\train.py
```

Training results are saved under `garbage-vision/runs/detect/garbage-yolov8n/`. Use validation results to select the best checkpoint. After model selection, evaluate the test set once, without using it to tune the model:

```powershell
yolo detect val model=.\garbage-vision\runs\detect\garbage-yolov8n\weights\best.pt data=.\GarbageDataset\splits\data.yaml split=test imgsz=416
```

Reduce the batch size if the available device memory is insufficient. Report per-class results because the dataset is imbalanced.
