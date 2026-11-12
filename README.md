# Garbage Detection — Task 1

## Dataset analysis

The supplied dataset contains **10,464 JPEG images** and **10,464 matching YOLO text labels**. All images are 416 × 416 pixels. Labels use the normalized detection format `class_id x_center y_center width height`. There are **74,090 annotated boxes**, an average of 7.08 boxes per image (median 2; 95th percentile 32), so images can contain multiple objects and some are crowded.

The class-name mapping below is inferred from image filename categories and the dominant class IDs in their labels.

| Class ID | Class | Boxes | Share of boxes |
|---:|---|---:|---:|
| 0 | biodegradable | 45,407 | 61.3% |
| 1 | cardboard | 4,698 | 6.3% |
| 2 | glass | 7,809 | 10.5% |
| 3 | metal | 5,841 | 7.9% |
| 4 | paper | 4,390 | 5.9% |
| 5 | plastic | 5,945 | 8.0% |

Biodegradable is the dominant class, while the other classes have fewer boxes. Therefore, a single overall score could hide poor results for less common classes.  A label audit found **17 boxes with zero width or height**. 

## Data split and preprocessing

The saved split manifests contain **7,325 training**, **1,570 validation**, and **1,569 test** images. The YOLO data configuration is `GarbageDataset/splits/data.yaml`; its manifests point to images in `GarbageDataset/images`, with corresponding labels in `GarbageDataset/labels`. The subsets are kept separate so validation can guide model selection and the test set can provide a final evaluation that was not used for tuning. The 70/15/15 ratio is a practical compromise: most examples remain available for learning, while validation and test each retain about 1,570 images. 

All source images already have the same 416 × 416 resolution, so no resizing or re-encoding is needed. YOLO performs input resizing and normalization when it loads images. 

## Detector and training pipeline

The chosen detector is **Ultralytics YOLOv8n**, initialized from the pretrained `yolov8n.pt` weights. YOLOv8n is a compact single-stage detector suitable as a practical baseline for this dataset. Fine-tuning pretrained weights provides a starting point with learned visual features instead of training a detector from scratch.

The implemented training entry point is `train.py`. It sets:

| Parameter | Value | Rationale |
|---|---:|---|
| Input size | 416 | Matches the source resolution and avoids unnecessary upscaling. |
| Maximum epochs | 100 | Allows training time to converge; it may stop earlier. |
| Batch size | 16 | Initial practical setting; lower it if device memory is insufficient. |
| Early-stopping patience | 20 | Stops if validation fitness does not improve for 20 epochs. |
| Seed | 42 | Makes the run more repeatable. |
| Pretrained weights | `yolov8n.pt` | Uses transfer learning for the six waste categories. |

### Augmentation choices

Augmentation is applied online by YOLO during training. YOLO transforms detection boxes along with the image, so labels stay aligned. Validation and test evaluation do not use training augmentation; original image and label files are not changed.

| Setting in `train.py` | Value | Reason |
|---|---:|---|
| `hsv_h` | 0.015 | Small hue variation adds color diversity without large color shifts. |
| `hsv_v` | 0.2 | Brightness variation helps with illumination differences. |
| `degrees` | 5 | Small rotations add pose variation while limiting unrealistic orientations. |
| `scale` | 0.3 | Varies apparent object size and framing. |
| `fliplr` | 0.5 | Horizontal flips add left-right variation where orientation is not class-defining. |
| `flipud` | 0.0 | Avoids upside-down scenes that may be unrealistic for these photos. |
| `mosaic` | 0.5 | Sometimes combines training images, helping the detector see varied object counts and layouts. |
| `close_mosaic` | 10 | Disables mosaic for the last 10 epochs so final learning uses more ordinary image layouts. |

These are starting choices.

## Running and evaluating

From the workspace root, install Ultralytics if needed and run:

```powershell
python -m pip install ultralytics
python .\garbage-vision\train.py
```