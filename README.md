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
