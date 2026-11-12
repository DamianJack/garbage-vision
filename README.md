# Task 1 — Dataset Analysis and Training Pipeline

## Dataset analysis

The supplied `GarbageDataset` was inspected before proposing the training setup. It contains **10,464 JPEG images and 10,464 matching YOLO text-label files**; every image has a label file and vice versa. All images decode successfully and have the same **416 × 416** resolution. The images occupy about **200 MB** in total. The six filename prefixes are `biodegradable`, `cardboard`, `glass`, `metal`, `paper`, and `plastic`.

The labels are YOLO detection rows (`class_id x_center y_center width height`), with normalized coordinates. There are **74,090 annotated boxes**, or **7.08 boxes per image on average** (median 2; 95th percentile 32). This is a multi-object detection task, and the long tail in object count means training and visualization should include crowded examples rather than only typical images.

| Class ID | Inferred class | Boxes | Share of boxes |
|---:|---|---:|---:|
| 0 | biodegradable | 45,407 | 61.3% |
| 1 | cardboard | 4,698 | 6.3% |
| 2 | glass | 7,809 | 10.5% |
| 3 | metal | 5,841 | 7.9% |
| 4 | paper | 4,390 | 5.9% |
| 5 | plastic | 5,945 | 8.0% |

Class names were inferred from the filename categories and the dominant class IDs in their annotations; the dataset folder does not include an authoritative class-name configuration. The mapping is strongly supported by the labels, but should be confirmed against the original dataset/source before final submission. Class 0 is notably dominant, so overall accuracy alone would hide weak performance on minority classes.

