"""Apply training augmentation to the dataset."""

from pathlib import Path
from ultralytics import YOLO

DATASET = Path(__file__).resolve().parent.parent / "GarbageDataset"
DATA_YAML = DATASET / "splits" / "data.yaml"

# YOLO resizes/normalizes images on load and applies these box-aware transforms to training only.
AUGMENTATION = {
    "hsv_h": 0.015,
    "hsv_v": 0.2,
    "degrees": 5.0,
    "scale": 0.3,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 0.5,
    "close_mosaic": 10,
}


def main():

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        imgsz=416,
        augment=True,
        epochs=100,
        batch=16,
        patience=20,
        seed=42,
        **AUGMENTATION,
        project=str(Path(__file__).resolve().parent / "runs" / "detect"),
        name="garbage-yolov8n",
    )


if __name__ == "__main__":
    main()