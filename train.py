"""train.py - Fine-tune a COCO-pretrained YOLOv5s on the garbage dataset.

Uses the original YOLOv5 repository (cloned into ./yolov5) with its default
hyperparameters and augmentation (mosaic, horizontal flip, HSV colour jitter).
Training was run on Google Colab (T4 GPU) with exactly these settings.
The best checkpoint (highest validation mAP) is saved as
runs/train/yolov5s_garbage/weights/best.pt.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YOLOV5 = ROOT / "yolov5"
DATA_YAML = ROOT / "data.yaml"

SETTINGS = {
    "img": 640,                # small objects stay visible; divisible by 32
    "batch": 16,
    "epochs": 70,
    "patience": 15,            # early stopping on validation mAP
    "weights": "yolov5s.pt",   # COCO-pretrained (transfer learning)
    "seed": 42,
    "project": str(ROOT / "runs" / "train"),
    "name": "yolov5s_garbage",
}


def main():
    cmd = [sys.executable, str(YOLOV5 / "train.py"), "--data", str(DATA_YAML)]
    for key, value in SETTINGS.items():
        cmd += [f"--{key}", str(value)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()