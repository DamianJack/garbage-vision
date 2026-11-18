"""evaluate.py - Evaluate the trained YOLOv5s model on the test set.

Runs YOLOv5's val.py with --task test (held-out test images, used once),
with the same settings as the Colab run behind the reported results
(test mAP50 0.636). Prints precision, recall, mAP50 and mAP50-95,
overall and per class (--verbose); YOLOv5 saves plots to runs/val/test_results/.
"""
import pathlib
import runpy
import sys
from pathlib import Path

# compatibility fix: best.pt was saved in Colab with Python 3.13,
# which stores a reference to 'pathlib._local' (not present in Python 3.11)
sys.modules['pathlib._local'] = pathlib

ROOT = Path(__file__).resolve().parent
YOLOV5 = ROOT / "yolov5"
DATA_YAML = ROOT / "data.yaml"
WEIGHTS = ROOT.parent / "best_model" / "best.pt"

SETTINGS = {
    "weights": str(WEIGHTS),
    "data": str(DATA_YAML),
    "task": "test",       # evaluate on the held-out test split
    "img": 640,           # same size as training
    "device": "cpu",      # same device as the reported run
    "project": str(ROOT / "runs" / "val"),
    "name": "test_results",
}


def main():
    sys.argv = ["val.py", "--verbose"]
    for key, value in SETTINGS.items():
        sys.argv += [f"--{key}", str(value)]
    sys.path.insert(0, str(YOLOV5))          # so val.py finds its own modules
    runpy.run_path(str(YOLOV5 / "val.py"), run_name="__main__")


if __name__ == "__main__":
    main()