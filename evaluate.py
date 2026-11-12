"""Evaluate the scratch-trained detector on the held-out test split."""

from pathlib import Path
from ultralytics import YOLO

project = Path(__file__).resolve().parent
data = project.parent / "GarbageDataset" / "splits" / "data.yaml"
run = project / "runs" / "detect" / "garbage-yolov8n-scratch"
checkpoint = run / "weights" / "best.pt"

if not checkpoint.is_file():
    raise FileNotFoundError(f"Train the model first; checkpoint not found: {checkpoint}")

model = YOLO(str(checkpoint))
metrics = model.val(
    data=str(data),
    split="test",
    imgsz=416,
    max_det=358,
    project=str(project / "runs" / "detect"),
    name="garbage-yolov8n-scratch-test",
    exist_ok=True,
)

print("\nOverall test metrics")
print(f"Precision:       {metrics.box.mp:.4f}")
print(f"Recall:          {metrics.box.mr:.4f}")
print(f"mAP@0.5:         {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:    {metrics.box.map:.4f}")

print("\nPer-class test metrics")
for index, class_id in enumerate(metrics.box.ap_class_index):
    precision, recall, map50, map5095 = metrics.box.class_result(index)
    print(
        f"{model.names[class_id]}: "
        f"precision={precision:.4f}, recall={recall:.4f}, "
        f"mAP@0.5={map50:.4f}, mAP@0.5:0.95={map5095:.4f}"
    )
