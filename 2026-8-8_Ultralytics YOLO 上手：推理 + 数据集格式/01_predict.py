from ultralytics import YOLO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT/"img"
OUTPUT_DIR = ROOT/"runs"
model = YOLO("yolo11n.pt")

results = model.predict(source = IMAGE_DIR, 
                        device = 0, imgsz = 640, 
                        conf = 0.25, save = True, project = OUTPUT_DIR, 
                        name = "bird_drone_test", exist_ok = True)

for result in results:
    print(
        f"{Path(result.path).name}:"
        f"检测到{len(result.boxes)}个目标"
    )