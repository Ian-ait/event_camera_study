from pathlib import Path
from PIL import Image
import shutil

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "visdrone_dataset" / "VisDrone2019-DET-val"
SOURCE_IMAGES = SOURCE / "images"
SOURCE_LABELS = SOURCE / "annotations"

OUTPUT = ROOT / "yolo_dataset"
OUTPUT_IMAGES = OUTPUT / "images" / "val"
OUTPUT_LABELS = OUTPUT / "labels" / "val"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

image_files = list(SOURCE_IMAGES.glob("*.jpg"))

for index, image_path in enumerate(image_files, start=1):
    label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    yolo_lines = []

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as file:
            for line in file:
                values = line.strip().split(",")

                if len(values) < 6:
                    continue

                x, y, box_width, box_height = map(float, values[:4])
                score = int(values[4])
                visdrone_class = int(values[5])

                if score == 0 or not 1 <= visdrone_class <= 10:
                    continue

                yolo_class = visdrone_class - 1

                center_x = (x + box_width / 2) / image_width
                center_y = (y + box_height / 2) / image_height
                norm_width = box_width / image_width
                norm_height = box_height / image_height

                yolo_lines.append(
                    f"{yolo_class} "
                    f"{center_x:.6f} {center_y:.6f} "
                    f"{norm_width:.6f} {norm_height:.6f}"
                )

    output_label = OUTPUT_LABELS / f"{image_path.stem}.txt"
    output_label.write_text("\n".join(yolo_lines), encoding="utf-8")

    shutil.copy2(image_path, OUTPUT_IMAGES / image_path.name)

                    