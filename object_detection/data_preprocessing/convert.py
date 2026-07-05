import os
import json
import shutil
import logging 
import argparse
from PIL import Image
from pathlib import Path
from xml.etree import ElementTree as ET
from dataclasses import dataclass, field
from object_detection.config import INTERIM_DATA_DIR, RAW_DATA_DIR, RAW_DATA_ANNO_DIR, RAW_DATA_IMG_DIR, PROCESSED_DATA_DIR


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("convert")

@dataclass
class Box:
    label: str
    x: float   #top x coordinate
    y: float   #left y coordinate
    w: float   #width of the box
    h: float   #height of the box

@dataclass
class Annotation:
    file: str
    path: Path   
    width: int   #width of the image
    height: int   #height of the image
    boxes: list[Box] = field(default_factory=list)    #list of boxes for this image


def parse_dlib_xml(xml_path: Path, image_dir: Path) -> list[Annotation]:
    root = ET.parse(xml_path).getroot()
    anns: list[Annotation] = []
    n_dropped = 0
    n_dropped_ls = []
    n_missing = 0
    n_missing_ls = []

    for img_el in root.find("images").findall("image"):
        file_name = img_el.attrib["file"]
        image_path = image_dir / file_name

        #list of annotion boxes for this image
        boxes: list[Box] = []

        if not image_path.exists():
            n_missing += 1
            n_missing_ls.append(file_name)
            log.warning("Image not found, skipping: %s", image_path)
            continue
        
        with Image.open(image_path) as img:
            W, H = img.size
            # print(f"Image size: {W}x{H}")
        
        for b in img_el.findall("box"):
            # print(f"box: {b.attrib}")
            left = float(b.attrib["left"])
            top = float(b.attrib["top"])
            bw = float(b.attrib["width"])
            bh = float(b.attrib["height"])
            label_el = b.find("label")
            if label_el is None or not label_el.text:
                log.warning("Missing label for box in image %s, skipping box.", file_name)
                continue
            label = label_el.text 

            #clamp corners from the image annotations if its going out of image.
            #top, left for bounding box
            x1, y1 = max(0.0, left), max(0.0, top)
            #bottom, right for bounding box
            x2, y2 = min(float(W), left + bw), min(float(H), top + bh)

            #drop if 
            if x2-x1 <= 1 or y2-y1 <= 1:
                n_dropped += 1
                n_dropped_ls.append(file_name)
                log.warning("Dropped box for image %s: (%.2f, %.2f, %.2f, %.2f)", file_name, x1, y1, x2, y2)
                continue

            boxes.append(Box(label=label, x=x1, y=y1, w=x2-x1, h=y2-y1))

        # print(f"boxes: {len(boxes)}")
        anns.append(Annotation(file=file_name, path=image_path, width=W, height=H, boxes=boxes))

    log.info(
        "Parsed %d images (%d missing, %d degenerate boxes dropped).",
        len(anns), n_missing, n_dropped,
    )
    return anns


def discover_classes(anns: list[Annotation]) -> list[str]:
    """Discover unique classes from the annotations."""
    names = sorted({box.label for a in anns for box in a.boxes})
    log.info("Discovered %d unique classes: %s", len(names), names)
    return names

def convert_to_coco(anns: list[Annotation], classes: list[str]) -> dict:
    """Convert the annotations to COCO format."""
    cat_id = {name: i + 1 for i, name in enumerate(classes)}
    images, annotations, categories = [], [], []
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(classes)]

    for img_id, a in enumerate(anns, start=1):
        images.append({
            "id": img_id,
            "file_name": a.file,
            "width": a.width,
            "height": a.height,
        })

        for box in a.boxes:
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": img_id,
                "category_id": cat_id[box.label],
                "bbox": [round(box.x, 2), round(box.y, 2), round(box.w, 2), round(box.h, 2)],
                "area": round(box.w * box.h, 2),
                "iscrowd": 0,
            })

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

#rel_path = Path('object_detection\data\processed\yolo')
def convert_to_yolo(anns: list[Annotation], classes: list[str], labels_dir: Path):
    """Convert the annotations to YOLO format."""
    cls_id = {name: i for i, name in enumerate(classes)}
    
    if Path(labels_dir).exists() is False:
        Path(labels_dir).mkdir(parents=True, exist_ok=True)
    
    for a in anns:
        lines = []
        for box in a.boxes:
            cx = (box.x + box.w / 2)/a.width
            cy = (box.y + box.h / 2)/a.height
            nw = box.w/a.width
            nh = box.h/a.height
            lines.append(f"{cls_id[box.label]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        (labels_dir / f"{Path(a.file).stem}.txt").write_text("\n".join(lines)+("\n" if lines else ""))


def collect_images(anns: list[Annotation], images_out: Path) -> None:
    """Flatten images from the 6 sequence folders into one dir."""
    images_out.mkdir(parents=True, exist_ok=True)
    for a in anns:
        dst = images_out / a.file
        if not dst.exists():
            shutil.copy2(a.path, dst)

def main() -> None:
    all_anns: list[Annotation] = []

    for raw_anno_path in os.listdir(RAW_DATA_ANNO_DIR):
        if raw_anno_path.endswith(".xml"):
            seq_number = raw_anno_path.split(".")[0][-1]
            all_anns.extend(parse_dlib_xml(Path(RAW_DATA_ANNO_DIR) / raw_anno_path, Path(RAW_DATA_IMG_DIR)/f'sequence_{seq_number}'))
    
    classes = discover_classes(all_anns)

    #convert to COCO format
    coco_data = convert_to_coco(all_anns, classes)
    coco_dir = INTERIM_DATA_DIR / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_dir/ f"anno_coco.json", "w") as f:
        json.dump(coco_data, f, indent=4)
    log.info(f"Saved COCO -> {coco_dir / f'anno_coco.json'} ,{len(coco_data['images'])} imgs, {len(coco_data['annotations'])} anns")


    # Convert to YOLO format
    yolo_dir = INTERIM_DATA_DIR / "yolo"
    yolo_labels_dir = yolo_dir / "labels"       
    convert_to_yolo(all_anns, classes, yolo_labels_dir)
    log.info(f"Saved YOLO -> {yolo_labels_dir} ({len(all_anns)} imgs, {sum(len(a.boxes) for a in all_anns)} anns)")

    collect_images(all_anns, INTERIM_DATA_DIR / "images")
    log.info("Flattened images -> %d", len(list((INTERIM_DATA_DIR / "images").glob("*"))))


if __name__ == "__main__":
    main()
