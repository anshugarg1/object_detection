import json
import shutil
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pathlib import Path
from object_detection.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RATIOS, SEED


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text())

def build_presence_matrix(coco: dict) -> dict:
    presence_matrix = np.zeros((len(coco["images"]), len(coco["categories"])), dtype=int)
    print(f"Presence matrix shape: {presence_matrix.shape}")
    
    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        presence_matrix[image_id-1][category_id-1] = 1

    return presence_matrix


def stratified_split(image_ids, presence_matrix):
    train_ratio, val_ratio, test_ratio = RATIOS
    image_ids = np.array([image["id"] for image in image_ids])
    X = np.zeros((len(image_ids), 1))  # Dummy feature matrix for stratification
    
    s1 = MultilabelStratifiedShuffleSplit(n_splits=1, random_state=SEED, test_size=test_ratio)
    train_val_id, test_id = next(s1.split(X, presence_matrix))

    val_size = val_ratio / (train_ratio + val_ratio)
    s2 = MultilabelStratifiedShuffleSplit(n_splits=1, random_state=SEED, test_size=val_size)
    train_id, val_id = next(s2.split(X[train_val_id], presence_matrix[train_val_id]))

    return {
        "train": set(image_ids[train_val_id][train_id].tolist()),
        "val": set(image_ids[train_val_id][val_id].tolist()),
        "test": set(image_ids[test_id].tolist())
    }


def verify_splits(coco: dict, splits: dict):
    train_categories = {cat["name"]:0 for cat in coco["categories"]}
    val_categories = {cat["name"]:0 for cat in coco["categories"]}
    test_categories = {cat["name"]:0 for cat in coco["categories"]}

    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if image_id in splits["train"]:
            train_categories[coco["categories"][category_id-1]["name"]] += 1
        elif image_id in splits["val"]:
            val_categories[coco["categories"][category_id-1]["name"]] += 1
        elif image_id in splits["test"]:
            test_categories[coco["categories"][category_id-1]["name"]] += 1

    print(f"Train categories: {train_categories}")
    print(f"Validation categories: {val_categories}")
    print(f"Test categories: {test_categories}")
    

def write_coco_splits(coco: dict, splits: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, image_ids in splits.items():
        split_coco = {
            "images": [image for image in coco["images"] if image["id"] in image_ids],
            "annotations": [annotation for annotation in coco["annotations"] if annotation["image_id"] in image_ids],
            "categories": coco["categories"]
        }
        output_path = output_dir / f"anno_coco_{split_name}.json"
        output_path.write_text(json.dumps(split_coco, indent=4))


def write_yolo_splits(coco: dict, splits: dict, out_dir: Path, image_src: Path, label_src: Path):
    id_to_file = {image["id"]: image["file_name"] for image in coco["images"]}
    cls_names = [cls["name"] for cls in coco["categories"]]

    for split_name, image_ids in splits.items():
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in image_ids:
            img_src_pth = image_src / id_to_file[img_id]
            lbl_src_path = label_src / (id_to_file[img_id].split('.jpg')[0]+".txt")
            # print(img_src_pth)
            # print(lbl_src_path)
            shutil.copy2(img_src_pth, img_out)
            shutil.copy2(lbl_src_path, lbl_out)
    
    data_yaml = "\n".join([
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(cls_names)}",
        f"names: {cls_names}",
        "\n"
    ])
    (out_dir/"data.yaml").write_text(data_yaml)



if __name__ == "__main__":
    coco_data = load_coco(INTERIM_DATA_DIR / "coco/anno_coco.json")
    presence_matrix = build_presence_matrix(coco_data)
    splits = stratified_split(coco_data["images"], presence_matrix)
    verify_splits(coco_data, splits)
    print("Final data processing to coco format..")
    write_coco_splits(coco_data, splits, PROCESSED_DATA_DIR / "coco")
    print("Final data processing to yolo format..")
    write_yolo_splits(coco_data, splits, PROCESSED_DATA_DIR/"yolo", INTERIM_DATA_DIR/"images", INTERIM_DATA_DIR/"yolo/labels")

