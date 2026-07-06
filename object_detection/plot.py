import shutil
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from object_detection.config import PROCESSED_DATA_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR, RUN_DIR, RUN_NAME


def _iou_pair(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _load_gt(label_path: Path, w: int, h: int):
    """Read YOLO txt -> (xyxy array [N,4], class_id array [N]) in pixels."""
    boxes, cls = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, bw, bh = map(float, line.split())
            x1, y1 = (cx - bw/2) * w, (cy - bh/2) * h
            x2, y2 = (cx + bw/2) * w, (cy + bh/2) * h
            boxes.append([x1, y1, x2, y2])
            cls.append(int(c))
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(cls, dtype=int)


def _image_score(pred_xyxy, pred_cls, gt_xyxy, gt_cls, iou_thr=0.5) -> float:
    """Per-image detection quality = TP / (TP + FP + FN) via greedy class-aware IoU match."""
    tp, matched, matched_pred  = 0, set(), set()
    for i in range(len(pred_xyxy)):
        best_iou, best_j = 0.0, -1
        for j in range(len(gt_xyxy)):
            if j in matched or pred_cls[i] != gt_cls[j]:
                continue
            iou = _iou_pair(pred_xyxy[i], gt_xyxy[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            tp += 1
            matched.add(best_j)
            matched_pred.add(i)
    fp = len(pred_xyxy) - tp
    fn = len(gt_xyxy) - len(matched)
    denom = tp + fp + fn
    score = 1.0 if denom == 0 else tp / denom
    err_cls = ([int(pred_cls[i]) for i in range(len(pred_cls)) if i not in matched_pred]  
               + [int(gt_cls[j]) for j in range(len(gt_cls)) if j not in matched])         
    return score, {"tp": tp, "fp": fp, "fn": fn, "err_cls": err_cls}                       


def _cls_tag(stats, gt_cls, p_cls, names) -> str:
    """Classes the model got wrong on this image (FP + FN); for a perfect image,
    the classes actually present. Used to label the output filename."""
    ids = stats["err_cls"] or list(gt_cls) or list(p_cls)
    uniq = sorted({names[int(c)] for c in ids})
    return "-".join(uniq) if uniq else "none"


def _render(img_path, pred, gt_xyxy, gt_cls, names) -> np.ndarray:
    """Green = predictions (with labels), red = ground truth."""
    image = cv2.imread(str(img_path))
    gt_det = sv.Detections(xyxy=gt_xyxy, class_id=gt_cls) if len(gt_xyxy) else sv.Detections.empty()
    image = sv.BoxAnnotator(color=sv.Color.RED, thickness=2).annotate(image, gt_det)
    if len(pred.boxes):
        pred_det = sv.Detections(
            xyxy=pred.boxes.xyxy.cpu().numpy(),
            class_id=pred.boxes.cls.cpu().numpy().astype(int),
            confidence=pred.boxes.conf.cpu().numpy(),
        )
        labels = [f"{names[c]} {cf:.2f}" for c, cf in
                  zip(pred_det.class_id, pred_det.confidence)]
        image = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2).annotate(image, pred_det)
        image = sv.LabelAnnotator(color=sv.Color.GREEN).annotate(image, pred_det, labels=labels)
    return image


def show_best_worst(split: str, k: int = 4, weights: Path = None, conf: float = 0.25) -> Path:
    """Render the k best and k worst detection examples on a split into reports/figures/.

    Score each image by TP/(TP+FP+FN), rank, and save annotated best/worst images
    (green predictions vs red ground truth). Returns the output directory.
    """
    weights = weights or MODELS_DIR / f"{RUN_NAME}_best.pt"
    model = YOLO(str(weights))
    names = model.names

    images_dir = PROCESSED_DATA_DIR / "yolo" / "images" / split
    labels_dir = PROCESSED_DATA_DIR / "yolo" / "labels" / split

    scored = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        pred = model.predict(str(img_path), conf=conf, verbose=False)[0]
        h, w = pred.orig_shape
        gt_xyxy, gt_cls = _load_gt(labels_dir / f"{img_path.stem}.txt", w, h)
        p_xyxy = pred.boxes.xyxy.cpu().numpy() if len(pred.boxes) else np.zeros((0, 4))
        p_cls = pred.boxes.cls.cpu().numpy().astype(int) if len(pred.boxes) else np.zeros((0,), int)
        score, stats = _image_score(p_xyxy, p_cls, gt_xyxy, gt_cls)
        cls_tag = _cls_tag(stats, gt_cls, p_cls, names)
        scored.append((score, cls_tag, stats, img_path, pred, gt_xyxy, gt_cls))

    scored.sort(key=lambda x: x[0])
    worst = scored[:k]
    best = scored[-k:][::-1]

    out_dir = FIGURES_DIR / f"{split}_examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, group in (("good", best), ("bad", worst)):
        for rank, (score, cls_tag, stats, img_path, pred, gt_xyxy, gt_cls) in enumerate(group, 1):
            img = _render(img_path, pred, gt_xyxy, gt_cls, names)
            dst = out_dir / (
                f"{tag} {rank} {cls_tag} score {score:.2f}"
                f" tp {stats['tp']} fp {stats['fp']} fn {stats['fn']} {img_path.stem}.jpg"
            )
            cv2.imwrite(str(dst), img)
    return out_dir


if __name__=="__main__":
    show_best_worst("val")
    show_best_worst("test")