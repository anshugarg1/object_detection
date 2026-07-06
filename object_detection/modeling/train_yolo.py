import yaml
import json
import torch
import shutil
import logging
from pathlib import Path
from ultralytics import YOLO
from object_detection.config import PROJ_ROOT, DATA_YAML, MODELS_DIR, SEED, REPORTS_DIR, RUN_NAME, RUN_DIR

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_yolo")

cfg = yaml.safe_load(open(PROJ_ROOT/ "object_detection/configs/train_yolo.yaml"))

def resolve_device(requested_device):
    if str(requested_device) == "cpu":
        return "cpu"
    
    if not torch.cuda.is_available():
            print("CUDA is not available")
    try: 
        t = torch.randn(64, 64, device="cuda")
        _ = t@t
    except:
        print("CUDA is available but Pytorch is not building.")
        
    print(torch.cuda.get_device_capability(0))
    print(torch.cuda.get_device_properties(0))

    return requested_device


def train() -> Path:
    device = resolve_device(cfg.get("device", 0))
    
    model = YOLO(cfg["arch"])

    results = model.train(
        data = str(DATA_YAML),
        epochs = cfg["epochs"],
        batch = cfg["batch"],
        imgsz = cfg["imgsz"],
        device = device,
        patience = cfg["patience"],
        deterministic = True,
        amp = True,
        project = str(RUN_DIR),
        seed = SEED,
        name = RUN_NAME,
        exist_ok = True,
        plots = True 
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"{RUN_NAME}_best.pt"
    shutil.copy2(best, out)
    log.info("Best weights -> %s", out)
    return out


def evaluate(weights: Path, split: str, cfg: dict) -> dict:
    model = YOLO(str(weights))
    metrics = model.val(
        data = str(DATA_YAML), split=split, device= resolve_device(cfg.get("device", 0)),
        project = str(RUN_DIR), name=f"{RUN_NAME}_{split}", exist_ok=True, plots=True
    )
    names = metrics.names
    per_class = {names[i]:round(float(metrics.box.maps[i]), 4) for i in names}
    summary = {
        "split": split,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "per_class_mAP50-95": per_class,
    }
    return summary

if __name__ == "__main__":
    weights = train()
    # weights = MODELS_DIR / "yolo11m_indoor_best.pt"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {s: evaluate(weights, s, cfg) for s in ("val", "test")}
    out = REPORTS_DIR / f"metrics_{RUN_NAME}.json"
    out.write_text(json.dumps(all_metrics, indent=2))
    log.info("Saved metrics -> %s", out)
