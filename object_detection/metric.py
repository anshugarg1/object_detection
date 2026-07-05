import json
import logging
from pathlib import Path

from object_detection.config import REPORTS_DIR, RUN_NAME

def load_metrics() -> dict:
    path = REPORTS_DIR / f"metrics_{RUN_NAME}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run evaluate_yolo.py first.")
    return json.loads(path.read_text())


def build_markdown(metrics: dict) -> str:
    val = metrics.get("val", {})
    test = metrics.get("test", {})
    classes = list(val.get("per_class_mAP50-95", {}).keys())
    
    lines = [
        f"## YOLO11 results",
        "",
        "| Class | val mAP@0.5:0.95 | test mAP@0.5:0.95 |",
        "|---|---:|---:|",
    ]
    for c in classes:
        v = val["per_class_mAP50-95"].get(c, float("nan"))
        t = test.get("per_class_mAP50-95", {}).get(c, float("nan"))
        lines.append(f"| {c} | {v:.3f} | {t:.3f} |")

    lines += [
        f"| Overall mAP@0.5 | {val.get('mAP50', 0):.3f} | {test.get('mAP50', 0):.3f} |",
        f"| Overall mAP@0.5:0.95 | {val.get('mAP50-95', 0):.3f} | {test.get('mAP50-95', 0):.3f} |",
    ]

    # auto reading: two lowest-AP classes on val
    if classes:
        ranked = sorted(classes, key=lambda c: val["per_class_mAP50-95"].get(c, 0))
        low = ranked[:2]
        lines += [
            "\n",
            f"Lowest per-class AP: {', '.join(low)} consistent with their smaller data points.",
        ]
    return "\n".join(lines) + "\n"


def print_console(metrics: dict) -> None:
    for split, m in metrics.items():
        print(f"\n=== {split.upper()} ===")
        print(f"{'Class':<18}{'mAP@0.5:0.95':>14}")
        print("-" * 32)
        for cls, ap in m["per_class_mAP50-95"].items():
            print(f"{cls:<18}{ap:>14.3f}")
        print("-" * 32)
        print(f"{'Overall mAP50':<18}{m['mAP50']:>14.3f}")
        print(f"{'Overall mAP50-95':<18}{m['mAP50-95']:>14.3f}")


def main() -> None:
    metrics = load_metrics()
    print_console(metrics)
    md = build_markdown(metrics)
    out = REPORTS_DIR / "metrics_table.md"
    out.write_text(md)
    


if __name__ == "__main__":
    main()