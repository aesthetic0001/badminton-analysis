import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

from ultralytics import YOLO

TRAIN_ARGS = {
    "data": "ul://oscar-he/datasets/shuttlecock-video-split",
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,
    "device": 0,
    "amp": True,
    "mosaic": 0.0,
    "scale": 0.2,
    "translate": 0.05,
    "patience": 20,
    "project": "badminton",
    "name": "shuttle_detector",
}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the shuttlecock detector.")
    parser.add_argument("--data", default=TRAIN_ARGS["data"], help="Platform dataset URI or local data.yaml")
    args = parser.parse_args()
    model = YOLO("yolo26m.pt")
    model.train(**{**TRAIN_ARGS, "data": args.data})


if __name__ == "__main__":
    main()