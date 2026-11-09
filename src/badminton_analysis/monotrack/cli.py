"""Portable commands for the MonoTrack research baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download and verify public TrackNetV3 and YOLO26n-pose weights")
    fetch.add_argument("--assets", type=Path, default=Path(".cache/monotrack-assets"))
    build = sub.add_parser("build-court", help="Build MonoTrack court detector using OpenCV 4")
    build.add_argument("--build-dir", type=Path, default=Path(".cache/monotrack-build"))
    build.add_argument("--jobs", type=int, default=4)
    run = sub.add_parser("run", help="Analyze consecutive frames or an original video")
    run.add_argument("source", type=Path)
    run.add_argument("--output", type=Path, default=Path("reports/monotrack-smoke"))
    run.add_argument("--assets", type=Path, default=Path(".cache/monotrack-assets"))
    run.add_argument("--court-binary", type=Path, default=Path(".cache/monotrack-build/bin/monotrack-court"))
    run.add_argument("--calibration", type=Path, help="JSON: corners [near-left,near-right,far-left,far-right], pole_tips [left,right]")
    run.add_argument("--fps", type=float, help="Verified original capture FPS, required for 3D")
    run.add_argument("--hits", type=Path, help='Optional verified [{"frame": 120, "player": 1}, ...]')
    run.add_argument("--device", default="auto", help="auto, cpu, mps or cuda:0")
    run.add_argument("--batch-size", type=int, default=1, help="1 keeps memory modest on an 8GB M2")
    run.add_argument("--pose-imgsz", type=int, default=1280, help="1280 helps detect the small far-court player")
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--max-frames", type=int, default=220, help="Maximum frames loaded into memory (default: 220)")
    synthetic = sub.add_parser("synthetic", help="Test calibrated 3D fitting against known synthetic truth")
    synthetic.add_argument("--output", type=Path, default=Path("reports/monotrack-synthetic.json"))
    args = parser.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    if args.command == "fetch":
        from .assets import fetch_pose, fetch_tracknet
        print(fetch_tracknet(args.assets))
        print(fetch_pose(args.assets))
    elif args.command == "build-court":
        source = Path(__file__).parent / "court_detector"
        subprocess.run(["cmake", "-S", str(source), "-B", str(args.build_dir),
                        "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", str(args.build_dir), "--target", "monotrack-court",
                        "--parallel", str(args.jobs)], check=True)
    elif args.command == "synthetic":
        from .synthetic import validate
        result = validate()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    else:
        from .pipeline import run as execute
        execute(args)


if __name__ == "__main__":
    main()
