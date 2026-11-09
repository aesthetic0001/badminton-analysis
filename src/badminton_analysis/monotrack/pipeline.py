"""Headless native MonoTrack baseline with explicit inputs and portable backends."""
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .assets import POSE_SHA256, sha256
from .court import Court
from .hit_detector import AdhocHitDetector
from .pose import Pose
from .reconstruction import Court3D, fit_shot
from .tracking import device_for, track


def load_frames(source, max_frames):
    source = Path(source)
    if source.is_dir():
        paths = [p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        def frame_id(path):
            if not re.fullmatch(r"\d+", path.stem):
                raise ValueError("Frame filenames must be integer IDs, e.g. 000100.jpg")
            return int(path.stem)
        paths.sort(key=frame_id)
        paths = paths[:max_frames]
        ids = [frame_id(p) for p in paths]
        if np.any(np.diff(ids) != 1):
            raise ValueError("Frames must be consecutive; split at missing frames before running TrackNet")
        frames = [cv2.imread(str(p)) for p in paths]
    else:
        if not source.is_file():
            raise FileNotFoundError(source)
        frames = []
        cap = cv2.VideoCapture(str(source))
        try:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        ids = list(range(len(frames)))
    if len(frames) < 8 or any(f is None for f in frames):
        raise ValueError("Need at least eight readable, consecutive frames")
    if any(f.shape != frames[0].shape for f in frames):
        raise ValueError("All frames must have the same dimensions")
    return ids, frames


def calibration_for(frames, output, binary, calibration_file=None):
    if calibration_file:
        calibration = json.loads(Path(calibration_file).read_text())
    else:
        sample = output / "court-input.png"
        cv2.imwrite(str(sample), frames[len(frames) // 2])
        binary = Path(binary).resolve()
        if not binary.is_file():
            raise FileNotFoundError("Build the court detector with `monotrack build-court` or pass --calibration")
        with (output / "court-detector.log").open("w") as log:
            subprocess.run([str(binary), str(sample), str(output / "court.txt"),
                            str(output / "court-automatic.jpg")], check=True, stdout=log, stderr=log)
        points = np.loadtxt(output / "court.txt", delimiter=";")
        # Upstream writes far-left, near-left, near-right, far-right, then net tips.
        calibration = {"corners": points[[1, 2, 0, 3]].tolist(), "pole_tips": points[4:].tolist(),
                       "source": "MonoTrack automatic court detector; inspect court.jpg before using 3D"}
    corners = np.asarray(calibration["corners"], float)
    poles = np.asarray(calibration["pole_tips"], float)
    if corners.shape != (4, 2) or poles.shape != (2, 2) or not np.isfinite(np.r_[corners.ravel(), poles.ravel()]).all():
        raise ValueError("Calibration requires four finite corners and two net-pole tips")
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    return calibration


def poses_for(frames, checkpoint, court, device, batch_size, image_size):
    from ultralytics import YOLO
    if sha256(checkpoint) != POSE_SHA256:
        raise ValueError("Unsupported pose checkpoint; use `monotrack fetch`")
    model = YOLO(str(checkpoint))
    players = [np.full((len(frames), 17, 2), np.nan) for _ in range(2)]
    all_candidates = 0
    pose_device = str(device)
    for start in tqdm(range(0, len(frames), batch_size), desc="Player poses"):
        results = model.predict(frames[start:start + batch_size], device=pose_device,
                                imgsz=image_size, conf=.25, verbose=False)
        for offset, result in enumerate(results):
            if result.keypoints is None:
                continue
            all_candidates += len(result.boxes)
            best_scores = [-1., -1.]
            for xy, confidence, box_conf in zip(result.keypoints.xy.cpu().numpy(),
                                                result.keypoints.conf.cpu().numpy(),
                                                result.boxes.conf.cpu().numpy()):
                xy = xy.copy()
                xy[(confidence < .25) | (xy == 0).all(axis=1)] = np.nan
                # Both feet must be visible to assign a court half reliably.
                feet = xy[15:17]
                if not np.isfinite(feet).all():
                    continue
                base = feet.mean(axis=0)
                half = int(court.in_court(base, slack=(.08, .12)))
                if half:
                    player = half - 1
                    score = float(box_conf * confidence.mean())
                    if score > best_scores[player]:
                        best_scores[player] = score
                        players[player][start + offset] = xy
    return players, all_candidates


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.max_frames < 8 or args.batch_size < 1:
        raise ValueError("--max-frames must be >=8 and --batch-size must be positive")
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0):
        raise ValueError("--fps must be the positive original capture frame rate")
    torch.set_num_threads(args.threads)
    device = device_for(args.device)
    frame_ids, frames = load_frames(args.source, args.max_frames)
    calibration = calibration_for(frames, output, args.court_binary, args.calibration)
    court = Court(calibration["corners"])
    trajectory = track(frames, args.assets / "TrackNet_best.pt", device, args.batch_size)
    shuttle = pd.DataFrame(trajectory, columns=["x", "y", "visible"])
    shuttle.insert(0, "frame", frame_ids)
    shuttle.to_csv(output / "shuttle.csv", index=False)
    players, pose_candidates = poses_for(frames, args.assets / "yolo26n-pose.pt", court,
                                         device, args.batch_size, args.pose_imgsz)
    pose_tables = [pd.DataFrame(p.reshape(len(frames), 34)) for p in players]
    for name, table in zip(("near", "far"), pose_tables):
        table.rename(columns={i: f"{Pose.joint_names[i // 2]}_{'xy'[i % 2]}" for i in range(34)}).assign(
            frame=frame_ids).to_csv(output / f"poses-{name}.csv", index=False)
    # Missing shuttles remain NaN: no unbounded interpolation across flight gaps.
    traj = SimpleNamespace(X=shuttle.x.tolist(), Y=shuttle.y.tolist())
    candidate_hits, hitter = AdhocHitDetector(pose_tables, traj).detect_hits()
    hits = [{"frame": frame_ids[i], "player": int(p), "method": "MonoTrack AdhocHitDetector"}
            for i, p in zip(candidate_hits, hitter) if p]
    if args.hits:
        hits = json.loads(args.hits.read_text())
        if any(h.get("player") not in (1, 2) or h.get("frame") not in frame_ids for h in hits):
            raise ValueError("Each supplied hit needs an included frame ID and player 1 (near) or 2 (far)")
        if any(a["frame"] >= b["frame"] for a, b in zip(hits, hits[1:])):
            raise ValueError("Supplied hits must be strictly ordered")
    (output / "hits.json").write_text(json.dumps(hits, indent=2) + "\n")
    flights = []
    xyz_rows = []
    reason = "Original capture FPS is unknown; pass --fps only when verified from the original footage"
    if args.fps is not None:
        reason = "Need at least two hits with visible player feet at both endpoints"
        camera = Court3D(calibration["corners"], calibration["pole_tips"])
        for first, last in zip(hits, hits[1:]):
            start, end = frame_ids.index(first["frame"]), frame_ids.index(last["frame"])
            feet_start = players[first["player"] - 1][start, 15:17]
            feet_end = players[last["player"] - 1][end, 15:17]
            if end - start < 7 or not np.isfinite(np.r_[feet_start.ravel(), feet_end.ravel()]).all():
                continue
            def world_xy(feet):
                return court.pixel_to_court(feet.mean(axis=0)) * [6.1, 13.4]
            try:
                points, diagnostics = fit_shot(camera, shuttle.loc[start:end, ["x", "y"]].to_numpy(),
                                              frame_ids[start:end + 1], args.fps,
                                              world_xy(feet_start), world_xy(feet_end))
            except ValueError as error:
                flights.append({"start": first["frame"], "end": last["frame"], "skipped": str(error)})
                continue
            flights.append({"start": first["frame"], "end": last["frame"], **diagnostics})
            if diagnostics["optimizer_success"]:
                xyz_rows.extend((fid, *point) for fid, point in zip(frame_ids[start:end + 1], points))
        if xyz_rows:
            reason = "Model estimates only; verify calibration, timing and hits before interpreting 3D"
            pd.DataFrame(xyz_rows, columns=["frame", "x_m", "y_m", "z_m"]).drop_duplicates("frame").to_csv(
                output / "trajectory-3d.csv", index=False)
    if not xyz_rows:
        (output / "trajectory-3d.csv").unlink(missing_ok=True)
    annotated = []
    for i, frame in enumerate(frames):
        overlay = court.draw_lines(frame)
        for p in players:
            pose = Pose()
            pose.init_from_kparray(p[i])
            overlay = pose.draw_skeleton(overlay, thickness=2)
        x, y, visible = trajectory[i]
        if visible:
            cv2.circle(overlay, (int(x), int(y)), 6, (0, 255, 255), 2)
        cv2.putText(overlay, f"frame {frame_ids[i]} | 3D: {'estimate' if xyz_rows else 'skipped'}",
                    (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        annotated.append(_image_for_gif(overlay))
    # GIF has portable playback and never supplies a made-up capture FPS to physics.
    annotated[0].save(output / "preview.gif", save_all=True, append_images=annotated[1:],
                      duration=100, loop=0)
    cv2.imwrite(str(output / "court.jpg"), court.draw_lines(frames[len(frames) // 2]))
    cv2.imwrite(str(output / "sample.jpg"), cv2.cvtColor(np.asarray(annotated[len(frames) // 2]), cv2.COLOR_RGB2BGR))
    summary = {"frames": len(frames), "frame_range": [frame_ids[0], frame_ids[-1]],
               "source": str(args.source), "device": str(device), "fps": args.fps,
               "shuttle_visible": int(shuttle.visible.sum()), "pose_candidates": pose_candidates,
               "player_pose_frames": [int(np.isfinite(p).any(axis=(1, 2)).sum()) for p in players],
               "hit_method": "manual" if args.hits else "MonoTrack heuristic (not trained HitNet)",
               "hits": hits, "reconstruction_status": reason, "flights": flights,
               "three_d_frames": len(set(row[0] for row in xyz_rows)),
               "calibration_source": calibration.get("source", "user-provided"),
               "versions": {"torch": torch.__version__, "opencv": cv2.__version__}}
    summary["pose_imgsz"] = args.pose_imgsz
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def _image_for_gif(bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
