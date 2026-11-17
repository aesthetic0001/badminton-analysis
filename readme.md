# Badminton Analysis

## Motivation

In games like chess, tools such as stockfish exist to provide insight into the effectiveness of decisions made by the
player. This allows for the player to more easily understand, from a quantifiable lens, the effectiveness (or lack
thereof) of their decisions and tie those insights to the overall outcome of the game. Having this insight is extremely
useful for one to improve their gameplay. This project aims to bring those quantative insights into badminton to learn optimal play and bring it to improve your own game.

## Setup

Use Python 3.12 and the locked environment:

```bash
uv sync --locked --extra monotrack
```

PyTorch uses official CUDA wheels on Linux x86-64 and native PyPI wheels on macOS. The original training entry point remains in `main.py`; it uses the corrected Platform dataset and reads `ULTRALYTICS_API_KEY` from the environment.

## MonoTrack native baseline

This adapts MonoTrack's court/physics code with TrackNetV3, YOLO pose and heuristic hits; it does not reproduce the unavailable trained HitNet pipeline.

```bash
.venv/bin/monotrack fetch
```

Model downloads and native build output live under `.cache/` and are excluded from Git. Earlier dataset exports, training runs and comparison experiments have been removed locally.

## Player positions and replay videos

Run `experiments/player_position_heatmap.ipynb` for player-position heatmaps. Its setup cell includes the CUDA pose dependencies. Raw poses, confidence scores, shuttle detections and calibration persist in compressed binary files under `data/player-position-heatmap/`, excluded from Git. Set `resume_run_id` to reuse a saved run; `chunk_frames` controls only the size of new chunks.

Motion filtering uses offline windows, retaining earlier valid observations once movement is confirmed. Derived positions, heatmaps and videos are stored separately for each analysis configuration. The notebook can export the full annotated replay, accepted frames, rejected frames, and rejected frames that passed the court check.

V3 analysis prunes unreliable foot landmarks, computes each available foot centre,
and uses the midpoint of available feet (including a single-foot fallback).
Each court half is selected independently; missing or ambiguous players stay
missing. Stance radius requires both feet. Offline motion windows accept moving
shuttle evidence or movement from every observed player, without measuring player
movement across visibility gaps. Heatmap coverage and CSV dwell time are counted
separately for each player. New position/motion policy IDs keep these results
separate from previous analyses; raw cached detections are reused.

To export the last view from an existing full replay:

```bash
.venv/bin/python -m badminton_analysis.replay data/player-position-heatmap/<run-id> \
  --rejected-court-passed-only --encoder h264_nvenc
```

Replay exports retain source frame IDs and matching audio; they use cached observations without model inference. FFmpeg is required for video export.
