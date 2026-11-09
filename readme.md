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

See [the native setup guide](docs/monotrack.md) for Apple Silicon and Linux commands, current OpenCV support, public model downloads, and a bundled 64-frame test. [The smoke-test report](reports/monotrack-smoke/report.md) records results and limitations. This adapts MonoTrack's court/physics code with TrackNetV3, YOLO pose and heuristic hits; it does not reproduce the unavailable trained HitNet pipeline.

```bash
.venv/bin/monotrack fetch
.venv/bin/monotrack run examples/monotrack/frames \
  --calibration examples/monotrack/calibration.json
```

Model downloads and native build output live under `.cache/` and are excluded from Git. Earlier dataset exports, training runs and comparison experiments have been removed locally.
