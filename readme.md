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

### Jump segmentation and correction

Train from a pose cache and a CSV of visually reviewed, half-open source-frame intervals. The columns are `player,start_frame,stop_frame,state,split,kind`. Use `near` or `far` for player and `train` or `validation` for split. Use `airborne` for jumps that warrant position correction, `grounded` when a foot is in contact, and `no_correction` for reviewed movements whose impact on the projected position is negligible. The latter two are negative classifier targets. Use `uncertain` when a frame cannot be judged reliably. Mark planted lunges and shuffles as `lunge` and `shuffle` to sample them more often.

```bash
.venv/bin/python -m badminton_analysis.train_jumps RUN_DIRECTORY LABELS.csv MODEL.pt
```

Training writes the model and a validation JSON report. Use `--no-augmentation` for the baseline, or `--no-landmark-perturbation` to keep mirroring and frame masking only.

To apply a model, set `jump_checkpoint` and `jump_projection_path` in the player-position notebook. The projection must be a 3×4 NumPy array mapping court metres (including height) into the cached processing pixels for the fixed view. `projection_from_pole_tops` can estimate it from the reference court homography and the annotated net-pole tops in court-X order. The fit uses SciPy from the `monotrack` extra. The notebook saves jump probabilities, trajectories, and event records with the derived analysis. A replay marks fitted positions with a diamond.

To find new training examples, run the review queue on a scored training video. Pass existing label files with `--reviewed` so the queue skips reviewed intervals and protects validation context.

```bash
.venv/bin/python -m badminton_analysis.review_jumps ANALYSIS_DIRECTORY REVIEW.csv \
    --reviewed LABELS.csv --reel REVIEW.mp4
```

The queue includes likely jumps, near-threshold clips, and randomly sampled low-score footage. The reel shows source frame numbers and the player to review. Record verified contact states in the training label format. Confirmed false positives can become hard negatives. Keep validation matches out of this feedback loop so they can measure gains between rounds.

After retraining, pass the pose-cache directory as the source and add `--checkpoint MODEL.pt` to generate a new queue directly from the updated model.

### RGB jump experiment

The RGB experiment takes 13 consecutive broadcast frames and scores the near and far players at the center frame. It preserves the full view at 320×192. During pretraining, optical flow propagates the masks through the clip and the model reconstructs the color in hidden patches. The supervised classifier uses the unmasked RGB frames.

```bash
.venv/bin/python -m badminton_analysis.train_rgb_mgmae prepare SOURCES.json CACHE_DIRECTORY
.venv/bin/python -m badminton_analysis.train_rgb_mgmae train CACHE_DIRECTORY RGB_MODEL.pt
```

The source configuration lists each video and its label CSV. A training source can also specify rally intervals or cached court scores for sampling unlabeled clips. Keep the validation match out of pretraining and fine-tuning. Pass `--pretrain-epochs 0` for a supervised RGB comparison.

For clips with no contact labels, omit `labels` and specify `pretraining`. Set `pretraining_groups` to change the sample size for a source. The training report gives the number of clips and the duration of unique footage used after deduplicating overlapping windows.

To try the same classifier on a single player, build a crop cache from the RGB cache and train from that directory. Both players use the same classifier. A fixed padded box across each 13-frame window keeps court contact and vertical motion in view.

```bash
.venv/bin/python -m badminton_analysis.prepare_rgb_player_crops CACHE_DIRECTORY CROP_CACHE --weights yolo26m.pt
.venv/bin/python -m badminton_analysis.train_rgb_mgmae train CROP_CACHE PLAYER_MODEL.pt
```

The pose masking experiment accepts `--regions body --bone-weight .5` to reconstruct both masked landmarks and vectors between connected joints.

### Jump benchmark

The benchmark reports strict frame recall and false-positive rate alongside one-to-one event recall. On 30 fps footage, a reference event up to six frames long allows one frame of error at either boundary. Longer reference events allow two frames. Convert these allowances to the source frame rate. Explicitly uncertain boundary frames widen the reference interval and are excluded from the frame metrics. Grounded frames and negligible split steps count as negatives at every scored frame.
