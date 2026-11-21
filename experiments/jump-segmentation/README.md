# Jump segmentation experiments

The target is at least 99% strict airborne-frame recall and below 0.5% false positives per reviewed negative player-frame. Report event recall separately, with a one-frame allowance at each boundary for jumps up to six frames at 30 fps and two frames for longer jumps. A jump is positive when it warrants correcting the court-plane position. Use `no_correction` for negligible split steps and `uncertain` for frames where contact cannot be verified.

## Footage and contact labels

| Pretraining run | Sampled windows | Unique footage |
| --- | ---: | --- |
| First full-frame RGB | 384 | 54.4 s across two matches |
| Expanded full-frame RGB | 2,580 | 366.0 s across six matches and 16 short clips |
| Pose, including body masking and the joint-vector objective | 10,322 | 525.4 s from one match |

The expanded RGB run used one pretraining epoch. Its footage included the BWF pilot and FineBadminton IDs `0001`, `0003`, `0033`, `0057`, and `0070`. The 16 clips from the Mendeley archive contributed 21.5 seconds of unique footage after sampling. They span forehand smash, defence, forehand drop, and backhand drive. The official archive contained 872 videos organized by stroke class. [FineBadminton-20K](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K) · [Mendeley badminton stroke dataset](https://data.mendeley.com/datasets/3sp4xntp34/1)

The contact CSVs use zero-based source frames and exclusive stops. FineBadminton shot intervals guided the review. Each contact label was checked against the corresponding frames. The FineBadminton source revision is `72cae601e08c83ee8ec7e19f8fd7850e9158af7d`.

| Reviewed labels | Partition | Jumps | Airborne frames | Negative frames | Uncertain frames |
| --- | --- | ---: | ---: | ---: | ---: |
| [FineBadminton 0057](labels/finebadminton-0057.csv) | Training | 7 | 59 | 80 | 44 |
| [FineBadminton 0070](labels/finebadminton-0070.csv) | Training | 8 | 74 | 67 | 46 |
| [Mendeley defence clips](labels) | Training | 0 | 0 | 80 | 3 |
| [FineBadminton 0063](labels/finebadminton-0063.csv) | Held out | 5 | 49 | 69 | 16 |

Normalize the Mendeley source clips to 30 fps before applying their frame labels. The foreground player is `near`. The four label files `mendeley-defence-1.csv` through `mendeley-defence-4.csv` map in order to `Defence (5).mp4`, `Defence (3).mp4`, `Defence (2).mp4`, and `Defence (4).mp4` in `Dataset/Defence/`.

## Results on FineBadminton 0063

All checkpoints used a threshold of 0.5. All 118 binary-labeled player-frames received scores. Only the reviewed intervals around five jumps were scored. Evaluating on full matches with a broader sample of grounded play is still needed.

| Experiment | Airborne frames detected (of 49) | False flags (of 69) | Jumps matched (of 5) |
| --- | ---: | ---: | ---: |
| Original full-frame RGB with motion masking | 30 | 42 | 0 |
| Expanded full-frame RGB with motion masking | 0 | 0 | 0 |
| Box-crop RGB with motion masking | 5 | 17 | 0 |
| Box-crop RGB with motion masking and the 15 new training jumps | 10 | 17 | 0 |
| Pose MLP trained from scratch with practice negatives, seeds 7 / 17 / 27 | 42 / 39 / 44 | 10 / 8 / 10 | 4 / 4 / 4 |
| Pose with masked body regions and practice negatives, same seeds | 42 / 40 / 43 | 10 / 10 / 10 | 4 / 4 / 4 |
| Pose with masked body regions, joint vectors and practice negatives, same seeds | 41 / 39 / 41 | 9 / 9 / 8 | 4 / 4 / 4 |

The pose experiments used the BWF training labels for the initial checkpoints and then fit on BWF, 0057, 0070, and the four practice clips. The joint-vector loss reconstructs the vector between connected landmarks when a mask hides either endpoint and both source landmarks are reliable. It reduced false flags from 10 per seed to 8–9 in this comparison and also reduced airborne frames detected. On the three-seed BWF development comparison, both pose masking variants detected 21 of 21 airborne frames per seed and averaged 9.7 false flags on 41 hard-negative frames. The objective has yet to meet both targets. Across the nine pose checkpoints trained with practice negatives, a threshold high enough for zero false flags on 0063 also removed every true positive.

## Reproducing the pose comparison

Prepare one RGB cache with reviewed frames from 0057 and 0070 and a separate cache with reviewed frames from 0063. Use their label CSVs and omit `pretraining` for these caches. WholeBody pose extraction operates on the corresponding original-resolution frames. The practice directory takes the four normalized clips named `defence-1.mp4` through `defence-4.mp4`.

```bash
.venv/bin/python experiments/jump-segmentation/prepare_pose.py \
  POSE_CACHE RGB_TRAIN_CACHE RGB_HOLDOUT_CACHE \
  --pose-model RTMPOSE_WHOLEBODY.onnx --practice-directory PRACTICE_CLIPS

.venv/bin/python experiments/jump-segmentation/pose_transfer.py \
  POSE_CACHE PILOT_DIRECTORY BWF_OBSERVATION_STORE NEW_OUTPUT_DIRECTORY --practice
```

The pilot directory supplies `bwf-correction-target-labels.csv` and the `motion-{7,17,27}.pt` checkpoints in `mgmae-body/` and `mgmae-relationships/`. To reproduce the joint-vector pretraining, pass `--regions body --bone-weight .5` to `badminton_analysis.train_pose_mgmae`. Use `--regions body` alone for the control.
