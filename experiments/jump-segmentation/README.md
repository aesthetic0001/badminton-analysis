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

## Court-aware player tracking

Nine of the ten previous false flags from the scratch seed 27 checkpoint on 0063 occurred while the far-player box was on a line judge. For the calibrated extraction, project detection boxes onto the court, screen for plausible player locations, and favor continuity with the last detection for each side. The calibrated extractor also runs person detection at 960 pixels and samples every other frame. The previous extractor used 640 pixels and sampled every fourth frame. All checkpoint weights and the 0.5 threshold were fixed for this comparison.

| Evaluation set | Earlier extraction | Court-aware extraction |
| --- | --- | --- |
| FineBadminton 0063, 5 reviewed jumps | 44/49 airborne frames, 10/69 false flags, 4/5 matched events | 48/49 airborne frames, 1/69 false flags, 5/5 matched events |
| FineBadminton 0033, 12 newly reviewed jumps | 101/113 airborne frames, 10/178 false flags, 11/12 matched events | 104/113 airborne frames, 4/178 false flags, 12/12 matched events |

Both rows use scratch seed 27 from the practice-augmented training run. On 0033, false flags averaged 8.9 per checkpoint with earlier extraction and 2.6 with court-aware extraction across all nine checkpoints. Two checkpoints recorded zero false flags there with 93/113 strict airborne-frame recall. The top-recall checkpoint’s nine remaining missed frames were within two frames of takeoff or landing. All reviewed binary labels were scored. The separately trained three-seed refit on court-aware inputs peaked at 99/113 strict recall on 0033. The sampled windows still leave the 99% strict recall and 0.5% FPR targets unverified on broader play.

Use [the reviewed intervals for 0033](labels/finebadminton-0033.csv) to prepare a separate RGB cache. Its V1 pose inputs and the court-aware version can be scored against the same checkpoint. The [court reference map](court-reference-frames.json) identifies fixed-camera calibration frames. Calibrations apply to the same camera view within each reviewed segment.

```bash
.venv/bin/python experiments/jump-segmentation/prepare_pose.py \
  COURT_POSE_CACHE RGB_CACHE_0033 --pose-model RTMPOSE_WHOLEBODY.onnx \
  --court-reference-frames experiments/jump-segmentation/court-reference-frames.json
.venv/bin/python experiments/jump-segmentation/score_pose.py \
  COURT_POSE_CACHE finebadminton-0033 PILOT_OUTPUT/scratch-27.pt
```

## Boundary experiments and ordinary rally play

[FineBadminton 0003](labels/finebadminton-0003.csv) adds 12 jumps and eight samples of ordinary rally play selected independently of model scores. It has 130 verified airborne frames and 509 negative player-frames, with 333 of those sampled from ordinary play. An audit at source frame 20700 confirmed that the far player's foot had planted several frames earlier than the first review indicated. The table uses the corrected boundary for every checkpoint. All binary labels received scores.

The boundary decoder keeps a 0.5 segment when it includes a sufficiently confident peak, then extends it by up to two frames at each end at 30 fps when the adjacent score supports the transition. Both settings were chosen on BWF, 0063, and 0033. The pose-score comparison with foot-motion features used one match at a time for evaluation while fitting on the other two. It missed more airborne frames, so the holdout comparison used the simpler decoder.

| Method on 0003 | Airborne frames detected (of 130) | False flags (of 509) | Events matched (of 12) |
| --- | ---: | ---: | ---: |
| Fixed scratch seed 27, direct 0.5 threshold | 123 | 26 | 12 |
| Same checkpoint, decoder with thresholds 0.98 / 0.30 / 0.50 | 119 | 13 | 11 |
| Same checkpoint, decoder with thresholds 0.98 / 0.10 / 0.10 | 121 | 19 | 11 |
| Expanded supervised training, highest-recall run | 129 | 45 | 12 |
| Expanded training with soft boundaries, highest-recall run with fewer false flags | 129 | 31 | 12 |
| Expanded training with soft boundaries, lowest-FPR run | 121 | 6 | 11 |

The expanded training adds [10 jumps and 131 negative frames from FineBadminton 0001](labels/finebadminton-0001.csv). The optional soft targets give partial probability to 143 uncertain training frames between verified grounded and airborne states. These targets model a transition with an equally likely landing or takeoff time inside the reviewed interval. The strict benchmark still uses verified binary labels. Across three seeds and three initializations per configuration, expanded training raised mean recall from 122.4 to 127.3 of 130 frames at 0.5 and increased false flags from 23.0 to 32.7. Adding soft boundaries yielded means of 125.4 detected and 23.7 false flags.

A sweep over the held-out frame scores quantifies the remaining gap. The best of 27 refitted checkpoints detected 119 of 130 airborne frames at the maximum of two false flags allowed by the 0.5% FPR target on this sample. The lowest false-positive count at the 99% recall target was 27 of 509. The sweep diagnoses separability on 0003. Any selected operating point needs verification on unused footage. The frame errors include confidently flagged approach steps and shuffles, so boundary recovery alone leaves a substantial false-positive problem.

To evaluate the decoder, add `--decode .98 .30 .50` to the `score_pose.py` command. To include 0001 when fitting, prepare a separate labels-only pose cache and add `--extra-source POSE_CACHE_0001 finebadminton-0001` to the `pose_transfer.py` command. Add `--soft-boundaries` to reproduce the soft-target comparison.

A follow-up run added [67 hard-negative frames](labels/bwf-mined-negatives.csv) from nine short, high-confidence model predictions in the BWF training match. Review confirmed planted lunges, strides, and negligible shuffles. At 0.5, mean false flags fell from 23.7 to 16.4 across the nine runs with soft boundaries. Mean airborne frames detected fell from 125.4 to 119.3. One run recorded 112/130 airborne frames with zero false flags. The holdout threshold diagnostic improved to 122/130 airborne frames with at most two false flags. At 99% recall, the lowest false-positive count among the new runs was 24/509.

To reproduce this follow-up, append the mined intervals to the existing BWF training label CSV and pass the new file with `--bwf-labels`. Keep the BWF validation rows in the combined CSV.

## Shoe motion and a ranking loss

[FineBadminton 0010](labels/finebadminton-0010.csv) provides seven jump events, 42 airborne frames, and 505 negative player-frames. It includes 365 negative frames from eight randomly sampled sections of ordinary play. The pose classifiers and shoe-motion classifiers in this comparison were fitted using other matches. Development cutoffs used 0003, 0033, and 0063.

The shoe experiment measures motion of textured points in small regions around the toe and heel landmarks. It tracks points in both directions, rejects inconsistent tracks, and expresses speed relative to player height. The comparison fits the same regularized logistic model with pose scores alone or with the extra shoe features. At each model's development cutoff for low false positives, pose alone detected 39/42 frames with 4/505 false flags. Adding shoe motion detected 39/42 with 5/505 false flags. Both matched all seven events. A sweep on 0010 found 24/505 false flags at full recall with shoe motion and 37/505 with pose alone. That sweep measures score ordering on 0010.

The ranking experiment adds a loss after the first 20 training epochs. It compares the four highest negative scores with the four lowest positive scores in each batch and penalizes pairs with less than a one-logit margin. It keeps binary cross-entropy as the main loss. Uncertain soft labels still contribute to binary cross-entropy.

| Ranking weight | Development-selected checkpoint | Development recall with at most 3/756 false flags | Held-out recall at the same cutoff | Held-out false flags | Matched events |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Body masking, seed 17 | 254/292 | 39/42 | 13/505 | 7/7 |
| 0.2 | Body masking, seed 7 | 271/292 | 40/42 | 22/505 | 7/7 |
| 1.0 | Body masking, seed 7 | 278/292 | 42/42 | 15/505 | 7/7 |

On 0010, a diagnostic sweep after training reduced the fewest false flags required for full recall across the nine runs from 21 at ranking weight 0 to 13 at ranking weight 1. The best recall at the 0.5% false-positive limit rose from 36/42 to 39/42. The remaining high scores cluster on lunges, approach steps, and shuffles.

Use `pose_transfer.py --ranking-weight 1` with the earlier hard-negative and soft-boundary options to reproduce the ranking run. Pass each development cache and source name as `--development POSE_CACHE SOURCE` to `score_pose.py` to choose cutoffs before scoring the holdout. The CLI reports both the transferred cutoff and the holdout diagnostic.

Extract shoe features with `shoe_motion.py POSE_NPZ VIDEO FLOW_CACHE/SOURCE.npz`. Pass trained pose checkpoints to `score_shoe_contact.py FLOW_CACHE CHECKPOINT ... --dev POSE_CACHE SOURCE --holdout POSE_CACHE SOURCE`. Repeat `--dev` for each development match. The shoe comparison used the soft-boundary scratch seed 27 and the mined-negative body-masking seed 17 checkpoints.

The next run added 144 no-correction player-frames from 15 reviewed net-play windows in [0001](labels/finebadminton-0001-net.csv), [0057](labels/finebadminton-0057-net.csv), and [0070](labels/finebadminton-0070-net.csv). A frozen ranked checkpoint scored 73 clips around net shots and lifts so review could prioritize high-scoring lunges and shuffles. The added training examples did not improve the selected operating point. Its best development recall was 269/292 at three false flags, and that checkpoint transferred to 40/42 frames and 34/505 false flags. The ranked model trained on the earlier data remains the best of these development-selected runs at full recall on 0010.

## Verification across an entire proposed jump

A second model scores each proposed jump using its duration, peak, mean logit, time above a high score, and relative peak timing. It adjusts the original per-frame logits, which retain the timing information. Development scores came from fitting the verifier on three matches and predicting the fourth in turn. The original ranked pose checkpoint remained frozen. Using a higher verifier weight recovered 320/334 reviewed development positives at the false-positive target, up from 308/334 with the pose checkpoint alone. A lower-weight version recovered 318/334 there and needed fewer false flags to reach the recall target. An exploratory version built from pose-derived foot speed and hip rise did not improve the low-FPR result.

[FineBadminton 0013](labels/finebadminton-0013.csv) was downloaded separately and kept out of both model fits and cutoff selection. It contributes 12 jumps, 106 confirmed airborne player-frames, and 555 confirmed negative frames. The source averages 24.80 fps. The native-rate test kept the original frame indices and used the source rate for event-timing allowances. All reviewed binary labels received scores. [FineBadminton source release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Model at its development cutoff for low false positives | Airborne frames detected (of 106) | False flags (of 555) | Matched events (of 12) |
| --- | ---: | ---: | ---: |
| Ranked pose model | 96 | 3 | 11 |
| Higher-weight verifier | 100 | 0 | 12 |
| Lower-weight verifier | 99 | 2 | 12 |

The six strict misses from the higher-weight verifier were all within two native frames of takeoff or landing. At the development cutoff for recall, the lower-weight verifier reached 104/106 with 14 false flags. A follow-up evaluation on the same match interpolated pose inputs to reproduce the training cadence of 30 fps. The higher-weight verifier then detected 97/106 with zero false flags and 11/12 matched events. The lower-weight version detected 103/106 with seven false flags at its development cutoff for recall. The interpolation run uses the same held-out match after the native-rate result was reviewed.

On the earlier BWF match, the higher-weight verifier retained 20/21 airborne frames and 6/69 false flags, matching two of three jumps at its selected cutoff. It kept the first frame of the two-frame short hop at 46949, missed the second, and rejected the negligible split step at 57279. BWF contributed other clips to pose training.

For pose-only sources, run `prepare_pose.py POSE_CACHE --sources SOURCES.json --pose-model RTMPOSE_WHOLEBODY.onnx --court-reference-frames court-reference-frames.json`. The manifest lists each video and label CSV. This preserves the native timing and bypasses the RGB pilot’s 30 fps preparation restriction. Run `verify_jump_events.py CHECKPOINT --dev CACHE SOURCE --holdout CACHE SOURCE`, repeating `--dev` for 0003, 0033, 0063, and 0010. Pass `--window-fps 30` for the cadence comparison.

A boundary follow-up required the original pose score to stay at or above 0.4 over a contiguous path before transferring verifier confidence by up to two frames before a jump and one frame after it at 30 fps. This setting was selected on the four development matches. Among other settings with 320/334 positives at the development false-positive limit, it needed the fewest false flags to reach the development recall target. Rechecking 0013 yielded 102/106 airborne frames, 1/555 false flags, and 12/12 matched jumps at the development cutoff for low false positives. This is a follow-up on the same match. On BWF it recovered both frames of the short hop, rejected the negligible split step, and scored 21/21 airborne frames with 7/69 false flags.

## Ensemble and boundary-weighted training

Three of the four remaining strict misses on 0013 scored below 0.5 in all nine ranked pose checkpoints. Averaging checkpoint logits therefore had limited scope to recover them. A separate training run sampled each labeled pose window at frame-spacing factors of 0.8, 1.0, and 1.25 with the same center target. It selected one of those views per sampled example and kept the number of updates per epoch fixed. Another run tripled the sampling weight of the first and last two confirmed airborne frames within positive examples. Both kept the ranking loss.

The development set for this experiment includes 0003, 0033, 0063, 0010, and 0013. At the development false-positive limit, the strongest single checkpoint with boundary recovery scored 424/440 airborne frames. The best of six ensemble groupings scored 418/440 and used all nine checkpoints trained with temporal augmentation. The best boundary-weighted checkpoint scored 416/440. Its selection used development results only.

[FineBadminton 0018](labels/finebadminton-0018.csv) supplied an independent match averaging 24.63 fps. The original review had 12 jumps, 116 positive frames, and 573 negative frames. The ensemble and boundary recovery detected 112 of those 116 frames and flagged four negative frames. The four flags were consecutive far-player frames at 52913–52916. Frame-by-frame review showed both shoes above the court during an attacking clear. The upstream stroke tag did not call it a jump. The audited label adds the verified airborne interval at 52911–52917 and its context. The rescore kept the original 798 pose observations, appended 11 frames to cover the extended review, and froze all models and thresholds.

| Model with boundary recovery | Airborne frames on audited 0018 (of 122) | False flags (of 574) | Matched events (of 13) |
| --- | ---: | ---: | ---: |
| Single ranked checkpoint | 115 | 0 | 13 |
| Nine checkpoints with temporal augmentation | 118 | 0 | 13 |
| Boundary-weighted checkpoint | 117 | 0 | 13 |

All three rows use cutoffs selected on the five development matches. At its development cutoff for higher recall, the ensemble scored 121/122 with 11 false flags. The boundary-weighted model scored 122/122 with 15 false flags. The best diagnostic sweep under the false-positive limit on audited 0018 recovered 120/122. That cutoff needs a separate evaluation. The strict frame targets remain unmet at a common verified operating point.

Add `--temporal-augmentation` or `--boundary-weight 3` to the ranked training command for these variants. Pass all nine checkpoint paths before the `--dev` arguments to `verify_jump_events.py` to reproduce logit averaging.

## Learned takeoff and landing transitions

This experiment freezes the nine-checkpoint temporal ensemble and its event verifier. Once an event passes the verifier, the transition model fits a probability distribution over nearby takeoff and landing frames. Its inputs include the base score pattern, vertical hip motion, and the slower shoe's motion between adjacent observations. Training sums the probability assigned to every boundary consistent with the reviewed labels, including explicitly uncertain frames between confirmed contact and airborne states. At inference each event keeps its original score peak, and adjustments respect camera gaps and separation between accepted events.

Training and cutoff selection used six prior matches. Each development estimate held one match out of both the verifier and transition fit, with a separate inner fold for setting the verifier cutoff. The score-only and motion variants used separately calibrated takeoff and landing quantiles.

| Development method | Airborne frames detected (of 562) | False flags (of 2,390) | Matched events (of 61) |
| --- | ---: | ---: | ---: |
| Frozen verifier and boundary recovery | 537 | 12 | 60 |
| Learned transitions, motion features | 538 | 11 | 58 |

The independent [FineBadminton 0009 review](labels/finebadminton-0009.csv) covers 137 airborne frames, 577 negative frames, and 13 events at 30 fps. The 13th jump came from checking an independently sampled rally around 49976. The source tags the stroke as a reverse-slice drop and supplies no jump tag. It was labeled before either detector ran on the match.

| Method on 0009 | Airborne frames detected (of 137) | False flags (of 577) | Matched events (of 13) |
| --- | ---: | ---: | ---: |
| Frozen verifier and boundary recovery | 127 | 9 | 12 |
| Learned transitions | 128 | 9 | 12 |

Seven of the remaining refined false flags came from a single short recovery shuffle at 40403–40409. The transition experiment adjusts existing events, so it carries through incorrect event proposals of this kind. No change to the main detection path was made.

Run `refine_jump_edges.py CHECKPOINT ... --dev CACHE SOURCE --holdout CACHE SOURCE` with the same temporal ensemble used by `verify_jump_events.py`. Repeat `--dev` for 0003, 0033, 0063, 0010, 0013, and the audited 0018 cache.
