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

## Court-calibrated event verification

This experiment adds geometric evidence to the score-only verifier for the same frozen nine-checkpoint ensemble. The court homography and two manually marked net-post tops give a projection matrix for each match. For each proposed event, the verifier compares three fits to the observed hip midpoint. The fits share linear horizontal motion and allow either linear vertical motion, fixed Earth gravity, or free vertical curvature. The shoes just outside the proposal supply soft position anchors. The verifier also measures how often at least one shoe moves slowly after projecting the shoes onto the court plane. All fits keep the same proposals and boundary recovery.

In a follow-up on 0009 using six other development matches, geometry reduced false flags from 9/577 to 3/577 while both methods found 127/137 airborne frames. All six false flags from the score-only verifier on the recovery shuffle at 40404–40409 disappeared. The event count with timing tolerance fell from 12/13 to 11/13. The reduced false-positive rate was 0.52%.

The independent [FineBadminton 0036 review](labels/finebadminton-0036.csv) covers 14 jumps, 125 confirmed airborne player-frames, and 538 negative frames at 25 fps. It includes clips selected from shot tags and eight sampled rally intervals. One of the additional jumps occurred during a clear that carried no jump tag. The contact review preceded scoring. All binary-labeled frames received scores. Both verifiers used the same seven development matches and selected their cutoffs from leave-one-match-out scores. [FineBadminton source release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Verifier at its development cutoff for low false positives | Development airborne frames (of 699) | 0036 airborne frames (of 125) | 0036 false flags (of 538) | 0036 matched events (of 14) |
| --- | ---: | ---: | ---: | ---: |
| Score-only | 657 | 122 | 15 | 14 |
| Scores and court geometry | 662 | 122 | 20 | 14 |

The three misses were at takeoff or landing. Both verifiers flagged a nine-frame planted recovery at 61144–61152. At the development cutoff for recall, both found 124/125 airborne frames. The score-only verifier had 144 false flags and the geometry verifier had 66. A threshold sweep on 0036 put both methods at 120/125 with at most two false flags. The geometry verifier remains experimental.

Run `verify_jump_geometry.py CHECKPOINT ... --dev POSE_CACHE SOURCE VIDEO --holdout POSE_CACHE SOURCE VIDEO` with the nine temporal checkpoints. Repeat `--dev` for 0003, 0033, 0063, 0010, 0013, 0018, and 0009, then use 0036 as the holdout. [Net-post tops](jump-projection-poles.json) are recorded left to right in a 960-pixel-wide reference frame.

## Training on four previously reviewed matches

The nine-model temporal ensemble was refit with the reviewed frames from 0003, 0010, 0033, and 0009. This increased confirmed airborne training frames from 362 to 784 and confirmed negatives from 713 to 2,482. The earlier recovery shuffle on 0009 is flagged as a hard negative. The fit kept the original checkpoint initializations, ranking loss, boundary targets, and temporal augmentation. The control trained on the original data for 114 epochs, giving each model the same 2,280 optimizer updates as 40 epochs on the expanded data.

Each ensemble uses the same score-only verification and boundary recovery procedure. Its cutoff is chosen from leave-one-match-out scores on 0063, 0013, 0018, and 0036. The two new women's singles matches were excluded from model fitting, verifier fitting, and cutoff selection. Contact labels were assigned before scoring, using stroke tags and independently sampled rally intervals to locate clips. Sixteen of the 29 reviewed jumps had no jump tag on the corresponding stroke. [FineBadminton source release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| New holdout | Match | Native fps | Airborne frames | Negative frames | Uncertain frames | Jumps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [0022](labels/finebadminton-0022.csv) | Akane Yamaguchi vs Carolina Marín | 24.74 | 108 | 415 | 49 | 16 |
| [0028](labels/finebadminton-0028.csv) | Tai Tzu Ying vs An Se Young | 24.72 | 86 | 405 | 64 | 13 |

| Ensemble at its development cutoff for low false positives | Development airborne frames (of 402) | Holdout airborne frames (of 194) | Holdout false flags (of 820) | Holdout matched jumps (of 29) |
| --- | ---: | ---: | ---: | ---: |
| Original training data, 40 epochs | 362 | 174 | 9 | 26 |
| Original training data, 114 epochs | 374 | 178 | 9 | 26 |
| Expanded training data, 40 epochs | 389 | 185 | 19 | 26 |

All three ensembles recorded eight false flags among 1,736 development negatives at these cutoffs. The expanded refit recovered 100/108 airborne frames with 14 false flags on 0022 and 85/86 with five false flags on 0028. Every reviewed binary label received a score. All 19 false flags were within four native frames of a confirmed airborne frame for the same player. The refit missed all four confirmed airborne frames of the short far-player jump around 19602 on 0022.

At the development cutoff chosen for recall, the expanded refit found 192/194 holdout airborne frames (98.97%) with 56 false flags (6.83%). A diagnostic sweep of the two holdouts found 129/194 with at most four false flags for the expanded refit. The longer-run control reached 144/194 at the same false-positive limit. The strict targets remain unmet at a common verified cutoff.

To reproduce the refit, start with the original ranked training command and include `--practice --soft-boundaries --ranking-weight 1 --temporal-augmentation`, the combined BWF labels with mined negatives, and the 0001 training cache. Repeat `--fit-reviewed-source POSE_CACHE SOURCE` for 0003, 0010, 0033, and 0009. For the control, omit those four sources and pass `--epochs 114`. Run `compare_pose_refit.py ENSEMBLE_DIRECTORY ... --dev POSE_CACHE SOURCE --holdout POSE_CACHE SOURCE`, repeating the development and holdout options for each match. The comparison uses each ensemble's training report to check for overlap with evaluation footage.

Run `refine_jump_edges.py CHECKPOINT ... --dev CACHE SOURCE --holdout CACHE SOURCE` with the same temporal ensemble used by `verify_jump_events.py`. Repeat `--dev` for 0003, 0033, 0063, 0010, 0013, and the audited 0018 cache.

## RGB evidence around the shoes

A small shared convolutional encoder scores a pair of five-frame shoe clips. Each 32×24 RGB crop uses the center frame's toe and heel landmarks and scales with the player box height. The crop stays in the same image coordinates for all five frames so shoe motion relative to the local court remains visible. It includes space below the shoe for shadows. Models use the reviewed airborne labels and color, brightness, horizontal flip, and shoe masking augmentation.

Three models trained on the seven FineBadminton sources already used by the expanded pose refit. The visual training set contained 652 airborne frames, 2,047 negatives, and 347 soft boundary targets. The pose ensemble and event verifier were held fixed. The comparison adds the mean shoe logit to the verified pose logit. A weight of 0.25 was selected on 0063, 0013, 0018, and 0036, where it recovered 392/402 airborne frames with eight false flags among 1,736 negatives. Pose scores alone recovered 389/402 at eight false flags.

In a follow-up on 0022 and 0028, the selected visual blend recovered 186/194 airborne frames with 17/820 false flags. The pose verifier alone recovered 185/194 with 19/820 false flags. Match timing within tolerance improved from 26/29 to 27/29. At the blend's development cutoff for recall, it reached 193/194 with 47/820 false flags.

[FineBadminton 0042](labels/finebadminton-0042.csv) is a separate 30 fps match between Han Yue and Wang Zhi Yi, filmed on a red court. The review was completed before scoring. It covers 17 jumps with 154 confirmed airborne frames and 599 negatives. Stroke tags and eight sampled rally intervals guided the review. Three jumps came from strokes without a jump tag. All binary-labeled frames received scores. [FineBadminton release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Detector at its development cutoff for low false positives | 0042 airborne frames (of 154) | 0042 false flags (of 599) | Matched jumps (of 17) |
| --- | ---: | ---: | ---: |
| Verified pose | 144 | 10 | 14 |
| Verified pose with RGB shoe clips | 148 | 17 | 14 |

Six of the extra false flags came from a slight foot lift around a slice drop at 83253–83258. At its recall cutoff, the blend reached 153/154 with 93 false flags. A cutoff sweep on 0042 required 36 false flags to retain that recall and recovered 36/154 with at most two false flags. The visual blend remains experimental.

Use `prepare_shoe_rgb.py POSE_NPZ VIDEO VISUAL_CACHE/SOURCE.npz` to extract the shoe clips. Run `compare_shoe_rgb.py train VISUAL_CACHE OUTPUT --source POSE_CACHE SOURCE LABEL_SPLIT` for the seven pose-training sources 0001, 0057, 0070, 0003, 0010, 0033, and 0009. Use the split stored in each source's label CSV. Pass the output directory and the expanded pose checkpoint directory to `compare_shoe_rgb.py compare VISUAL_CACHE VISUAL_MODELS POSE_MODELS --dev POSE_CACHE SOURCE --holdout POSE_CACHE SOURCE`, repeating the match arguments.

## Reviewed recovery shuffles and hard-negative sampling

Three previously reviewed matches, 0022, 0036, and 0042, were added to the nine-model pose refit. The new training set contains 1,171 airborne frames, 4,034 negatives, and 507 soft targets. Of the negatives, 648 were classified as shuffles, lunges, or planted strokes. The nine-frame planted recovery at 61144–61152 on 0036 and the slight foot lift around 83253–83258 on 0042 are now marked as hard negatives. These edits preserve the frame-level airborne labels.

Two refits used the same data, architecture, seeds, and 40-epoch schedule. The only difference was the sampling weight assigned to reviewed hard negatives, 2 or 4 times the initial weight of an ordinary negative. The development matches were 0063, 0013, 0018, and 0028. At their development cutoffs for low false positives, the earlier refit recovered 355/363 airborne frames, the 2× refit recovered 350, and the 4× refit recovered 335. All three had 8/1,603 false flags. At the development recall target, the 2× refit reduced false flags from 36 to 22. The 4× refit had 34.

[FineBadminton 0025](labels/finebadminton-0025.csv) is a previously unused match between Gregoria Mariska Tunjung and Chen Yu Fei, at 24.72 fps. Review covered shot candidates and six sampled rally intervals before scoring. It identified 18 jumps, including five without a jump tag, with 145 confirmed airborne frames, 474 negatives, and 66 uncertain frames. All binary-labeled frames received predictions. [FineBadminton release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Ensemble at its development cutoff for low false positives | Airborne frames (of 145) | False flags (of 474) | Matched jumps (of 18) |
| --- | ---: | ---: | ---: |
| Earlier refit | 130 | 11 | 16 |
| Expanded refit, 2× hard-negative weight | 117 | 9 | 14 |
| Expanded refit, 4× hard-negative weight | 122 | 9 | 15 |

Six of the earlier refit's false flags fell immediately after the switch from a player close-up to the wide court view at frame 11579. For a diagnostic rescore, mark the six decoded context frames 11573–11578 as invalid in the holdout sequence's existing `court_valid` mask. At the frozen cutoffs, all three models then had 5/474 false flags with the same airborne counts. The audit used a manually verified camera cut on the scored match. The remaining false flags fell near takeoff and landing. This match still falls short of the joint recall and false-positive targets.

For the refits, add 0022, 0036, and 0042 to the earlier `--fit-reviewed-source` arguments and pass `--hard-negative-weight 2` or `--hard-negative-weight 4`. Evaluate with `compare_pose_refit.py` on the four development matches above and the 0025 holdout.

## Court visibility around broadcast cuts

The pose pipeline already uses a court-validity mask to exclude context from other camera views. An optional cache check populates this mask using the existing court-line agreement score against the match's reference homography. On 0025, the check identified the same six close-up frames as the manual review. A cutoff of 0.45 also rejected seven valid frames on 0022 during a score overlay. A cutoff of 0.40 retained those seven and still rejected the six close-up frames on 0025. This cutoff was chosen before reviewing or scoring 0024.

[FineBadminton 0024](labels/finebadminton-0024.csv) is a 24.73 fps match between Loh Kean Yew and Toma Junior Popov. Contact review preceded scoring and covered 13 jumps, six sampled rally intervals, and 80 confirmed grounded player-frames immediately after five returns to the wide court view. One jump had no jump tag. The full review contains 117 airborne, 504 negative, and 63 uncertain player-frames. All binary-labeled frames received scores in both runs. The sample includes extra post-cut frames to measure the effect of those cuts. [FineBadminton release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Court-validity mask on 0024 | Airborne frames (of 117) | False flags (of 504) | False flags just after a cut (of 80) | Matched jumps (of 13) |
| --- | ---: | ---: | ---: | ---: |
| Observed poses | 113 | 25 | 24 | 12 |
| Observed poses with court agreement of at least 0.40 | 113 | 1 | 0 | 12 |

Both runs used the frozen expanded-reviewed-round1 pose ensemble, the score-only verifier fit on 0063, 0013, 0018, and 0028, and the same score cutoff (0.955325862163521). The court check reached 0.20% false positives and 96.58% airborne-frame recall on 0024.

Use `check_pose_court.py VIDEO POSE_NPZ REFERENCE_FRAME FILTERED_CACHE/SOURCE.npz --minimum 0.40` to generate the filtered cache. Run `compare_pose_refit.py` with the four development sources above and score the original and filtered holdout caches separately. The reference frame for 0024 is 22133.

## Boundary parameter sweep after the broadcast-cut check

The four audited matches below use the frozen pose ensemble and the court check at 0.40 where it changed the pose context. All 2,516 binary labels received scores. Every false flag at the development cutoff fell in a reviewed jump's setup or recovery, including a grounded frame just before takeoff at 16395 on 0024. The four missed frames on 0024 came from the far player's landing at 8923–8925 and the near player's takeoff at 22074. The raw pose scores at those four frames were 0.361, 0.029, 0.003, and 0.372.

| Audit match | Airborne frames found | False flags | Matched jumps with timing allowance |
| --- | ---: | ---: | ---: |
| 0022 | 101/108 | 14/415 | 13/16 |
| 0042 | 146/154 | 10/599 | 14/17 |
| 0025 | 130/145 | 5/474 | 16/18 |
| 0024 | 113/117 | 1/504 | 12/13 |
| Pooled | 490/524 | 30/1,992 | 55/64 |

Leave-one-match-out scoring on 0063, 0013, 0018, and 0028 compared 1,824 combinations of existing verifier settings. The grid crossed native versus 30 fps input windows, proposal thresholds of 0.20, 0.30, 0.40, and 0.50, verifier penalties of 2 and 8, verifier weights of 0.25, 0.50, and 1, and verifier extensions of zero or one frame. Boundary recovery used either no extension or takeoff extensions of 1–3 frames, landing extensions of 1–2 frames, and minimum supporting pose scores of 0.25, 0.40, or 0.55. All extensions are expressed at 30 fps. The existing configuration tied for best at 355/363 development airborne frames under the false-positive limit and was retained. At the retained configuration, a diagnostic pooled audit threshold sweep recovered at most 245/524 airborne frames under 0.5% false positives and needed 318/1,992 false flags to reach 99% recall. Those audit cutoffs were diagnostic.

The existing takeoff and landing refiner was also tested using nested development folds. Tested quantiles were 0.1 through 0.9 on both edges, with and without the existing motion features. The best settings under the development false-positive limit matched the nested baseline at 351/363 airborne frames and 7/1,603 false flags. At the selected score-only quantiles of 0.6 for takeoff and 0.3 for landing, the refiner recovered one additional airborne frame on 0042. Pooled false flags and duration-tolerant event counts were unchanged. The detector still misses both joint targets across these matches.

Two refits held the training sources, seeds, 40-epoch schedule, and architecture fixed while raising the relative weight of the first and last two confirmed airborne frames. Verifier fitting and cutoff selection used the same development matches as the grid above. At each refit's development cutoff for low false positives, both weighting changes reduced recall on the four audit matches.

| Airborne boundary weight | Development airborne frames (of 363) | Audit airborne frames (of 524) | Audit false flags (of 1,992) | Audit matched jumps (of 64) |
| --- | ---: | ---: | ---: | ---: |
| 1× | 355 | 490 | 30 | 55 |
| 2× | 350 | 469 | 25 | 54 |
| 4× | 351 | 462 | 27 | 51 |

All three settings had eight false flags among 1,603 development negatives. At the separate cutoff selected for 99% development recall, the 1×, 2×, and 4× runs found 512, 509, and 511 audit airborne frames with 81, 96, and 84 false flags, respectively. Use the four-match expanded refit command with `--boundary-weight 2` or `--boundary-weight 4` to reproduce the variants. The 1× weights remain the baseline.

## Mendeley backhand-drive negatives

The four files `labels/mendeley-backhand-drive-{1,2,3,4}.csv` add 152 reviewed negative player-frames across four broadcasts, including planted strokes, shuffles, and recovery steps. They map to source files `backhand_drive (87).mp4`, `(52).mp4`, `(104).mp4`, and `(89).mp4` in `Dataset/Backhand_drive/`. Convert each clip to 30 fps with the FFmpeg `fps=30` filter before applying the intervals. These are negative examples for the pose benchmark, and the same clips appeared as unlabeled footage in the earlier RGB pretraining experiment. [Mendeley source release](https://data.mendeley.com/datasets/3sp4xntp34/1)

Score with the same frozen pose model, verifier, and cutoff as the boundary audit. The calibrated crop on clip 3 mistook the near player for the far player due to a bad court fit. The standard uncalibrated player detector recovered both slots when the court reference was omitted for that clip. The resulting comparison covered all 152 labeled frames with zero false flags. The 30 fps normalized clips include duplicated source frames and the labels are clustered within four excerpts.

## Fresh match after selecting boundary settings

[FineBadminton 0043](labels/finebadminton-0043.csv) shows Kunlavut Vitidsarn and Jonatan Christie on a red court at 30 fps. The review covered twelve jump-smash tags balanced across player sides and match time, three other overhead strokes, and six sampled rally intervals. The footage contained fourteen confirmed jumps, including one without a jump tag. Foot contact was reviewed before scoring. There are 165 airborne frames, 491 negatives, and 94 uncertain player-frames. [FineBadminton source release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Airborne boundary weight at its development cutoff for low false positives | Airborne frames found (of 165) | False flags (of 491) | Matched jumps (of 14) |
| --- | ---: | ---: | ---: |
| 1× | 138 | 5 | 11 |
| 2× | 115 | 3 | 9 |
| 4× | 116 | 4 | 10 |

All binary-labeled frames received scores. Every cached frame passed the 0.40 court check, and all five baseline false flags were in grounded recovery. The baseline missed the entire ten-frame near-side jump at 17511–17520. At the baseline cutoff selected for 99% development recall, 0043 reached 156/165 with 9/491 false flags and 13/14 matched jumps with timing allowance. A diagnostic cutoff sweep on 0043 reached at most 102/165 airborne frames below 0.5% false positives. The fresh match remains below the strict recall target and above the false-positive target at the development cutoff.

## Repeated broadcast frames in pose windows

Frame differencing on the court region showed that several nominally 30 fps broadcasts repeat one image in each six-frame cycle. The repeated source frame satisfies `frame % 6 == 1` on 0042 and 0043, and `frame % 6 == 5` on 0063. The phase varies by video. Near and far player slots were present throughout the missed 0043 jump at 17511–17520. The input included repeated images at 17515 and 17521, and the tracked hip moved down the image during most of the jump as the player advanced toward the camera.

The optional cadence setting removes the repeated images when assembling the existing 13-frame pose inputs. It maps the predicted scores back to all original source frames before verification and event scoring. This preserves native labels, time offsets, score coverage, and the court validity mask at gaps. The model checkpoint is unchanged. At the original verifier and cutoff, the setting recovered seven of the ten previously missed airborne frames on 0043.

| Frozen pose ensemble at its development cutoff | Airborne frames (of 689) | False flags (of 2,483) | Matched jumps (of 78) |
| --- | ---: | ---: | ---: |
| Original input windows | 628 | 35 | 66 |
| Skip repeats on 0042 and 0043, original verifier and cutoff | 643 | 36 | 68 |
| Skip repeats on 0063, 0042, and 0043, refit verifier and recalibrate | 645 | 36 | 68 |

The pooled audit covers 0022, 0024, 0025, 0042, and 0043. The second row raised detections from 146 to 150 on 0042 and from 138 to 149 on 0043. The only additional false flag was on 0042. The recalibrated third row uses the usual four development matches and reaches 93.61% strict recall and 1.45% false positives on that audit. At the development cutoff for higher recall, it reached 671/689 with 103 false flags. All binary labels received scores. A 1.2-frame spacing adjustment on 0042 and 0043 recovered more airborne frames at the original cutoff, with eight more false flags on 0042. The repeat-removal configuration remains an offline experiment pending validation on new broadcast footage.

Run `compare_pose_refit.py` with the same caches and development sources as above and add `--repeat-cadence finebadminton-0042 6 1 --repeat-cadence finebadminton-0043 6 1`. For the recalibrated row, also add `--repeat-cadence finebadminton-0063 6 5`. Use this override for a source only after checking its actual repetition pattern.

## Mendeley portrait stroke negatives

Six portrait excerpts add 186 reviewed grounded player-frames and nine ambiguous frames. The files `labels/mendeley-backhand-clear-{1,2,3,4,5}.csv` use the same numbered clips under `Dataset/Backhand_clear/`, and `labels/mendeley-forehand-smash-1.csv` uses `Dataset/Forehand_Smash/Forehand_Smash (1).mp4`. The reviewed figures use the near player and copies normalized with the FFmpeg filter `fps=30,scale=960:-2`. These excerpts share a camera setup, so they provide ground-contact examples of different strokes and recovery steps in that view. [Mendeley source release](https://data.mendeley.com/datasets/3sp4xntp34/1)

Use `"practice": true` in each source manifest entry for `prepare_pose.py`. Add `"practice_max_foot_y": 0.94` on backhand clear 3 to cover the foreground player at the bottom of the portrait view. This restored scores for nine previously unscored labels. With full coverage, the frozen pose baseline had four false flags among 186 negatives, all on frames 10–13 during a grounded shuffle in backhand clear 3. The other five clips had zero false flags at the same cutoff.

## Training the pose ensemble on repeat-free windows

Two refits used the same pose model, three seeds for each of three initializations, and 40-epoch schedule as the expanded reviewed baseline. The first kept all training labels and removed periodic repeats from their input windows. The second also added the 338 grounded labels from the Mendeley backhand-drive and portrait excerpts. Each refit uses the same 784 confirmed airborne labels. The extra Mendeley samples add optimizer steps within the fixed epoch schedule. The comparisons use repeat-free inference on 0063, 0042, and 0043 for every checkpoint.

Apply `--repeat-cadence SOURCE 6 PHASE` to the four-match expanded training command for source-phase pairs 0001/1, 0003/1, 0009/3, 0033/2, 0057/4, and 0070/1. Add 0063/5 to use the same convention for the training report. Original frame indices and labels remain on the source timeline. The six phases were verified by decoded frame comparisons over the pose caches. The near-duplicate frame pairs on 0010 did not follow a stable six-frame phase in the reviewed footage, so the original cadence was used there.

[FineBadminton 0044](labels/finebadminton-0044.csv) is a separate 25 fps excerpt featuring Lakshya Sen and Jia Heng Jason Teh. Nine tagged jump windows, one additional overhead stroke, and five sampled rally windows were reviewed before scoring. It has 87 confirmed airborne, 341 grounded, and 94 uncertain player-frames. Only the original four development matches were used to fit the verifier and choose cutoffs. [FineBadminton source release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Pose training run at its development cutoff for low false positives | Development airborne frames (of 363) | Earlier audit airborne frames (of 689) | Earlier audit false flags (of 2,483) | 0044 airborne frames (of 87) | 0044 false flags (of 341) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen checkpoint with repeat-free inference | 355 | 645 | 36 | 83 | 8 |
| Repeat-free training, ranking weight 1 | 354 | 645 | 41 | 82 | 7 |
| Repeat-free training with Mendeley negatives, ranking weight 1 | 351 | 624 | 36 | 82 | 7 |
| Repeat-free training, ranking weight 0 | 351 | 619 | 37 | 81 | 8 |
| Repeat-free training, ranking weight 0.2 | 355 | 634 | 37 | 82 | 8 |

The training runs with lower ranking weight used the same training set as the repeat-free run without Mendeley. Every 0044 binary label received a score and every run matched all nine jumps with the existing timing allowance. The baseline recovered 83/87 airborne frames (95.40%) with 8/341 false flags (2.35%). Four airborne misses were on far-side takeoff. Seven of the eight false flags came from a near-side overhead stroke with a planted support shoe, frames 3957–3963. The court check excluded zero frames around the reviewed intervals, and decoded footage showed no periodic repeats, so 0044 used native input cadence. At the separate cutoff selected for 99% development recall, the baseline recovered 85/87 and flagged 15/341 on 0044. A diagnostic 0044 threshold sweep reached at most 83/87 below 0.5% false positives. The frozen checkpoint remains the reference across these trials.

## Boundary errors and planted overhead strokes

At the fixed cutoff for low false positives, the repeat-aware baseline scored all binary labels across 0022, 0024, 0025, 0042, 0043, and 0044. Of 48 missed airborne frames, 18 were at takeoff and 19 at landing within the per-jump timing allowance. Seven were outside that allowance on jumps with some correct detections. The remaining four were the complete far-side jump at 19601–19604 on 0022. Of 44 false flags, 23 were near a same-player jump within its timing allowance, including uncertain contact frames in the boundary range. The other 21 occurred away from any such boundary. The strict frame scores continue to charge all 48 misses and 44 false flags.

On 0044, the existing geometry verifier measured a slow support shoe in 7 of 9 observed frame transitions for the false overhead proposal at 3957–3964. Its free fitted hip rise was 0.14 m. The true near-side jump proposal at 4092–4103 had no slow-support transitions and a 0.39 m fitted rise. Both use the existing 0.75 m/s definition of slow shoe motion. Feeding the motion features into the regularized verifier left all seven planted-stroke false flags on 0044.

An exploratory setting subtracts four from the score-only logit within a candidate span if at least 70% of observed transitions have a slow support shoe and the fitted rise is at most 0.18 m. These values were chosen after inspecting the 0044 case. They left the development operating point unchanged and reduced that match from eight false flags to one, with 83/87 airborne frames recovered in both runs. This is a diagnostic result on the match that prompted the setting. The rule uses the existing geometry features and frozen pose ensemble. Net-post tops for the new comparisons are recorded in the 960-pixel-wide reference view.

[FineBadminton 0045](labels/finebadminton-0045.csv) is a separate 25 fps excerpt of Jia Heng Jason Teh against Priyanshu Rajawat, recorded at the same event as 0044. The frame review preceded scoring and covered eight jump-tagged strokes, eight other overhead-stroke candidates, and four sampled rally windows. It found ten confirmed jumps, including two without jump tags, with 95 airborne, 465 grounded, and 83 uncertain player-frames. The three settings below used the same frozen checkpoint and the existing development matches 0063, 0013, 0018, and 0028. All binary labels received scores. [FineBadminton source release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Verifier at its development cutoff for low false positives | 0044 airborne frames (of 87) | 0044 false flags (of 341) | 0045 airborne frames (of 95) | 0045 false flags (of 465) | 0045 matched jumps (of 10) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Score-only | 83 | 8 | 90 | 17 | 8 |
| Learned geometry weights | 83 | 8 | 90 | 17 | 8 |
| Score-only with support penalty | 83 | 1 | 90 | 17 | 8 |

On 0045, 14 false flags came during two planted overhead strokes or their immediate recovery. The two proposals had fitted rises of about 0.15 m, but the slow-support fractions were 0.33 and 0.00, below the 0.70 penalty threshold. The other three false flags came just after a separately labeled short jump. At the development cutoff for higher recall, 0045 recovered 93/95 airborne frames with 30 false flags for each setting. The support penalty remains an experiment.

To reproduce, run `verify_jump_geometry.py` with the nine expanded-reviewed-round1 checkpoints and `--dev POSE_CACHE SOURCE VIDEO` for the four development matches above. Include `--repeat-cadence finebadminton-0063 6 5 --support-penalty 0.70 0.18 4`. Use the court-checked pose cache and corresponding video in a separate `--holdout POSE_CACHE SOURCE VIDEO` run for each excerpt.

## Individual foot landmarks as support evidence

A 72-setting diagnostic varied how the existing geometry check measures a planted shoe. It compared the mean of heel and toe coordinates, any individual foot landmark per transition, and one consistent landmark across the proposal. The grid crossed speed limits of 0.75, 1.25, 1.75, and 2.25 m/s, transition lengths of one and two native frames, and support fractions of 0.60, 0.70, and 0.80. It kept the 0.18 m maximum fitted rise and four-logit penalty. Individual landmarks needed confidence of at least 0.35, with the consistent-landmark version requiring valid measurements for at least 60% of the proposal transitions and at least two measurements.

The six calibrated audits 0024, 0025, and 0042–0045 contain 763 airborne and 2,874 grounded frames. The score-only baseline found 717 airborne frames with 47 false flags. The strongest consistent-landmark setting that preserved those counts of airborne detections used a two-frame difference, a 1.75 m/s speed limit, and an 80% support fraction. It reduced false flags to 40, all from the already identified 0044 case. It left the 0.5% false-positive development cutoff unchanged, and reduced development false flags from 37 to 32 at the separate cutoff for 99% recall. A more permissive setting (2.25 m/s and 60% support) removed seven additional false flags on 0045 and lost 16 airborne detections on other sources. The original mean-shoe penalty also gives 717 airborne detections and 40 false flags on the same audit.

Of the 47 baseline false flags across those six sources, 28 were in proposals that contained no correctly flagged airborne frame. The other 19 shared a proposal with at least one correctly flagged airborne frame. Even rejecting all 28 isolated errors would leave 19/2,874 false flags (0.66%) if existing airborne detections are preserved. This limits an approach that only accepts or rejects an entire proposal. When a separate uniform score penalty is allowed for each proposal, an oracle using ground truth could reduce the 47 flags to 10 while keeping the original 717 detections. That diagnostic uses the labels to choose each penalty and leaves the original recall shortfall.

[FineBadminton 0046](labels/finebadminton-0046.csv) is a separate 25 fps excerpt from the same event, featuring Lakshya Sen and Shogo Ogawa. The blinded frame review sampled eight tagged jumps, five other overhead strokes, and four ordinary rally windows. It found 81 airborne, 408 grounded, and 61 uncertain player-frames. Both players had poses in every extracted frame and every binary label received a score. The operating thresholds came from the same four development matches, and the contact settings tested here were chosen before scoring 0046. [FineBadminton source release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Verifier at its development cutoff for low false positives | 0046 airborne frames (of 81) | 0046 false flags (of 408) | Matched jumps (of 8) |
| --- | ---: | ---: | ---: |
| Score-only | 79 | 22 | 8 |
| Mean-shoe support penalty | 79 | 22 | 8 |
| Consistent-landmark support penalty | 79 | 16 | 8 |
| Learned geometry weights with consistent-landmark support | 80 | 14 | 8 |

At the separate development cutoff for high recall, the score-only verifier recovered all 81 with 41 false flags, and the consistent-landmark penalty recovered all 81 with 29 false flags. A diagnostic sweep of the consistent-landmark penalty on 0046 recovered at most 75/81 below 0.5% false positives. Run the previous geometry command with the 0046 court-checked cache and video, adding `--landmark-support-speed 1.75 --support-penalty 0.80 0.18 4` to reproduce the two consistent-landmark rows. The learned verifier using mean-shoe support also reached 80/81 with 14 flags on 0046. On the six earlier audits, learned geometry with consistent landmarks found 718/763 with 48 flags, compared with 717/763 and 47 flags for the score-only verifier. The landmark option remains experimental.

## Refit on confirmed overhead strokes

The Lucknow excerpts 0044–0046 add 263 airborne and 1,214 grounded frames to the repeat-aware training set. The 1,184 reviewed hard negatives in the expanded set include planted overhead strokes, shuffles, and lunges. The two refits differ only in how heavily they sample those negatives. Both use the original 40-epoch schedule and the same three seeds for each of three initializations. The four development matches and earlier five audit matches are separate from the added training footage.

| Pose run at its development cutoff for low false positives | Development airborne frames (of 363) | Audit airborne frames (of 689) | Audit false flags (of 2,483) |
| --- | ---: | ---: | ---: |
| Frozen baseline, repeat-free inference | 355 | 645 | 36 |
| Same training set with repeat-free inputs | 354 | 645 | 41 |
| Added 0044–0046, 2× hard-negative sampling | 352 | 624 | 42 |
| Added 0044–0046, 4× hard-negative sampling | 334 | 603 | 31 |
| Frozen baseline with 12.5% of the 2× refit logits | 355 | 648 | 36 |

All runs had 8/1,603 development false flags at their cutoffs for low false positives. At their separate cutoffs for 99% development recall, the 2× and 4× refits reduced the audit false-flag count from the frozen baseline's 103 to 79 and 70, with 667 and 665 airborne detections versus the baseline's 671. To select an ensemble blend, compare refit weights from zero to one in steps of 0.125 on development scores. Rank by most airborne frames recovered under the false-positive limit, then fewest flags at the separate cutoff for 99% recall, then the smaller weight. This selected the 2× refit at weight 0.125 before the new match was scored.

To reproduce, add the court-checked 0044, 0045, and 0046 caches with `--fit-reviewed-source` to the repeat-aware training command, retaining `--hard-negative-weight 2` for the selected refit. Use `compare_pose_refit.py` with the original expanded-reviewed-round1 ensemble and `--blend-refit REFIT_DIRECTORY 0.125` to fit and evaluate the score verifier on the combined logits.

[FineBadminton 0051](labels/finebadminton-0051.csv) features Su Li Yang and Weng Hong Yang on a different broadcast court. Twelve jumps, including four without jump tags, were found in eight tagged jump windows, six other overhead windows, and four sampled rally windows. The independent review contains 123 airborne, 397 grounded, and 103 uncertain player-frames. The source is 30 fps and the court-region audit found the repeating image at phase five of the six-frame cycle. The benchmark uses `--repeat-cadence finebadminton-0051 6 5` for all models, with scores mapped back to the labeled frames. [FineBadminton source release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

The existing takeoff and landing refiner was also rerun with 0044–0046 added to the four development matches. With nested folds, the score verifier recovered 496/626 airborne frames and had 19 false flags among 2,817 grounded frames. Across 162 combinations of the two edge quantiles and the optional motion features, no setting reached fewer than 0.5% false positives in those out-of-fold predictions. The boundary refiner was kept out of the 0051 comparison.

| Pose run at its development cutoff for low false positives | 0051 airborne frames (of 123) | 0051 false flags (of 397) | Matched jumps (of 12) |
| --- | ---: | ---: | ---: |
| Frozen baseline | 118 | 13 | 12 |
| Same training set with repeat-free inputs | 119 | 12 | 12 |
| Added overhead footage, 2× hard-negative sampling | 116 | 11 | 11 |
| Added overhead footage, 4× hard-negative sampling | 109 | 8 | 10 |
| Frozen baseline with 12.5% of the 2× refit logits | 118 | 13 | 12 |

Every binary label received a score. All five baseline misses were at takeoff. Nine of its 13 false flags were consecutive frames of a far-side planted overhead at 130737–130745. The individually calibrated repeat-free refit reduced that match's false flags by one, but had more false flags than the frozen baseline across the five earlier audits. At the separate cutoffs for 99% development recall, every variant recovered 121/123 on 0051, with 17–20 false flags. The diagnostic cutoff sweep of the baseline on 0051 recovered at most 96/123 below 0.5% false positives. The ensemble blend remains experimental.

## Player-box sampling and pose smoothing

The court-calibrated pose extractor normally runs the player detector on every second cached frame and uses the nearest available box for each RTMPose call. The comparison used `prepare_pose.py --court-detection-stride 1` to estimate a new box on every cached frame for 0045, 0046, and the original 0051 review windows. The frozen baseline, court checks, verified repeat patterns, score verifier, and development cutoff stayed fixed. Both extractions scored every binary label.

| Audited match | Airborne frames, stride 2 | Airborne frames, stride 1 | False flags, stride 2 | False flags, stride 1 |
| --- | ---: | ---: | ---: | ---: |
| 0045 | 90/95 | 90/95 | 17/465 | 18/465 |
| 0046 | 79/81 | 78/81 | 22/408 | 21/408 |
| 0051 | 118/123 | 120/123 | 13/397 | 14/397 |
| Pooled | 287/299 | 288/299 | 52/1,270 | 53/1,270 |

On the planted 0051 overhead at 130737–130745, median one-frame displacement fell from 21–35 pixels to 4–6 pixels across the six heel and toe landmarks with per-frame crops. The classifier continued to flag the stroke, including when the existing verifier used consistent heel or toe support. At the development cutoff for low false positives, the learned geometry verifier gave 118/123 airborne and 12/397 false flags with stride 2, versus 120/123 and 14/397 with stride 1. The extractor retains the stride-2 default.

Eight offline smoothing settings varied centered median versus triangular-weighted mean filters, three versus five frames, and feet versus all keypoints. The filters used same-view confident landmarks after known repeats were removed. The verifier was recalibrated on the same four development matches for each setting. Across nine audited matches at the development cutoff for low false positives, the unfiltered baseline found 1,015/1,075 airborne frames and flagged 96/4,094 grounded frames. The three-frame median on feet gave the fewest false flags (82), with 990 airborne detections. The three-frame median on all landmarks recovered 1,016 airborne frames with 100 false flags. The five-frame median on all landmarks also recovered 1,016 with 106 false flags. All eight smoothing settings fell short of both strict targets.

For the supplemental 0051 audit, seven additional overhead windows were selected by shot type and player side without viewing predictions. These included two genuine jumps without jump tags. The review added 18 confirmed airborne, 130 grounded, and 62 uncertain frames. On these new windows the frozen baseline at its previous cutoff found all 18 airborne frames and flagged 1/130 grounded frames, immediately after landing. Combined with the original review, it found 136/141 airborne frames and flagged 14/527 grounded frames. All 14 jumps matched with timing allowance. The original 0051 label counts and box-stride row refer to the version in commit `d524dbe`. To assemble the expanded evaluation cache with the same stride parity, extract the original and supplemental windows separately and concatenate the two disjoint sets of pose and court-validity rows in source-frame order.

A controlled training rerun omitted soft targets for uncertain contact frames. It kept the same training sources, confirmed labels, seeds, cadence handling, and epoch schedule as the repeat-free refit. On the nine original audits, both runs recovered 1,016/1,075 at their own development cutoffs for low false positives. Omitting the soft targets reduced false flags from 109 to 100 and increased timing-tolerant matches from 106 to 108. The frozen baseline recovered 1,015 with 96 false flags and 105 timing-tolerant matches. The confirmed-only rerun remains short of the strict targets.

## Sampling difficult airborne frames

The overhead refit with 2× hard-negative sampling was repeated with three weights for difficult airborne frames. All four refits used the same data, three seeds per initialization, and 40 epochs. During the second half of training, `--hard-positive-weight` upweights confirmed airborne windows at or below the current 25th-percentile positive logit. The sampler then rescales the positive weights to preserve their total mass. The ranking loss and hard-negative sampling use the previous settings. The default positive multiplier is 1. To reproduce a weighted run, add `--hard-positive-weight 1.25`, `2`, or `4` to the preceding overhead training command.

The four development matches select the verifier and both operating thresholds for each ensemble. The six earlier audits are 0022, 0024, 0025, 0042, 0043, and expanded 0051. All audit labels received scores.

| Pose run | Development airborne detections under the false-positive limit (of 363) | Earlier audit detections and flags at that cutoff (830 airborne, 3,010 grounded) | Earlier audit detections and flags at the 99% development recall cutoff |
| --- | ---: | ---: | ---: |
| Frozen baseline | 355 | 781 / 50 | 810 / 121 |
| Overhead refit, 1× | 352 | 758 / 54 | 806 / 99 |
| Overhead refit, 1.25× | 349 | 750 / 57 | 801 / 88 |
| Overhead refit, 2× | 337 | 717 / 54 | 806 / 102 |
| Overhead refit, 4× | 271 | 544 / 35 | 813 / 165 |

At the cutoff for low false positives, the 2× refit missed 36/184 takeoff frames, 44/182 landing frames, and 33/464 interior frames. The matching 1× refit missed 23, 30, and 19 respectively. Each boundary covers up to two confirmed frames per jump, with takeoff taking precedence for any overlap. The 4× run reduced false flags at this cutoff but lost 214 airborne detections compared with 1×. The baseline remains the frozen reference.

[FineBadminton 0054](labels/finebadminton-0054.csv) is a different broadcast featuring Viktor Axelsen and Lei Lan Xi. The independent review covered eight jump-tagged windows, six other overhead windows, and four sampled rally windows. It contains 13 confirmed jumps, six without jump tags, with 128 airborne, 418 grounded, and 124 uncertain player-frames. The 30 fps footage contains repeated images at phase three of the six-frame cycle. Use `--repeat-cadence finebadminton-0054 6 3` with the court-checked pose cache and include 0054 only as a holdout. All 786 extracted frames passed the reference court check, and all binary labels received scores. [FineBadminton source release](https://huggingface.co/datasets/iLearn-Lab/Finebadminton-20K)

| Pose run | 0054 detections and flags at the low-false-positive development cutoff | 0054 matched jumps at that cutoff (of 13) | 0054 detections and flags at the 99% development recall cutoff |
| --- | ---: | ---: | ---: |
| Frozen baseline | 125 / 5 | 12 | 128 / 9 |
| Overhead refit, 1× | 121 / 5 | 11 | 128 / 7 |
| Overhead refit, 1.25× | 124 / 6 | 11 | 128 / 8 |
| Overhead refit, 2× | 112 / 6 | 8 | 128 / 12 |
| Overhead refit, 4× | 92 / 6 | 6 | 128 / 15 |

All five baseline false flags followed landing on the far side. Its three misses were two far-side takeoff frames and one far-side landing frame. A diagnostic threshold sweep on 0054 recovered at most 114/128 below 0.5% false positives. Across the seven audits, the frozen baseline recovered 906/958 airborne frames and flagged 55/3,428 grounded frames at the fixed cutoff for low false positives. It matched 94/105 jumps using the duration-dependent timing allowance. A pooled diagnostic sweep of the frozen scores recovered at most 630/958 after enforcing the false-positive limit. At the highest cutoff still recovering 99% of the pooled airborne frames, it flagged 539/3,428 grounded frames. This sweep uses audit labels to measure score separation.

A separate diagnostic adjusted the frozen ensemble's mean logit by its per-frame logit standard deviation across the nine member checkpoints. Multiplier candidates were −0.5, −0.25, −0.125, 0, 0.125, 0.25, and 0.5. Selecting by most airborne detections under the development false-positive limit chose +0.5, with 356 detections versus 355 for the mean. On the seven audits, this setting recovered 892/958 and flagged 53/3,428 at its own low-false-positive cutoff, compared with the mean's 906 and 55. The ensemble retains mean logits.
