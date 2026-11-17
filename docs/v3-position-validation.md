# V3 partial-foot position validation

Related issue: https://github.com/aesthetic0001/badminton-analysis/issues/1

V3 prunes unreliable foot landmarks, uses available foot centres, and retains
unique players independently on each court half. Fully hidden players remain
missing; jump correction and temporal reconstruction are future work.

## Cached match evaluation

Reprocessed all 48,512 source frames in raw run
`f7d755110dc74ca7f882b074`, interval `[11701, 60213)`, without model inference.
Derived V3 analysis: `41d8e0863f6aa138065d3e2d`.

- Accepted video frames: 18,607.
- Retained player observations: 36,706, up from 34,056 (7.8%).
- Near-player observations: 18,595; far-player observations: 18,111.
- Complete-foot coordinates matched the previous calculation on 66,063 cached
  poses across all views.
- Six regression tests and the notebook's projection, motion, court, heatmap,
  and frame/context checks passed.
- Heatmap bin totals matched per-player observation counts. Missing-player CSV
  rows had zero dwell time. A 24-frame annotated replay export passed.

## Single-foot displacement measurement

On court-valid views, require all six toe/heel scores >= 0.4, retain a unique
eligible pose per half, and measure each projected foot centre's Euclidean
distance from the two-foot midpoint. Each foot has the same distance, so count
one measurement per player per frame. This measures deviation from the existing
projected midpoint, not ground-truth accuracy; jumps remain included.

| Minimum landmark score | Player observations | Median | 95th percentile | Above 20 cm |
| --- | ---: | ---: | ---: | ---: |
| 0.4 | 37,720 | 37.0 cm | 70.6 cm | 84.3% |
| 0.6 | 32,966 | 37.5 cm | 69.9 cm | 86.3% |

## Manual review by the user

Reviewed on November 16, 2026:

- The court-valid rejected-frame video contained 999 frames (33.3 seconds).
  The user identified these as serves or rally endings with stationary players,
  and confirmed they were appropriate to discard.
- The accepted single-player video contained 508 frames (16.9 seconds):
  496 near-only and 12 far-only. The user found these were mostly jumps;
  correcting jump positions is the next task.

Generated videos, raw detections, and derived arrays remain local artifacts.
These observations validate this reviewed match segment, not all videos.
