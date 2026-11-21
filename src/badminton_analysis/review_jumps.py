"""Build a human review queue from a position analysis with jump predictions."""

import argparse
import csv
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.jumps import HALF_WINDOW, jump_segments
from badminton_analysis.observation_store import ObservationStore


PLAYERS = ('near', 'far')


@dataclass(frozen=True)
class Candidate:
    player: int
    focus: int
    start: int
    stop: int
    score: float


def _reviewed_mask(paths, frame_ids):
    """Skip reviewed targets and keep all training windows away from existing validation."""
    excluded = np.zeros((len(frame_ids), 2), dtype=bool)
    for path in paths:
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                start, stop = int(row['start_frame']), int(row['stop_frame'])
                if not frame_ids[0] <= start < stop <= frame_ids[-1] + 1:
                    raise ValueError(f'Reviewed interval falls outside this source: {row}')
                player = PLAYERS.index(row['player'])
                before, after = np.searchsorted(frame_ids, [start, stop])
                if row['split'] == 'validation':
                    excluded[max(0, before-2*HALF_WINDOW):after+2*HALF_WINDOW] = True
                else:
                    excluded[before:after, player] = True
    return excluded


def _pick_candidates(scores, excluded, fps, threshold, per_group, seed):
    before = round(.55 * fps)
    after = round(.70 * fps)
    if len(scores) < before + after + 1:
        return []
    margin = min(.15, threshold / 2, (1-threshold) / 2)
    occupied = np.zeros(len(scores), dtype=bool)
    likely, borderline, audit = [], [], []
    for player, start, stop in jump_segments(scores, threshold-margin):
        peak = start + int(np.argmax(scores[start:stop, player]))
        event = Candidate(player, peak, start, stop, float(scores[peak, player]))
        (likely if event.score >= threshold+margin else borderline).append(event)
    likely.sort(key=lambda item: item.score, reverse=True)
    borderline.sort(key=lambda item: abs(item.score-threshold))

    rng = np.random.default_rng(seed)
    for player in range(len(PLAYERS)):
        quiet = np.isfinite(scores[:, player]) & (scores[:, player] < threshold-margin)
        window = np.ones(before + after + 1, dtype=int)
        centers = np.flatnonzero(np.convolve(quiet.astype(int), window, mode='valid') == len(window))
        for center in centers + before:
            audit.append(Candidate(player, int(center), int(center), int(center)+1,
                                   float(scores[center, player])))
    rng.shuffle(audit)

    result = []
    for kind, pool in (('likely_jump', likely), ('near_threshold', borderline), ('quiet_audit', audit)):
        queues = [iter([item for item in pool if item.player == slot]) for slot in range(len(PLAYERS))]
        group = []
        while len(group) < per_group:
            added = False
            for slot, queue in enumerate(queues):
                for item in queue:
                    start = max(0, min(item.start, item.focus-before))
                    stop = min(len(scores), max(item.stop, item.focus+after+1))
                    if occupied[start:stop].any() or excluded[start:stop, slot].any():
                        continue
                    if np.isfinite(scores[start:stop, slot]).mean() < .8:
                        continue
                    occupied[start:stop] = True
                    group.append((kind, item, start, stop))
                    added = True
                    break
                if len(group) == per_group:
                    break
            if not added:
                break
        result.extend(group)
    return result


@dataclass(frozen=True)
class ReviewClip:
    clip: int
    reason: str
    player: str
    start_frame: int
    stop_frame: int
    focus_frame: int
    suggested_start_frame: int | None
    suggested_stop_frame: int | None
    peak_score: str


def _render_reel(source, clips, output, fps):
    capture = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not capture.isOpened():
            raise FileNotFoundError(source)
        width = 960
        height = round(width * capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                       / capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'),
                                 min(fps, 15), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot open review video for writing: {output}')
        for item in clips:
            capture.set(cv2.CAP_PROP_POS_FRAMES, item.start_frame)
            for frame in range(item.start_frame, item.stop_frame):
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError(f'Could not decode source frame {frame}.')
                view = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                cv2.rectangle(view, (8, 9), (425, 46), (22, 22, 22), -1)
                caption = f'Clip {item.clip:02d}  |  {item.player.upper()} PLAYER  |  Frame {frame}'
                cv2.putText(view, caption, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, .58,
                            (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(view)
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _write_reel(source, clips, output, fps):
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        intermediate = Path(temporary)/'review.mp4'
        encoded = Path(temporary)/'encoded.mp4'
        _render_reel(source, clips, intermediate, fps)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(intermediate),
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(encoded)], check=True)
        encoded.replace(output)


def _load_scores(source, checkpoint):
    if checkpoint is not None:
        import torch
        from badminton_analysis.jumps import JumpSegmenter, collect_player_poses, predict_airborne

        store = ObservationStore.open(source)
        calibration = store.load_calibration()
        if calibration is None:
            raise FileNotFoundError(f'No reference court calibration in {source}.')
        sequence = collect_player_poses(store, calibration['H_court_to_image'])
        model = JumpSegmenter()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        return (sequence.frame_ids, predict_airborne(model, sequence),
                Path(store.manifest['video']['path']), float(store.manifest['fps']), .5)

    settings = json.loads((source/'settings.json').read_text())
    with np.load(source/'analysis.npz', allow_pickle=False) as data:
        if 'jump_probabilities' not in data:
            raise ValueError('Use an analysis with jump probabilities or pass a cache and checkpoint.')
        return (data['frame_ids'], data['jump_probabilities'], Path(settings['video']),
                float(settings['fps']), float(settings['jumps']['threshold']))


def build_review_queue(source, output, reviewed, *, per_group=6, seed=7, video=None,
                       checkpoint=None):
    if per_group < 1:
        raise ValueError('At least one clip per group is required.')
    frames, scores, footage, fps, threshold = _load_scores(source, checkpoint)
    if (not len(frames) or scores.shape != (len(frames), 2)
            or not np.all(np.diff(frames) == 1) or not 0 < threshold < 1 or not fps > 0):
        raise ValueError('Expected contiguous source frames, two player scores, and valid settings.')
    excluded = _reviewed_mask(reviewed, frames)
    selected = _pick_candidates(scores, excluded, fps, threshold, per_group, seed)
    clips = [
        ReviewClip(i, kind, PLAYERS[item.player], int(frames[start]), int(frames[stop-1]+1),
                   int(frames[item.focus]), None if kind == 'quiet_audit' else int(frames[item.start]),
                   None if kind == 'quiet_audit' else int(frames[item.stop-1]+1), f'{item.score:.4f}')
        for i, (kind, item, start, stop) in enumerate(selected, 1)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    if video is not None and clips:
        video.parent.mkdir(parents=True, exist_ok=True)
        _write_reel(footage, clips, video, fps)
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ReviewClip.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(clip) for clip in clips)
    return clips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Scored analysis, or the pose-cache directory when passing a checkpoint.')
    parser.add_argument('output_csv', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Score a pose cache with this model.')
    parser.add_argument('--reviewed', action='append', type=Path, default=[])
    parser.add_argument('--per-group', type=int, default=6)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--reel', type=Path)
    args = parser.parse_args()
    if args.checkpoint is not None:
        import torch

        torch.set_num_threads(min(4, torch.get_num_threads()))
    clips = build_review_queue(args.source, args.output_csv, args.reviewed,
                               per_group=args.per_group, seed=args.seed, video=args.reel,
                               checkpoint=args.checkpoint)
    print(f'Saved {len(clips)} candidate clips to {args.output_csv}.')


if __name__ == '__main__':
    main()
