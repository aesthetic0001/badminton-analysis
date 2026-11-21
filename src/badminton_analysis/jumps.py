"""Per-player jump segmentation and court-space ballistic reconstruction."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from badminton_analysis.positions import (
    court_length_m, court_width_m, wholebody_court_positions,
)


HALF_WINDOW = 6
WINDOW_SIZE = 2 * HALF_WINDOW + 1
LANDMARKS = np.r_[5:7, 11:23]  # Shoulders, hips, knees, ankles, and both feet.
HIP_INDICES = (2, 3)
FEATURES = len(LANDMARKS) * 5 + 3


class JumpSegmenter(nn.Module):
    """Return one airborne logit for slot seven of each 13-frame window."""

    def __init__(self, features=FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW_SIZE * features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, windows):
        return self.network(windows).squeeze(-1)


@dataclass(frozen=True)
class PoseSequence:
    frame_ids: np.ndarray
    court_valid: np.ndarray
    points: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray

    @property
    def present(self):
        return np.isfinite(self.boxes).all(axis=-1)


def collect_player_poses(store, homography, court_min_score=.45, foot_min_score=.4,
                         court_margin_m=1., *, landmarks=LANDMARKS):
    """Associate singles poses in absolute frame order, including across cache chunks."""
    ids, valid, points, scores, boxes = [], [], [], [], []
    previous = np.full((2, 4), np.nan)
    last_seen = np.full(2, -100)
    court_bounds = np.array([court_width_m, court_length_m])
    for chunk in store.iter_chunks():
        frame_ids = chunk['frame_ids']
        court_valid = chunk['court_scores'] >= court_min_score
        _, stances = wholebody_court_positions(
            chunk['keypoints_px'], chunk['keypoint_scores'], homography, foot_min_score,
        )
        candidates = chunk['boxes_xyxy']
        sizes = candidates[:, 2:] - candidates[:, :2]
        usable = np.isfinite(candidates).all(axis=1) & (sizes > 0).all(axis=1)
        on_court = ((stances >= -court_margin_m)
                    & (stances <= court_bounds + court_margin_m)).all(axis=1)
        side = (stances[:, 1] >= court_length_m / 2).astype(int)
        selected = np.full((len(frame_ids), 2), -1, dtype=int)
        offsets = chunk['pose_offsets']
        for i, frame_id in enumerate(frame_ids):
            if not court_valid[i]:
                previous.fill(np.nan)
                continue
            pool = np.flatnonzero(usable[offsets[i]:offsets[i+1]]) + offsets[i]
            if not pool.size:
                continue
            # An established track can keep a pose whose projected feet leave the court.
            centers = (candidates[pool, :2] + candidates[pool, 2:]) / 2
            prior_centers = (previous[:, :2] + previous[:, 2:]) / 2
            heights = previous[:, 3] - previous[:, 1]
            distances = np.linalg.norm(centers[None] - prior_centers[:, None], axis=2)
            distances /= heights[:, None]
            distances[frame_id - last_seen > HALF_WINDOW] = np.inf
            claimed = set()
            for slot in range(2):
                order = np.argsort(distances[slot])
                best = order[0]
                if not distances[slot, best] < .38:
                    continue
                if len(order) > 1 and distances[slot, order[1]] - distances[slot, best] < .12:
                    continue
                if distances[1-slot, best] <= distances[slot, best] + .12:
                    continue
                selected[i, slot] = pool[best]
                claimed.add(pool[best])
            for slot in range(2):
                if selected[i, slot] < 0:
                    seed = [j for j in pool if j not in claimed and on_court[j] and side[j] == slot
                            and not distances[1-slot, np.searchsorted(pool, j)] < .38]
                    if len(seed) == 1:
                        selected[i, slot] = seed[0]
                        claimed.add(seed[0])
                if selected[i, slot] >= 0:
                    previous[slot] = candidates[selected[i, slot]]
                    last_seen[slot] = frame_id
        shape = selected.shape
        chunk_points = np.full((*shape, len(landmarks), 2), np.nan, dtype=np.float32)
        chunk_scores = np.zeros((*shape, len(landmarks)), dtype=np.float32)
        chunk_boxes = np.full((*shape, 4), np.nan, dtype=np.float32)
        matched = selected >= 0
        poses = selected[matched]
        chunk_points[matched] = chunk['keypoints_px'][poses][:, landmarks]
        chunk_scores[matched] = chunk['keypoint_scores'][poses][:, landmarks]
        chunk_boxes[matched] = candidates[poses]
        for output, values in zip((ids, valid, points, scores, boxes),
                                  (frame_ids, court_valid, chunk_points, chunk_scores, chunk_boxes)):
            output.append(values)
    return PoseSequence(*(np.concatenate(parts) for parts in (ids, valid, points, scores, boxes)))


def _resampled_window_inputs(sequence,frames,players,frame_step):
    offsets = frame_step*np.arange(-HALF_WINDOW,HALF_WINDOW+1)
    samples = frames[:,None]+offsets
    start = np.floor(samples).astype(int)
    stop = np.ceil(samples).astype(int)
    last = len(sequence.frame_ids)-1
    lower,upper = start.clip(0,last),stop.clip(0,last)
    mix = samples-start
    selection = players[:,None]
    # Both interpolation endpoints must belong to the same uninterrupted view.
    cuts = np.cumsum(~sequence.court_valid)
    valid = ((start>=0)&(stop<=last)&sequence.court_valid[frames,None]
             &sequence.court_valid[lower]&sequence.court_valid[upper]
             &(cuts[lower]==cuts[frames,None])&(cuts[upper]==cuts[frames,None]))
    points_before = sequence.points[lower,selection]
    points_after = sequence.points[upper,selection]
    points = (1-mix[...,None,None])*points_before+mix[...,None,None]*points_after
    confidence = np.minimum(sequence.scores[lower,selection],sequence.scores[upper,selection])
    boxes = ((1-mix[...,None])*sequence.boxes[lower,selection]
             +mix[...,None]*sequence.boxes[upper,selection])
    return points,confidence,boxes,valid


def pose_windows(sequence, frames, players, *, frame_step=1.):
    """Return zero-padded windows with explicit landmark confidences and frame presence."""
    if not np.isfinite(frame_step) or frame_step<=0:
        raise ValueError('Window frame spacing must be finite and positive.')
    frames, players = np.asarray(frames), np.asarray(players)
    if frame_step == 1.:
        indices = frames[:, None] + np.arange(-HALF_WINDOW, HALF_WINDOW + 1)
        clipped = indices.clip(0, len(sequence.frame_ids) - 1)
        points = sequence.points[clipped, players[:, None]]
        scores = sequence.scores[clipped, players[:, None]]
        boxes = sequence.boxes[clipped, players[:, None]]
        valid = (indices == clipped) & sequence.court_valid[clipped]
        center = valid[:, HALF_WINDOW:HALF_WINDOW+1]
        # Keep windows on the target side of a cut or failed court check.
        left = np.logical_and.accumulate(valid[:, :HALF_WINDOW][:, ::-1], axis=1)[:, ::-1]
        valid[:, :HALF_WINDOW] &= left & center
        valid[:, HALF_WINDOW+1:] &= np.logical_and.accumulate(valid[:, HALF_WINDOW+1:], axis=1) & center
    else:
        points,scores,boxes,valid = _resampled_window_inputs(sequence,frames,players,frame_step)
    present = valid & np.isfinite(boxes).all(axis=-1)
    box_centers = (boxes[..., :2] + boxes[..., 2:]) / 2
    height = boxes[..., 3] - boxes[..., 1]
    typical_height = np.ma.median(np.ma.array(height, mask=~present), axis=1).filled(1)
    scale = typical_height[:, None, None, None]
    hips = points[..., HIP_INDICES, :]
    hip_scores = scores[..., HIP_INDICES]
    hips_valid = np.isfinite(hips).all(axis=(-1, -2)) & (hip_scores >= .3).all(axis=-1)
    root = np.where(hips_valid[..., None], hips.mean(axis=-2), box_centers)
    visible = present[..., None] & np.isfinite(points).all(axis=-1) & np.isfinite(scores) & (scores > 0)
    normalized = np.where(visible[..., None], (points - root[..., None, :]) / scale, 0)
    delta = np.zeros_like(normalized)
    paired = visible[:, 1:] & visible[:, :-1]
    delta[:, 1:] = np.where(paired[..., None], (points[:, 1:] - points[:, :-1]) / scale, 0)
    root_delta = np.zeros_like(root)
    root_delta[:, 1:] = np.where(
        (present[:, 1:] & present[:, :-1])[..., None],
        (root[:, 1:] - root[:, :-1]) / scale[..., 0], 0,
    )
    confidence = np.where(visible, scores.clip(0, 1), 0)
    joint_coordinates = 2 * points.shape[-2]
    return np.concatenate((normalized.reshape(len(frames), WINDOW_SIZE, joint_coordinates),
                           delta.reshape(len(frames), WINDOW_SIZE, joint_coordinates), confidence,
                           root_delta, present[..., None]), axis=-1).astype(np.float32)


def predict_airborne(model, sequence, batch_size=512, *, frame_step=1.):
    """Score each eligible source frame with at least one pose in its centered window."""
    probabilities = np.full((len(sequence.frame_ids), 2), np.nan, dtype=np.float32)
    frames, players = np.nonzero(np.broadcast_to(sequence.court_valid[:, None], probabilities.shape))
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            rows = frames[start:start+batch_size]
            slots = players[start:start+batch_size]
            windows = pose_windows(sequence,rows,slots,frame_step=frame_step)
            has_data = windows[..., -1].any(axis=1)
            scores = model(torch.from_numpy(windows[has_data]).to(device)).sigmoid().cpu().numpy()
            probabilities[rows[has_data], slots[has_data]] = scores
    return probabilities


def jump_segments(probabilities, threshold=.5):
    """Yield (player, start, stop) indices with stop exclusive."""
    for player, scores in enumerate(probabilities.T):
        airborne = np.isfinite(scores) & (scores >= threshold)
        boundaries = np.flatnonzero(np.diff(np.r_[False, airborne, False]))
        yield from ((player, start, stop) for start, stop in boundaries.reshape(-1, 2))


def _project(projection, xyz):
    camera = xyz @ projection[:, :3].T + projection[:, 3]
    depth = camera[:, 2:]
    safe_depth = np.where(np.abs(depth) > 1e-9, depth, np.copysign(1e-9, depth))
    return camera[:, :2] / safe_depth


def projection_from_pole_tops(homography, pole_pixels, height_m=1.55):
    """Recover the vertical column of P using the court plane and two net-pole tops."""
    h = np.asarray(homography, dtype=float)
    pixels = np.asarray(pole_pixels, dtype=float)
    if h.shape != (3, 3) or pixels.shape != (2, 2) or not height_m > 0:
        raise ValueError('Expected a court homography and two pole-top pixels.')
    bases = np.array([[0, court_length_m / 2, 1], [court_width_m, court_length_m / 2, 1]])
    image = bases @ h.T
    system, rhs = [], []
    for (u, v), (x, y, w) in zip(pixels, image):
        system.extend(((height_m, 0, -height_m*u), (0, height_m, -height_m*v)))
        rhs.extend((u*w-x, v*w-y))
    vertical, _, rank, _ = np.linalg.lstsq(system, rhs, rcond=None)
    if rank != 3:
        raise ValueError('Net-pole observations do not determine a vertical projection.')
    return np.column_stack((h[:, :2], vertical, h[:, 2]))


@dataclass(frozen=True)
class TrajectoryFit:
    xyz: np.ndarray
    reprojection_median_px: float


def fit_projectile(seconds, hip_pixels, confidence, projection, anchor_times, anchors,
                   *, max_error_px=8.):
    """Fit metric position and velocity with fixed gravity and soft XY anchors."""
    from scipy.optimize import least_squares

    t = np.asarray(seconds, dtype=float)
    pixels = np.asarray(hip_pixels, dtype=float)
    weights = np.asarray(confidence, dtype=float)
    p = np.asarray(projection, dtype=float)
    anchor_times = np.asarray(anchor_times, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    if p.shape != (3, 4) or not np.isfinite(p).all() or np.linalg.matrix_rank(p) != 3:
        raise ValueError('Expected a finite, rank-three 3x4 camera projection.')
    visible = np.isfinite(pixels).all(axis=1) & np.isfinite(weights) & (weights >= .3)
    if visible.sum() < 3 or not np.isfinite(anchors).all() or anchor_times[0] >= anchor_times[1]:
        return None
    initial_velocity = (anchors[1] - anchors[0]) / np.diff(anchor_times)[0]
    xy_start = anchors[0] - initial_velocity * anchor_times[0]
    initial = np.r_[xy_start, 1., initial_velocity, 4.905*(t[-1]-t[0])]

    def trajectory(parameters, times):
        xyz = parameters[:3] + times[:, None] * parameters[3:]
        xyz[:, 2] -= 4.905 * times**2
        return xyz

    def residual(parameters):
        predicted = _project(p, trajectory(parameters, t[visible]))
        pixel_error = (predicted - pixels[visible]) * np.sqrt(weights[visible, None])
        anchor_error = trajectory(parameters, anchor_times)[:, :2] - anchors
        return np.r_[pixel_error.ravel(), 12 * anchor_error.ravel()]

    result = least_squares(residual, initial, loss='soft_l1', f_scale=3, max_nfev=120)
    xyz = trajectory(result.x, t)
    error = float(np.median(np.linalg.norm(_project(p, xyz[visible]) - pixels[visible], axis=1)))
    if (not result.success or not np.isfinite(xyz).all() or error > max_error_px
            or np.any((xyz[:, 2] < .35) | (xyz[:, 2] > 2.7))
            or np.linalg.norm(result.x[3:5]) > 12
            or np.max(np.linalg.norm(trajectory(result.x, anchor_times)[:, :2] - anchors, axis=1)) > .75):
        return None
    return TrajectoryFit(xyz, error)


def rectify_jumps(sequence, positions, probabilities, projection, fps, threshold=.5):
    """Return corrected XY, fitted XYZ, and a record of each detected jump."""
    xy = positions.copy()
    xyz = np.full((*positions.shape[:2], 3), np.nan)
    events = []
    frames = sequence.frame_ids
    for player, start, stop in jump_segments(probabilities, threshold):
        xy[start:stop, player] = np.nan
        def find_anchor(index, step):
            for _ in range(4):
                if (not 0 <= index < len(frames) or not sequence.court_valid[index]
                        or not probabilities[index, player] < threshold):
                    return None
                if np.isfinite(positions[index, player]).all():
                    return index
                index += step
            return None

        before, after = find_anchor(start-1, -1), find_anchor(stop, 1)
        fit = None
        if before is not None and after is not None:
            hips = sequence.points[start:stop, player][:, HIP_INDICES]
            scores = sequence.scores[start:stop, player][:, HIP_INDICES].min(axis=1)
            seconds = (frames[start:stop] - frames[start]) / fps
            anchors = positions[[before, after], player]
            anchor_times = (frames[[before, after]] - frames[start]) / fps
            fit = fit_projectile(seconds, hips.mean(axis=1), scores, projection, anchor_times, anchors)
            if fit is not None:
                xy[start:stop, player] = fit.xyz[:, :2]
                xyz[start:stop, player] = fit.xyz
        events.append({'player': ('near', 'far')[player], 'takeoff_frame': int(frames[start]),
                       'landing_frame': int(frames[stop]) if stop < len(frames)
                       and np.isfinite(probabilities[stop, player]) else None,
                       'stop_frame_exclusive': int(frames[stop-1]+1),
                       'median_reprojection_px': None if fit is None else fit.reprojection_median_px})
    return xy, xyz, events
