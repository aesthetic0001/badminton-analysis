"""Compare score-only and camera-calibrated jump event verification."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from badminton_analysis.jumps import JumpSegmenter, _project, projection_from_pole_tops
from badminton_analysis.positions import court_length_m, court_width_m, detect_court
from refine_jump_edges import load_match, verified_scores
from verify_jump_events import JumpEnsemble, fit_verifier, propose, report


@dataclass(frozen=True)
class Camera:
    image_to_ground: np.ndarray
    projection: np.ndarray


def calibrate(video, reference, pole_tops):
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 3])
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, reference)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f'Cannot read calibration frame {reference} in {video}.')
    height, width = frame.shape[:2]
    reduced_size = (960, round(height * 960 / width))
    reduced = cv2.resize(frame, reduced_size, interpolation=cv2.INTER_AREA)
    ground = detect_court(reduced)['H_court_to_image']
    bases = np.array([[0., court_length_m / 2, 0.], [court_width_m, court_length_m / 2, 0.]])
    plane = np.column_stack((ground[:, :2], np.zeros(3), ground[:, 2]))
    # The recorded pole tops are in image order, which can differ from court X order.
    ordered = np.asarray(pole_tops)[np.argsort(_project(plane, bases)[:, 0])]
    scale = np.diag([width / reduced_size[0], height / reduced_size[1], 1.])
    homography = scale @ ground
    projection = scale @ projection_from_pole_tops(ground, ordered)
    return Camera(np.linalg.inv(homography), projection)


def project_to_court(inverse, pixels):
    points = pixels @ inverse[:, :2].T + inverse[:, 2]
    depth = points[..., 2:]
    return np.divide(points[..., :2], depth, out=np.full(points[..., :2].shape, np.nan),
                     where=np.abs(depth) > 1e-8)


def fit_arc(times, pixels, anchors, projection, height, rise):
    """Fit two hip endpoints with optional free vertical curvature."""
    linear = np.column_stack((1. - times, times))
    arc = 4. * times * (1. - times)
    initial = np.column_stack((linear @ anchors, np.full(len(times), .95)))
    depth = initial @ projection[2, :3] + projection[2, 3]
    equations = (projection[:2] - pixels[..., None] * projection[2]) / depth[:, None, None]
    xy_regularization = .18 * height
    z_regularization = .04 * height
    weights = np.tile([xy_regularization, xy_regularization, z_regularization], 2)
    prior = np.column_stack((anchors, [.95, .95])).ravel()
    design = (equations[:, :, None, :3] * linear[:, None, :, None]).reshape(-1, 6)
    curvature = (equations[:, :, 2] * arc[:, None]).ravel()
    target = -equations[:, :, 3].ravel()
    if rise is None:
        design = np.column_stack((design, curvature))
        weights = np.r_[weights, z_regularization]
        prior = np.r_[prior, 0.]
    else:
        target -= rise * curvature
    parameters = np.linalg.lstsq(np.vstack((design, np.diag(weights))),
                                 np.r_[target, weights * prior], rcond=None)[0]
    rise = parameters[6] if rise is None else rise
    endpoints = parameters[:6].reshape(2, 3)
    points = linear @ endpoints
    points[:, 2] += rise * arc
    errors = _project(projection, points) - pixels
    return np.sqrt(np.mean(np.sum(errors**2, axis=1))) / height, rise


def _anchor(shoes, observed, first, step, count):
    for index in range(first, first + count * step, step):
        if not 0 <= index < len(shoes) or not observed[index]:
            break
        valid = shoes[index][np.isfinite(shoes[index]).all(axis=1)]
        if len(valid):
            return index, valid.mean(axis=0)
    return None


def motion_features(match, camera, player, start, stop):
    """Measure arc fit and foot support using contiguous pose observations."""
    observed = np.isfinite(match.scores[:, player])
    shoes = match.shoes[:, player]
    buffer = max(1, round(3 * match.fps / 30))
    before = _anchor(shoes, observed, start - 1, -1, buffer)
    after = _anchor(shoes, observed, stop, 1, buffer)
    missing = np.zeros(8)
    if before is None or after is None:
        return missing
    left, right = before[0], after[0]
    feet = project_to_court(camera.image_to_ground, np.stack((before[1], after[1])))
    span = slice(left, right + 1)
    pixels = match.hips[span, player]
    heights = match.heights[span, player]
    valid_heights = heights[np.isfinite(heights) & (heights > 0)]
    visible = np.isfinite(pixels).all(axis=1)
    projection = camera.projection
    anchor_hips = np.column_stack((feet, [.95, .95]))
    depth = anchor_hips @ projection[2, :3] + projection[2, 3]
    if (visible.sum() < 3 or not len(valid_heights) or
            not np.isfinite(depth).all() or abs(depth).min() < 1e-8 or depth[0] * depth[1] <= 0):
        return missing
    times = np.linspace(0., 1., right - left + 1)[visible]
    args = (times, pixels[visible], feet, projection, np.median(valid_heights))
    rise = 9.81 * ((right - left) / match.fps)**2 / 8.
    ground_error, _ = fit_arc(*args, rise=0.)
    flight_error, _ = fit_arc(*args, rise=rise)
    free_error, free_rise = fit_arc(*args, rise=None)
    shoe_xy = project_to_court(camera.image_to_ground, shoes[span])
    movement = np.linalg.norm(np.diff(shoe_xy, axis=0), axis=-1) * match.fps
    slowest = np.min(np.where(np.isfinite(movement), movement, np.inf), axis=1)
    supported = slowest[np.isfinite(slowest)]
    features = np.array([1., ground_error, flight_error, ground_error - flight_error,
                         free_error, free_rise, float(len(supported) > 0),
                         (supported < .75).mean() if len(supported) else 0.])
    return features if np.isfinite(features).all() else missing


def candidates(match, camera):
    spans, features = propose(match.scores, match.fps, .3)
    motion = np.array([motion_features(match, camera, *span) for span in spans]).reshape(-1, 8)
    return (spans, features), (spans, np.column_stack((features, motion)))


def compare(development, holdout, variant):
    examples = [(match.verifier_input, proposals[variant]) for match, proposals in development]
    predictions = []
    for index, (match, proposals) in enumerate(development):
        verifier = fit_verifier(examples[:index] + examples[index + 1:], 2.)
        predictions.append(verified_scores(match, proposals[variant], verifier))
    verifier = fit_verifier(examples, 2.)
    match, proposals = holdout
    scores = verified_scores(match, proposals[variant], verifier)
    return report([example[0] for example in examples], predictions, match.verifier_input, scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--dev', action='append', nargs=3, required=True,
                        metavar=('POSE_CACHE', 'SOURCE', 'VIDEO'))
    parser.add_argument('--holdout', nargs=3, required=True, metavar=('POSE_CACHE', 'SOURCE', 'VIDEO'))
    args = parser.parse_args()
    names = [name for _, name, _ in args.dev]
    if len(names) < 3 or len(set(names)) != len(names) or args.holdout[1] in names:
        parser.error('Supply at least three distinct development matches and a separate holdout.')

    torch.set_num_threads(1)
    cv2.setNumThreads(2)
    models = []
    for checkpoint in args.checkpoints:
        model = JumpSegmenter()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        models.append(model)
    model = models[0] if len(models) == 1 else JumpEnsemble(models)
    reference = Path(__file__).with_name('court-reference-frames.json')
    references = json.loads(reference.read_text())
    poles = json.loads(reference.with_name('jump-projection-poles.json').read_text())

    def prepare(cache, name, video):
        match = load_match(Path(cache), name, model)
        camera = calibrate(video, references[name], poles[name])
        return match, candidates(match, camera)

    development = [prepare(*source) for source in args.dev]
    holdout = prepare(*args.holdout)
    result = {'development': names, 'holdout': args.holdout[1], 'holdout_fps': holdout[0].fps,
              'scores': compare(development, holdout, 0), 'geometry': compare(development, holdout, 1)}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
