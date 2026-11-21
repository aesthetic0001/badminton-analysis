"""Train the jump segmenter from visually reviewed source-frame intervals."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from badminton_analysis.jumps import (
    HALF_WINDOW, LANDMARKS, WINDOW_SIZE, JumpSegmenter, collect_player_poses,
    jump_segments, pose_windows,
)
from badminton_analysis.observation_store import ObservationStore


def load_labels(path, frame_ids):
    """Read exclusive-stop intervals. Unknown and unreviewed frames are unlabelled."""
    labels = np.full((len(frame_ids), 2), -1, dtype=np.int8)
    split = np.full(labels.shape, '', dtype='<U10')
    hard = np.zeros(labels.shape, dtype=bool)
    assigned = np.zeros(labels.shape, dtype=bool)
    slots = {'near': 0, 'far': 1}
    states = {'uncertain': -1, 'grounded': 0, 'no_correction': 0, 'airborne': 1}
    first, stop = frame_ids[0], frame_ids[-1] + 1
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            player = slots[row['player']]
            start, end = int(row['start_frame']), int(row['stop_frame'])
            state, partition = states[row['state']], row['split']
            if not first <= start < end <= stop or partition not in ('train', 'validation'):
                raise ValueError(f'Bad label interval: {row}')
            index = (slice(start-first, end-first), player)
            if assigned[index].any():
                raise ValueError(f'Overlapping label interval: {row}')
            assigned[index] = True
            labels[index], split[index] = state, partition
            hard[index] = state == 0 and row['kind'] in ('lunge', 'shuffle')
    train = np.flatnonzero((split == 'train').any(axis=1))
    validation = np.flatnonzero((split == 'validation').any(axis=1))
    if not len(train) or not len(validation):
        raise ValueError('Include training and validation clips.')
    if np.intersect1d(train, validation).size:
        raise ValueError('Training and validation labels share source frames.')
    combined = np.sort(np.r_[train, validation])
    same_partition = (split[combined] == 'train').any(axis=1)
    if np.any(np.diff(combined)[np.diff(same_partition)] <= 2 * HALF_WINDOW):
        raise ValueError('Training and validation windows have overlapping context.')
    return labels, split, hard


def _examples(sequence, labels, split, hard, partition):
    frames, players = np.nonzero((labels >= 0) & (split == partition))
    windows = pose_windows(sequence, frames, players)
    available = windows[..., -1].any(axis=1)
    index = (frames[available], players[available])
    targets = labels[index]
    if set(targets) != {0, 1}:
        raise ValueError(f'{partition} needs examples of both contact states.')
    return (torch.from_numpy(windows[available]), torch.from_numpy(targets.astype(np.float32)),
            torch.from_numpy(hard[index]), index)


def _evaluate(model, sequence, validation, labels, split, fps):
    inputs, targets, hard, _ = validation
    model.eval()
    with torch.inference_mode():
        scores = model(inputs).sigmoid().numpy()
    positive = scores >= .5
    expected = targets.numpy() == 1
    correct_airborne = int((positive & expected).sum())
    # Score uncertain boundary frames too, so they don't force artificial splits.
    predicted = np.full(labels.shape, np.nan)
    frames, players = np.nonzero(split == 'validation')
    windows = pose_windows(sequence, frames, players)
    visible = windows[..., -1].any(axis=1)
    with torch.inference_mode():
        predicted[frames[visible], players[visible]] = model(
            torch.from_numpy(windows[visible])).sigmoid().numpy()
    truth = np.where((split == 'validation') & (labels >= 0), labels, np.nan)
    found = list(jump_segments(predicted))
    reference = list(jump_segments(truth))
    pairs = sorted(
        ((min(pe, re)-max(ps, rs), pi, ri)
         for pi, (player, ps, pe) in enumerate(found)
         for ri, (rp, rs, re) in enumerate(reference)
         if player == rp and min(pe, re) > max(ps, rs)),
        reverse=True,
    )
    used, matched, starts, ends = set(), set(), [], []
    for _, pi, ri in pairs:
        if pi not in used and ri not in matched:
            used.add(pi)
            matched.add(ri)
            player, first, stop = reference[ri]
            lower, upper = first, stop
            while lower > 0 and split[lower-1, player] == 'validation' and labels[lower-1, player] == -1:
                lower -= 1
            while upper < len(labels) and split[upper, player] == 'validation' and labels[upper, player] == -1:
                upper += 1
            start_prediction, end_prediction = found[pi][1:]
            starts.append(max(lower - start_prediction, start_prediction - first, 0))
            ends.append(max(stop - end_prediction, end_prediction - upper, 0))
    return {
        'threshold': .5,
        'reviewed_frames_scored': len(scores),
        'airborne_frames': int(expected.sum()),
        'frame_precision': correct_airborne/max(1, int(positive.sum())),
        'frame_recall': correct_airborne/int(expected.sum()),
        'hard_negative_frames': int(hard.sum()),
        'hard_negative_false_positives': int((positive & hard.numpy()).sum()),
        'reference_events': len(reference),
        'predicted_events_in_reviewed_frames': len(found),
        'matched_events': len(matched),
        'mean_takeoff_ms_outside_labelled_range': float(np.mean(starts)*1000/fps) if starts else None,
        'mean_landing_ms_outside_labelled_range': float(np.mean(ends)*1000/fps) if ends else None,
    }


def augment_windows(windows, *, perturb_landmarks=True):
    """Mirror pose, simulate absent context and degrade a few toe/heel measurements."""
    result = windows.clone()
    n, _, _ = result.shape
    joints = len(LANDMARKS)
    xy = result[..., :2*joints].unflatten(-1, (joints, 2))
    velocity = result[..., 2*joints:4*joints].unflatten(-1, (joints, 2))
    confidence = result[..., 4*joints:5*joints]
    root_velocity = result[..., -3:-1]

    mirrored = torch.rand(n) < .5
    order = [1, 0, 3, 2, 5, 4, 7, 6, 11, 12, 13, 8, 9, 10]
    xy[mirrored] = xy[mirrored][:, :, order]
    velocity[mirrored] = velocity[mirrored][:, :, order]
    confidence[mirrored] = confidence[mirrored][:, :, order]
    xy[mirrored, :, :, 0] *= -1
    velocity[mirrored, :, :, 0] *= -1
    root_velocity[mirrored, :, 0] *= -1

    slots = torch.arange(WINDOW_SIZE)
    trims = torch.randint(1, 5, (n, 2)) * (torch.rand(n, 2) < .20)
    missing_frame = ((slots < trims[:, :1]) | (slots >= WINDOW_SIZE-trims[:, 1:])
                     | ((torch.rand(n, WINDOW_SIZE) < .025) & (slots != HALF_WINDOW)))
    result[missing_frame] = 0
    root_pair = (result[:, 1:, -1] > 0) & (result[:, :-1, -1] > 0)
    root_velocity[:, 1:].masked_fill_(~root_pair[..., None], 0)

    if perturb_landmarks:
        foot_confidence = confidence[..., -6:]
        masked = (torch.rand(n, 1, 6) < .10) & (torch.rand(n, WINDOW_SIZE, 1) < .6)
        foot_confidence.masked_fill_(masked, 0)
        visible = foot_confidence > 0
        foot_xy = xy[..., -6:, :]
        jitter = torch.randn_like(foot_xy).clamp(-2, 2) * .006
        jitter *= (torch.rand(n, 1, 6, 1) < .25) & visible[..., None]
        foot_xy.add_(jitter).masked_fill_(~visible[..., None], 0)
        velocity[:, 1:, -6:] += jitter[:, 1:] - jitter[:, :-1]
    paired = (confidence[:, 1:] > 0) & (confidence[:, :-1] > 0)
    velocity[:, 1:].masked_fill_(~paired[..., None], 0)
    return result


def fit_classifier(model, x, y, hard_train, *, epochs, augment=True, perturb_landmarks=True,
                   ranking_weight=0.):
    """Fit a classifier using reviewed pose windows and weighted hard negatives."""
    if epochs < 1:
        raise ValueError('Use at least one training epoch.')
    if not np.isfinite(ranking_weight) or ranking_weight<0:
        raise ValueError('Use a finite, nonnegative ranking weight.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    weights = torch.ones_like(y)
    weights[hard_train] = 2
    weights[y == 1] = weights[y == 0].sum() / (y == 1).sum()
    for epoch in range(epochs):
        sampling = weights.clone()
        if epoch >= epochs//2 and epoch > 0:
            model.eval()
            with torch.inference_mode():
                # Only reviewed grounded examples can be mined as negatives.
                false_positive = (y == 0) & (model(x) >= 0)
            sampling[false_positive] *= 2
        model.train()
        for index in torch.multinomial(sampling, len(y), replacement=True).split(64):
            optimizer.zero_grad()
            batch = augment_windows(x[index], perturb_landmarks=perturb_landmarks) if augment else x[index]
            targets = y[index]
            logits = model(batch)
            loss = criterion(logits,targets)
            if ranking_weight and epoch>=epochs//2:
                negatives = logits[targets==0]
                positives = logits[targets==1]
                if len(negatives) and len(positives):
                    highest = negatives.topk(min(4,len(negatives))).values
                    lowest = positives.topk(min(4,len(positives)),largest=False).values
                    loss = loss+ranking_weight*torch.nn.functional.softplus(
                        highest[:,None]-lowest[None,:]+1
                    ).mean()
            loss.backward()
            optimizer.step()


def train(run, labels_path, output, *, epochs=40, seed=7, augment=True,
          perturb_landmarks=True, initial_weights=None):
    if epochs < 1:
        raise ValueError('Use at least one training epoch.')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    store = ObservationStore.open(run)
    sequence = collect_player_poses(store, store.load_calibration()['H_court_to_image'])
    labels, split, hard = load_labels(labels_path, sequence.frame_ids)
    x, y, hard_train, _ = _examples(sequence, labels, split, hard, 'train')
    validation = _examples(sequence, labels, split, hard, 'validation')
    model = JumpSegmenter()
    if initial_weights is not None:
        model.load_state_dict(initial_weights)
    fit_classifier(model,x,y,hard_train,epochs=epochs,augment=augment,
                   perturb_landmarks=perturb_landmarks)
    report = _evaluate(model, sequence, validation, labels, split, store.manifest['fps'])
    report.update(train_frames=len(y), train_airborne_frames=int(y.sum()),
                  raw_run_id=store.run_id, augmentation=augment,
                  landmark_perturbation=augment and perturb_landmarks)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--no-augmentation', action='store_true', help='Train a comparison baseline.')
    parser.add_argument('--no-landmark-perturbation', action='store_true',
                        help='Keep mirroring and context masking without perturbing individual landmarks.')
    args = parser.parse_args()
    print(json.dumps(train(args.run, args.labels, args.output, epochs=args.epochs,
                           augment=not args.no_augmentation,
                           perturb_landmarks=not args.no_landmark_perturbation), indent=2))


if __name__ == '__main__':
    main()
