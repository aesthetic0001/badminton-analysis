"""Try motion-guided partial-body masking using cached whole-body poses."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from badminton_analysis.jumps import (
    HALF_WINDOW, LANDMARKS, WINDOW_SIZE, JumpSegmenter, collect_player_poses, pose_windows,
)
from badminton_analysis.observation_store import ObservationStore, file_fingerprint
from badminton_analysis.train_jumps import load_labels, train


JOINTS = len(LANDMARKS)
MASK_FRAMES = 7
LEGS = ((4, 6, 8, 9, 10), (5, 7, 11, 12, 13))  # Knee, ankle, toes, heel.
BODY_LANDMARKS = np.r_[LANDMARKS, 0:5, 7:11]
BODY_REGIONS = {
    'Left leg': LEGS[0],
    'Right leg': LEGS[1],
    'Left arm': (0, 19, 21),
    'Right arm': (1, 20, 22),
    'Shoulders': (0, 1),
    'Head': (14, 15, 16, 17, 18),
}
MIN_TARGET_CONFIDENCE = .3


def _feature_views(windows):
    joints = (windows.shape[-1] - 3) // 5
    position = windows[..., :2*joints].unflatten(-1, (joints, 2))
    motion = windows[..., 2*joints:4*joints].unflatten(-1, (joints, 2))
    confidence = windows[..., 4*joints:5*joints]
    return position, motion, confidence


def mask_leg_track(windows, *, motion_guided):
    """Hide one anatomical leg track over consecutive slots and blank leaked deltas."""
    _, motion, confidence = _feature_views(windows)
    visible = confidence >= MIN_TARGET_CONFIDENCE
    motion_valid = torch.zeros_like(visible)
    motion_valid[:, 1:] = visible[:, 1:] & visible[:, :-1]
    legs = torch.tensor(LEGS, device=windows.device)
    relative_motion = motion - windows[..., None, -3:-1]
    speed = relative_motion.norm(dim=-1).clamp(max=.08) * motion_valid
    joint_coverage = visible[..., legs].unfold(1, MASK_FRAMES, 1).sum(dim=(3, 4))
    weights = torch.ones_like(joint_coverage, dtype=windows.dtype)
    if motion_guided:
        energy = speed[..., legs].unfold(1, MASK_FRAMES, 1).sum(dim=(3, 4))
        average = energy / joint_coverage.clamp(min=1)
        weights += 5 * average / average.mean(dim=(1, 2), keepdim=True).clamp(min=.002)
    weights *= joint_coverage > 0
    # A window with missing legs has no reconstruction targets.
    weights += (weights.sum(dim=(1, 2), keepdim=True) == 0).to(weights.dtype)
    block = torch.multinomial(weights.flatten(1), 1).squeeze(1)

    start, side = block // len(LEGS), block % len(LEGS)
    same_leg = (torch.arange(JOINTS, device=windows.device)[None, :, None]
                == legs[side, None]).any(dim=-1)
    return _apply_track_mask(windows, start, same_leg, visible, motion_valid)


def mask_body_track(windows, *, motion_guided):
    """Pick a region uniformly, then pick an eligible temporal span within it."""
    _, motion, confidence = _feature_views(windows)
    if confidence.shape[-1] != len(BODY_LANDMARKS):
        raise ValueError('Body masking requires the expanded body landmark layout.')
    visible = confidence >= MIN_TARGET_CONFIDENCE
    motion_valid = torch.zeros_like(visible)
    motion_valid[:, 1:] = visible[:, 1:] & visible[:, :-1]
    regions = windows.new_zeros((len(BODY_REGIONS), len(BODY_LANDMARKS)))
    for index, joints in enumerate(BODY_REGIONS.values()):
        regions[index, list(joints)] = 1
    coverage = visible.to(windows.dtype) @ regions.T
    coverage = coverage.unfold(1, MASK_FRAMES, 1).sum(dim=-1)
    eligible = coverage > 0
    region_weights = eligible.any(dim=1).to(windows.dtype)
    region_weights += (region_weights.sum(dim=1, keepdim=True) == 0).to(windows.dtype)
    region = torch.multinomial(region_weights, 1).squeeze(1)
    weights = eligible.to(windows.dtype)
    if motion_guided:
        relative = motion - windows[..., None, -3:-1]
        speed = relative.norm(dim=-1).clamp(max=.08) * motion_valid
        energy = (speed @ regions.T).unfold(1, MASK_FRAMES, 1).sum(dim=-1)
        average = energy / coverage.clamp(min=1)
        weights *= 1 + 5 * average / average.mean(dim=1, keepdim=True).clamp(min=.002)
    rows = torch.arange(len(windows), device=windows.device)
    span_weights = weights[rows, :, region]
    span_weights += (span_weights.sum(dim=1, keepdim=True) == 0).to(windows.dtype)
    start = torch.multinomial(span_weights, 1).squeeze(1)
    return _apply_track_mask(windows, start, regions[region].bool(), visible, motion_valid)


def _apply_track_mask(windows, start, region, visible, motion_valid):
    slots = torch.arange(WINDOW_SIZE, device=windows.device)
    same_span = (slots >= start[:, None]) & (slots < start[:, None] + MASK_FRAMES)
    hidden = same_span[..., None] & region[:, None] & (windows[..., -1:] > 0)
    hidden_motion = hidden.clone()
    hidden_motion[:, 1:] |= hidden[:, :-1]

    masked = windows.clone()
    masked_position, masked_motion, mask_tokens = _feature_views(masked)
    masked_position.masked_fill_(hidden[..., None], 0)
    masked_motion.masked_fill_(hidden_motion[..., None], 0)
    mask_tokens.masked_fill_(hidden, -1)
    return masked, hidden & visible, hidden_motion & motion_valid


def _pretraining_inputs(sequence, split, fps):
    # Exclude the area around every validation clip for both players, including
    # context on either side of the sampled centre.
    blocked = np.zeros(len(sequence.frame_ids), dtype=bool)
    pad = round(30 * fps) + HALF_WINDOW
    for index in np.flatnonzero((split == 'validation').any(axis=1)):
        blocked[max(0, index-pad):index+pad+1] = True
    frames, players = np.nonzero(sequence.present & sequence.court_valid[:, None] & ~blocked[:, None])
    sampled = frames % 3 == 0
    windows = pose_windows(sequence, frames[sampled], players[sampled])
    usable = windows[..., -1].sum(axis=1) >= 9
    if not usable.any():
        raise ValueError('No training pose windows remain after excluding validation clips.')
    return torch.from_numpy(windows[usable])


def _pretrain(model, windows, mode, epochs, mask_track=mask_leg_track):
    encoder = model.network[:-1]
    joints = (windows.shape[-1] - 3) // 5
    decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU(),
                            nn.Linear(128, WINDOW_SIZE * joints * 4))
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=.001)
    history = []
    for _ in range(epochs):
        total = 0.
        for index in torch.randperm(len(windows)).split(128):
            target = windows[index]
            masked, reconstruct_position, reconstruct_motion = mask_track(
                target, motion_guided=(mode == 'motion'),
            )
            position, motion, _ = _feature_views(target)
            prediction = decoder(encoder(masked)).reshape(-1, WINDOW_SIZE, joints, 4)
            # Score only hidden, confident coordinates and motion samples.
            position_error = F.smooth_l1_loss(prediction[..., :2], position, reduction='none')
            relative_motion = motion - target[..., None, -3:-1]
            motion_error = F.smooth_l1_loss(prediction[..., 2:], 4 * relative_motion, reduction='none')
            position_loss = (position_error * reconstruct_position[..., None]).sum() / (
                2 * reconstruct_position.sum()).clamp(min=1)
            motion_loss = (motion_error * reconstruct_motion[..., None]).sum() / (
                2 * reconstruct_motion.sum()).clamp(min=1)
            loss = position_loss + motion_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        history.append(total / len(windows))
    return history


def _classifier_weights(model):
    """Transfer the pretrained weights for the classifier's existing pose inputs."""
    state = model.state_dict()
    weights = state['network.1.weight']
    joints = len(BODY_LANDMARKS)
    columns = np.r_[0:2*JOINTS, 2*joints:2*joints+2*JOINTS, 4*joints:4*joints+JOINTS,
                    5*joints:5*joints+3]
    state['network.1.weight'] = weights.unflatten(1, (WINDOW_SIZE, 5*joints+3))[:, :, columns].flatten(1)
    return state


def run_experiment(run, labels_path, output, *, masking='motion', seed=7,
                   pretrain_epochs=12, epochs=40, regions='legs'):
    output = Path(output)
    if output.suffix != '.pt' or output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Use a new .pt output path for this experiment.')
    if masking not in ('motion', 'uniform', 'none') or pretrain_epochs < 1:
        raise ValueError('Choose motion, uniform or none, and at least one pretraining epoch.')
    if regions not in ('legs', 'body'):
        raise ValueError('Choose legs or body regions.')
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    initial_weights = None
    pretraining = {'masking': masking, 'epochs': 0}
    if masking != 'none':
        store = ObservationStore.open(run)
        landmarks = BODY_LANDMARKS if regions == 'body' else LANDMARKS
        sequence = collect_player_poses(store, store.load_calibration()['H_court_to_image'],
                                       landmarks=landmarks)
        _, split, _ = load_labels(labels_path, sequence.frame_ids)
        windows = _pretraining_inputs(sequence, split, store.manifest['fps'])
        model = JumpSegmenter(windows.shape[-1])
        mask_track = mask_body_track if regions == 'body' else mask_leg_track
        history = _pretrain(model, windows, masking, pretrain_epochs, mask_track)
        initial_weights = _classifier_weights(model) if regions == 'body' else model.state_dict()
        pretraining = {'masking': masking, 'epochs': pretrain_epochs, 'windows': len(windows),
                       'leg_frames': MASK_FRAMES, 'validation_padding_seconds': 30,
                       'first_epoch_loss': history[0], 'last_epoch_loss': history[-1]}
        if regions == 'body':
            pretraining.update(regions=list(BODY_REGIONS), pretraining_landmarks=landmarks.tolist())
    report = train(run, labels_path, output, seed=seed, epochs=epochs,
                   initial_weights=initial_weights)
    report.update(seed=seed, supervised_epochs=epochs, masked_pretraining=pretraining,
                  labels=file_fingerprint(labels_path, content=True))
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--masking', choices=('motion', 'uniform', 'none'), default='motion')
    parser.add_argument('--regions', choices=('legs', 'body'), default='legs')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--pretrain-epochs', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=40)
    args = parser.parse_args()
    report = run_experiment(args.run, args.labels, args.output, masking=args.masking,
                            seed=args.seed, pretrain_epochs=args.pretrain_epochs, epochs=args.epochs,
                            regions=args.regions)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
