"""Train a visual shoe-contact model and compare its evidence with verified pose scores."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
import torch
from torch import nn
from torch.nn import functional as F

from compare_pose_refit import evaluate, load_ensemble
from pose_transfer import load_examples
from prepare_shoe_rgb import OFFSETS, PATCH_SIZE
from refine_jump_edges import fit_verification, load_match, verified_scores
from verify_jump_events import operating_points, propose


class ShoeContact(nn.Module):
    """Score airborne state from paired, fixed-coordinate shoe clips."""

    def __init__(self):
        super().__init__()
        width, height = PATCH_SIZE
        self.shoe = nn.Sequential(
            nn.Conv2d(3 * len(OFFSETS), 12, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(24 * (width // 8) * (height // 8), 24), nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Dropout(.15), nn.Linear(24, 1))

    def forward(self, video):
        count, _, frames, height, width, channels = video.shape
        stacked = video.permute(0, 1, 2, 5, 3, 4).reshape(2 * count, frames * channels, height, width)
        features = self.shoe(stacked).reshape(count, 2, -1)
        joint = torch.cat((features.amin(dim=1), features.amax(dim=1)), dim=1)
        return self.classifier(joint).squeeze(-1)


def load_patches(visual_cache, pose_cache, name):
    with np.load(Path(pose_cache) / f'{name}.npz', allow_pickle=False) as poses:
        frames = poses['frames']
    with np.load(visual_cache / f'{name}.npz', allow_pickle=False) as pixels:
        if not np.array_equal(frames, pixels['frames']):
            raise ValueError(f'The visual and pose caches cover different frames for {name}.')
        clips, visible = pixels['patches'], pixels['visible']
    return frames - frames[0], clips, visible.any(axis=-1)


def training_samples(visual_cache, sources):
    batches = []
    for cache, name, label_split in sources:
        targets, hard, _, _, _ = load_examples(Path(cache), name, label_split, soft_boundaries=True)
        frames, clips, present = load_patches(visual_cache, cache, name)
        labels = targets[frames]
        usable = present & (labels >= 0)
        batches.append((clips[usable], labels[usable], hard[frames][usable]))
    return tuple(torch.from_numpy(np.concatenate(samples)) for samples in zip(*batches))


def augment(clips):
    present = clips.any(dim=(-3, -2, -1), keepdim=True)
    # Share exposure and color shifts across the five frames and both shoes.
    intensity = .65 + .7 * torch.rand((len(clips), 1, 1, 1, 1, 1))
    color = .2 * torch.rand((len(clips), 1, 1, 1, 1, 3)) - .1
    clips = (clips * intensity + color).clamp(0, 1) * present
    if torch.rand(()) < .5:
        clips = clips.flip((-2,))
    hide = torch.rand(len(clips)) < .08
    clips[hide, torch.randint(2, (int(hide.sum()),))] = 0
    return clips


def fit(clips, targets, hard, epochs, seed):
    torch.manual_seed(seed)
    model = ShoeContact()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    weights = torch.where(hard, 2., 1.)
    weights[targets == 1] = weights[targets == 0].sum() / (targets == 1).sum()
    for epoch in range(epochs):
        model.train()
        for index in torch.multinomial(weights, len(targets), replacement=True).split(64):
            logits = model(augment(clips[index].float() / 255.))
            labels = targets[index]
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            positive, negative = logits[labels == 1], logits[labels == 0]
            if epoch >= epochs // 2 and len(positive) and len(negative):
                low = positive.topk(min(4, len(positive)), largest=False).values
                high = negative.topk(min(4, len(negative))).values
                loss = loss + .3 * F.softplus(high[:, None] - low[None] + 1.).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def train(visual_cache, sources, destination, epochs):
    if epochs < 1:
        raise ValueError('Use at least one training epoch.')
    clips, targets, hard = training_samples(visual_cache, sources)
    positive = int((targets == 1).sum())
    negative = int((targets == 0).sum())
    if not positive or not negative:
        raise ValueError('Training needs both airborne and negative examples.')
    names = [name for _, name, _ in sources]
    if len(set(names)) != len(names):
        raise ValueError('Supply each training source once.')
    destination.mkdir(parents=True)
    seeds = (7, 17, 27)
    for seed in seeds:
        model = fit(clips, targets, hard, epochs, seed)
        torch.save(model.state_dict(), destination / f'shoe-rgb-{seed}.pt')
    report = {
        'training': [(name, split) for _, name, split in sources], 'seeds': seeds,
        'positive_frames': positive, 'negative_frames': negative,
        'soft_frames': len(targets) - positive - negative, 'epochs': epochs,
    }
    (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def visual_logits(models, visual_cache, cache, source, shape):
    frames, clips, present = load_patches(visual_cache, cache, source)
    position, player = np.nonzero(present)
    inputs = torch.from_numpy(clips[position, player])
    output = np.full(shape, np.nan, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(inputs), 128):
            batch = inputs[start:start + 128].float() / 255.
            logits = torch.stack([model(batch) for model in models]).mean(dim=0)
            batch_slice = slice(start, start + len(batch))
            output[frames[position[batch_slice]], player[batch_slice]] = logits.numpy()
    return output


def fuse(probabilities, rgb, weight):
    fused = expit(logit(probabilities.clip(1e-6, 1. - 1e-6)) + weight * rgb.clip(-5., 5.))
    return np.where(np.isfinite(rgb), fused, probabilities)


def compare(visual_cache, visual_models, pose_models, development, holdouts):
    all_names = {name for _, name in development + holdouts}
    training = json.loads((visual_models / 'report.json').read_text())
    visual_names = {name for name, _ in training['training']}
    if visual_names & all_names:
        raise ValueError(f'Visual training overlaps evaluation: {sorted(visual_names & all_names)}')
    pose_model, _ = load_ensemble(pose_models, all_names)
    rgb_models = []
    for seed in training['seeds']:
        model = ShoeContact()
        model.load_state_dict(torch.load(visual_models / f'shoe-rgb-{seed}.pt',
                                         map_location='cpu', weights_only=True))
        rgb_models.append(model.eval())

    def load(sources):
        matches, visual = [], []
        for cache, name in sources:
            match = load_match(Path(cache), name, pose_model)
            matches.append((name, match))
            visual.append(visual_logits(rgb_models, visual_cache, cache, name, match.targets.shape))
        return matches, visual

    development, dev_rgb = load(development)
    dev_matches = [match for _, match in development]
    verifier, _, dev_pose = fit_verification(dev_matches)
    variants = {}
    for weight in (0., .25, .5, 1., 2.):
        scores = [fuse(p, v, weight) if weight else p for p, v in zip(dev_pose, dev_rgb, strict=True)]
        thresholds = operating_points([m.verifier_input for m in dev_matches], scores)
        variants[weight] = {'thresholds': thresholds, 'development': evaluate(development, scores, thresholds)}

    best_rgb = max((weight for weight in variants if weight),
                   key=lambda w: (variants[w]['thresholds']['true_positive_at_fp_limit'], -w))
    independent, test_rgb = load(holdouts)
    test_pose = [verified_scores(m, propose(m.scores, m.fps, .3), verifier) for _, m in independent]
    for weight in (0., best_rgb):
        scores = [fuse(p, v, weight) if weight else p for p, v in zip(test_pose, test_rgb, strict=True)]
        variants[weight]['holdouts'] = evaluate(independent, scores, variants[weight]['thresholds'])
        if independent:
            variants[weight]['holdout_diagnostic'] = operating_points(
                [match.verifier_input for _, match in independent], scores)
    return {'selected_rgb_weight': best_rgb, 'variants': variants}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('visual_cache', type=Path)
    training.add_argument('output', type=Path)
    training.add_argument('--source', action='append', nargs=3, required=True,
                          metavar=('POSE_CACHE', 'SOURCE', 'LABEL_SPLIT'))
    training.add_argument('--epochs', type=int, default=32)
    scoring = sub.add_parser('compare')
    scoring.add_argument('visual_cache', type=Path)
    scoring.add_argument('visual_models', type=Path)
    scoring.add_argument('pose_models', type=Path)
    scoring.add_argument('--dev', action='append', nargs=2, required=True, metavar=('POSE_CACHE', 'SOURCE'))
    scoring.add_argument('--holdout', action='append', nargs=2, default=[], metavar=('POSE_CACHE', 'SOURCE'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'train':
        train(args.visual_cache, args.source, args.output, args.epochs)
    else:
        names = [source for _, source in args.dev + args.holdout]
        if len(args.dev) < 3 or len(set(names)) != len(names):
            parser.error('Supply at least three development matches and distinct sources for all evaluations.')
        result = compare(args.visual_cache, args.visual_models, args.pose_models, args.dev, args.holdout)
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
