"""Compare pose refits at cutoffs calibrated on separate matches."""

import argparse
import json
from pathlib import Path

import torch

from badminton_analysis.jumps import JumpSegmenter
from pose_transfer import add_repeat_cadence_option, parse_repeat_cadences
from refine_jump_edges import _totals, fit_verification, load_match, measure, verified_scores
from verify_jump_events import JumpEnsemble, operating_points, propose


def load_ensemble(directory, evaluation_names):
    training = json.loads((directory / 'report.json').read_text())
    overlap = set(training['train_sources']) & evaluation_names
    if overlap:
        raise ValueError(f'Evaluation matches used for training in {directory}: {sorted(overlap)}')
    models = []
    for row in training['results']:
        checkpoint = directory / f"{row['model']}-{row['seed']}.pt"
        model = JumpSegmenter()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        models.append(model)
    return JumpEnsemble(models), training['train_sources']


def evaluate(matches, scores, thresholds):
    by_cutoff = {}
    for name in ('specificity_threshold', 'recall_threshold'):
        cutoff = thresholds[name]
        if cutoff is None or not matches:
            continue
        rows = {source: measure(match, probabilities, cutoff)
                for (source, match), probabilities in zip(matches, scores, strict=True)}
        by_cutoff[name] = {'matches': rows, 'totals': _totals(rows.values())}
    return by_cutoff


def compare(directory, development, holdouts, *, repeat_cadences=None, blend_refit=None):
    evaluated = {source for _, source in development + holdouts}
    model, training = load_ensemble(directory, evaluated)
    if blend_refit is not None:
        refit_path, weight = blend_refit
        if not 0 < weight < 1:
            raise ValueError('The refit blend weight must be between zero and one.')
        refit, refit_training = load_ensemble(refit_path, evaluated)
        model = JumpEnsemble([model,refit],weights=(1-weight,weight))
        training = sorted(set(training) | set(refit_training))
    cadences = repeat_cadences or {}
    dev = [(source, load_match(Path(cache), source, model, repeat_cadence=cadences.get(source)))
           for cache, source in development]
    dev_matches = [match for _, match in dev]
    verifier, _, out_of_fold = fit_verification(dev_matches)
    thresholds = operating_points([match.verifier_input for match in dev_matches], out_of_fold)

    independent = [(source, load_match(Path(cache), source, model, repeat_cadence=cadences.get(source)))
                   for cache, source in holdouts]
    predicted = [verified_scores(match, propose(match.scores, match.fps, .3), verifier)
                 for _, match in independent]
    report = {
        'training_sources': training,
        'repeat_cadences': cadences,
        'thresholds': thresholds,
        'development': evaluate(dev, out_of_fold, thresholds),
        'holdouts': evaluate(independent, predicted, thresholds),
    }
    if blend_refit is not None:
        report['blend_refit'] = {'directory':str(refit_path),'weight':weight}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ensembles', nargs='+', type=Path)
    parser.add_argument('--dev', action='append', nargs=2, required=True, metavar=('POSE_CACHE', 'SOURCE'))
    parser.add_argument('--holdout', action='append', nargs=2, default=[], metavar=('POSE_CACHE', 'SOURCE'))
    parser.add_argument('--blend-refit', nargs=2, metavar=('CHECKPOINT_DIRECTORY','WEIGHT'),
                        help='Blend an existing refit into each ensemble before calibrating the verifier.')
    add_repeat_cadence_option(parser)
    args = parser.parse_args()
    names = [source for _, source in args.dev + args.holdout]
    if len(args.dev) < 3 or len(set(names)) != len(names):
        parser.error('Use at least three development matches and distinct sources for all evaluations.')
    cadences = parse_repeat_cadences(parser,args.repeat_cadence)
    if cadences.keys() - set(names):
        parser.error('Repeat cadences must match evaluated sources.')
    blend = None
    if args.blend_refit is not None:
        directory, value = args.blend_refit
        try:
            weight = float(value)
        except ValueError:
            parser.error('The refit blend weight must be numeric.')
        if not 0 < weight < 1:
            parser.error('The refit blend weight must be between zero and one.')
        blend = (Path(directory),weight)
    torch.set_num_threads(1)
    result = {str(path): compare(path, args.dev, args.holdout, repeat_cadences=cadences,blend_refit=blend)
              for path in args.ensembles}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
