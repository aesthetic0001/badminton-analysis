"""Score pose checkpoints on a separately reviewed match."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from badminton_analysis.jump_benchmark import evaluate_jump_scores, jump_operating_points
from badminton_analysis.jump_decode import decode_jump_scores
from badminton_analysis.jumps import JumpSegmenter, predict_airborne
from pose_transfer import load_examples, load_sequence, pose_cache_fps, score


def score_at_development_cutoffs(model,development,holdout,sequence,threshold,fps):
    positive,negative = [],[]
    for targets,source_sequence in development:
        estimated = predict_airborne(model,source_sequence)
        positive.append(estimated[targets==1])
        negative.append(estimated[targets==0])
    operating = jump_operating_points(np.concatenate(positive),np.concatenate(negative))
    estimated = predict_airborne(model,sequence)
    scores = {'development_operating_points':operating,
              'holdout_at_fixed_threshold':evaluate_jump_scores(holdout,estimated,fps=fps,
                                                               threshold=threshold),
              'holdout_threshold_diagnostic':jump_operating_points(estimated[holdout==1],
                                                                   estimated[holdout==0])}
    for kind in ('specificity','recall'):
        cutoff = operating[f'{kind}_threshold']
        scores[f'holdout_at_development_{kind}_threshold'] = (
            evaluate_jump_scores(holdout,estimated,fps=fps,threshold=cutoff) if cutoff is not None else None
        )
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose_cache',type=Path)
    parser.add_argument('source')
    parser.add_argument('checkpoints',nargs='+',type=Path)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--decode',nargs=3,type=float,metavar=('PEAK','TAKEOFF','LANDING'),
                        help='Confirm a jump by its peak score and extend its boundaries.')
    parser.add_argument('--development',action='append',nargs=2,metavar=('POSE_CACHE','SOURCE'),
                        help='Choose raw-score cutoffs using separate development footage.')
    args = parser.parse_args()
    names = [name for _,name in args.development] if args.development else []
    if args.source in names or len(names)!=len(set(names)):
        parser.error('Supply distinct development sources and holdout source.')
    if names and args.decode:
        parser.error('Development cutoff calibration requires raw pose scores.')
    torch.set_num_threads(1)
    holdout = load_examples(args.pose_cache,args.source,'validation')
    fps = pose_cache_fps(args.pose_cache,args.source)
    sequence = load_sequence(args.pose_cache,args.source) if args.decode or names else None
    development = []
    for cache,name in args.development or []:
        cache = Path(cache)
        development.append((load_examples(cache,name,'validation')[0],load_sequence(cache,name)))
    model = JumpSegmenter()
    report = {}
    for checkpoint in args.checkpoints:
        weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(weights)
        if args.decode:
            peak,takeoff,landing = args.decode
            probabilities = predict_airborne(model,sequence)
            decisions = decode_jump_scores(probabilities,peak_threshold=peak,
                                           takeoff_threshold=takeoff,landing_threshold=landing,fps=fps)
            result = evaluate_jump_scores(holdout[0],decisions,fps=fps)
        elif development:
            result = score_at_development_cutoffs(model,development,holdout[0],sequence,args.threshold,fps)
        else:
            result = score(model,holdout,args.threshold,fps=fps)
        report[str(checkpoint)] = result
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
