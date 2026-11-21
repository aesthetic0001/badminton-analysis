"""Score pose checkpoints on a separately reviewed match."""

import argparse
import json
from pathlib import Path

import torch

from badminton_analysis.jump_benchmark import evaluate_jump_scores
from badminton_analysis.jump_decode import decode_jump_scores
from badminton_analysis.jumps import JumpSegmenter, predict_airborne
from pose_transfer import load_examples, load_sequence, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose_cache',type=Path)
    parser.add_argument('source')
    parser.add_argument('checkpoints',nargs='+',type=Path)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--decode',nargs=3,type=float,metavar=('PEAK','TAKEOFF','LANDING'),
                        help='Confirm a jump by its peak score and extend its boundaries.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    holdout = load_examples(args.pose_cache,args.source,'validation')
    sequence = load_sequence(args.pose_cache,args.source) if args.decode else None
    model = JumpSegmenter()
    report = {}
    for checkpoint in args.checkpoints:
        weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(weights)
        if args.decode:
            peak,takeoff,landing = args.decode
            probabilities = predict_airborne(model,sequence)
            decisions = decode_jump_scores(probabilities,peak_threshold=peak,
                                           takeoff_threshold=takeoff,landing_threshold=landing)
            result = evaluate_jump_scores(holdout[0],decisions,fps=30)
        else:
            result = score(model,holdout,args.threshold)
        report[str(checkpoint)] = result
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
