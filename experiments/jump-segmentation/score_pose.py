"""Score pose checkpoints on a separately reviewed match."""

import argparse
import json
from pathlib import Path

import torch

from pose_transfer import JumpSegmenter, load_examples, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose_cache',type=Path)
    parser.add_argument('source')
    parser.add_argument('checkpoints',nargs='+',type=Path)
    parser.add_argument('--threshold',type=float,default=.5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    holdout = load_examples(args.pose_cache,args.source,'validation')
    model = JumpSegmenter()
    report = {}
    for checkpoint in args.checkpoints:
        weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(weights)
        report[str(checkpoint)] = score(model,holdout,args.threshold)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
