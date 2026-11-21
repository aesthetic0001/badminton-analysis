"""Compare pose masking checkpoints on the separately reviewed FineBadminton match."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from badminton_analysis.jump_benchmark import evaluate_jump_scores
from badminton_analysis.jumps import JumpSegmenter, PoseSequence, collect_player_poses, pose_windows
from badminton_analysis.observation_store import ObservationStore
from badminton_analysis.train_jumps import _examples, fit_classifier, load_labels


LABELS = Path(__file__).parent/'labels'
STATES = {'grounded':0,'no_correction':0,'airborne':1,'uncertain':-1}


def load_examples(cache,source,split):
    with np.load(cache/f'{source}.npz',allow_pickle=False) as data:
        frame_ids = data['frames']
        first = frame_ids[0]
        dense = np.arange(first,frame_ids[-1]+1)
        sparse = frame_ids-first
        values = {}
        for key,fill in [('points',np.nan),('boxes',np.nan),('scores',0)]:
            values[key] = np.full((len(dense),*data[key].shape[1:]),fill,dtype=np.float32)
            values[key][sparse] = data[key]
    sequence = PoseSequence(dense,np.isin(dense,frame_ids),**values)
    targets = np.full((len(dense),2),-2,dtype=np.int8)
    hard = np.zeros_like(targets,dtype=bool)
    with (LABELS/f'{source}.csv').open() as handle:
        for row in csv.DictReader(handle):
            start,stop = int(row['start_frame']),int(row['stop_frame'])
            if row['split']!=split or not first<=start<stop<=dense[-1]+1:
                raise ValueError(f'Unexpected interval for {source}: {row}')
            slot = ('near','far').index(row['player'])
            interval = (slice(start-first,stop-first),slot)
            if (targets[interval] != -2).any():
                raise ValueError(f'Overlapping labels for {source}: {row}')
            state = STATES[row['state']]
            targets[interval] = state
            hard[interval] = state==0 and row['kind'] in ('lunge','shuffle')
    index = np.nonzero(targets != -2)
    windows = torch.from_numpy(pose_windows(sequence,*index))
    keep = windows[..., -1].any(dim=1).numpy()
    return targets,hard,index,windows,keep


def score(model,holdout,threshold=.5):
    labels,_,index,windows,keep = holdout
    probabilities = np.full(labels.shape,np.nan)
    model.eval()
    with torch.inference_mode():
        probabilities[index[0][keep],index[1][keep]] = model(windows[keep]).sigmoid().numpy()
    return evaluate_jump_scores(labels,probabilities,fps=30,threshold=threshold)


def training_samples(example):
    labels,hard,index,windows,keep = example
    known = keep & (labels[index]>=0)
    return windows[known],torch.tensor(labels[index][known],dtype=torch.float32),torch.from_numpy(hard[index][known])


def run(cache,pilot,bwf_run,output,*,practice):
    torch.set_num_threads(1)
    store = ObservationStore.open(bwf_run)
    sequence = collect_player_poses(store,store.load_calibration()['H_court_to_image'])
    labels,split,hard = load_labels(pilot/'bwf-correction-target-labels.csv',sequence.frame_ids)
    samples = [_examples(sequence,labels,split,hard,'train')[:3]]
    train_sources = ['finebadminton-0057','finebadminton-0070']
    if practice:
        train_sources.extend(f'mendeley-defence-{index}' for index in range(1,5))
    samples.extend(training_samples(load_examples(cache,source,'train')) for source in train_sources)
    train = tuple(torch.cat(tensors) for tensors in zip(*samples))
    holdout = load_examples(cache,'finebadminton-0063','validation')
    results = []
    output.mkdir(parents=True)
    for method in ('scratch','mgmae-body','mgmae-relationships'):
        for seed in (7,17,27):
            torch.manual_seed(seed)
            model = JumpSegmenter()
            initial = None
            if method != 'scratch':
                weights = torch.load(pilot/method/f'motion-{seed}.pt',weights_only=True,map_location='cpu')
                model.load_state_dict(weights)
                initial = score(model,holdout)
            fit_classifier(model,*train,epochs=40)
            results.append({'model':method,'seed':seed,'initial':initial,'fine_tuned':score(model,holdout)})
            torch.save(model.state_dict(),output/f'{method}-{seed}.pt')
    report = {'train_sources':['bwf-pilot',*train_sources],'positive_training_frames':int(train[1].sum()),
              'negative_training_frames':int((train[1]==0).sum()),'holdout':'finebadminton-0063','results':results}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose_cache',type=Path)
    parser.add_argument('pilot',type=Path)
    parser.add_argument('bwf_run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--practice',action='store_true')
    args = parser.parse_args()
    run(args.pose_cache,args.pilot,args.bwf_run,args.output,practice=args.practice)


if __name__ == '__main__':
    main()
