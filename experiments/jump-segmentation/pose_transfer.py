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


def load_sequence(cache,source):
    with np.load(cache/f'{source}.npz',allow_pickle=False) as data:
        frame_ids = data['frames']
        first = frame_ids[0]
        dense = np.arange(first,frame_ids[-1]+1)
        sparse = frame_ids-first
        values = {}
        for key,fill in [('points',np.nan),('boxes',np.nan),('scores',0)]:
            values[key] = np.full((len(dense),*data[key].shape[1:]),fill,dtype=np.float32)
            values[key][sparse] = data[key]
    return PoseSequence(dense,np.isin(dense,frame_ids),**values)


def load_examples(cache,source,split,*,soft_boundaries=False):
    if soft_boundaries and split!='train':
        raise ValueError('Soft boundaries are for training labels.')
    sequence = load_sequence(cache,source)
    dense = sequence.frame_ids
    first = dense[0]
    dtype = np.float32 if soft_boundaries else np.int8
    targets = np.full((len(dense),2),-2,dtype=dtype)
    hard = np.zeros_like(targets,dtype=bool)
    boundaries = []
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
            if soft_boundaries and state==-1 and row['kind'] in ('takeoff','landing'):
                boundaries.append((start-first,stop-first,slot,row['kind']))
    for start,stop,slot,kind in boundaries:
        if start == 0 or stop == len(targets):
            continue
        before,after = (0,1) if kind=='takeoff' else (1,0)
        if targets[start-1,slot]==before and targets[stop,slot]==after:
            targets[start:stop,slot] = np.linspace(before,after,stop-start+2)[1:-1]
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


def run(cache,pilot,bwf_run,output,*,practice,extra_sources=(),soft_boundaries=False,bwf_labels=None):
    torch.set_num_threads(1)
    store = ObservationStore.open(bwf_run)
    sequence = collect_player_poses(store,store.load_calibration()['H_court_to_image'])
    labels,split,hard = load_labels(bwf_labels or pilot/'bwf-correction-target-labels.csv',
                                   sequence.frame_ids)
    samples = [_examples(sequence,labels,split,hard,'train')[:3]]
    train_sources = [(cache,name) for name in ('finebadminton-0057','finebadminton-0070')]
    if practice:
        train_sources.extend((cache,f'mendeley-defence-{index}') for index in range(1,5))
    train_sources.extend(extra_sources)
    names = [name for _,name in train_sources]
    if len(set(names))!=len(names):
        raise ValueError('Give each training source only once.')
    samples.extend(training_samples(load_examples(path,source,'train',soft_boundaries=soft_boundaries))
                   for path,source in train_sources)
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
    report = {'train_sources':['bwf-pilot',*names],'positive_training_frames':int((train[1]==1).sum()),
              'negative_training_frames':int((train[1]==0).sum()),
              'soft_boundary_frames':int(((train[1]>0)&(train[1]<1)).sum()),
              'holdout':'finebadminton-0063','results':results}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose_cache',type=Path)
    parser.add_argument('pilot',type=Path)
    parser.add_argument('bwf_run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--practice',action='store_true')
    parser.add_argument('--extra-source',action='append',nargs=2,default=[],metavar=('CACHE','SOURCE'))
    parser.add_argument('--bwf-labels',type=Path,help='Use a different reviewed BWF label CSV.')
    parser.add_argument('--soft-boundaries',action='store_true',
                        help='Supervise ambiguous takeoff and landing intervals with soft targets.')
    args = parser.parse_args()
    extra = [(Path(path),source) for path,source in args.extra_source]
    run(args.pose_cache,args.pilot,args.bwf_run,args.output,practice=args.practice,
        extra_sources=extra,soft_boundaries=args.soft_boundaries,bwf_labels=args.bwf_labels)


if __name__ == '__main__':
    main()
