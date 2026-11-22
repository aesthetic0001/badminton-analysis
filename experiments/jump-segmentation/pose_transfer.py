"""Compare pose masking checkpoints on the separately reviewed FineBadminton match."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from badminton_analysis.jump_benchmark import evaluate_jump_scores
from badminton_analysis.jumps import HALF_WINDOW, JumpSegmenter, PoseSequence, collect_player_poses, pose_windows
from badminton_analysis.observation_store import ObservationStore
from badminton_analysis.train_jumps import (
    _examples, airborne_boundary_frames, fit_classifier, load_labels, temporal_views,
)


LABELS = Path(__file__).parent/'labels'
STATES = {'grounded':0,'no_correction':0,'airborne':1,'uncertain':-1}


def pose_cache_fps(cache,source):
    with np.load(cache/f'{source}.npz',allow_pickle=False) as data:
        return float(data.get('fps',30.))


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


def load_examples(cache,source,split,*,soft_boundaries=False,frame_steps=()):
    """Read a labeled source and optionally create soft targets or augmented pose windows."""
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
            hard[interval] = state==0 and row['kind'] in ('lunge','shuffle','planted_stroke')
            if soft_boundaries and state==-1 and row['kind'] in ('takeoff','landing'):
                boundaries.append((start-first,stop-first,slot,row['kind']))
    for start,stop,slot,kind in boundaries:
        if start == 0 or stop == len(targets):
            continue
        before,after = (0,1) if kind=='takeoff' else (1,0)
        if targets[start-1,slot]==before and targets[stop,slot]==after:
            targets[start:stop,slot] = np.linspace(before,after,stop-start+2)[1:-1]
    index = np.nonzero(targets != -2)
    windows = pose_windows(sequence,*index)
    keep = windows[..., -1].any(axis=1)
    if frame_steps:
        windows = temporal_views(sequence,index,windows,frame_steps)
    return targets,hard,index,torch.from_numpy(windows),keep


def score(model,holdout,threshold=.5,*,fps=30.):
    labels,_,index,windows,keep = holdout
    probabilities = np.full(labels.shape,np.nan)
    model.eval()
    with torch.inference_mode():
        probabilities[index[0][keep],index[1][keep]] = model(windows[keep]).sigmoid().numpy()
    return evaluate_jump_scores(labels,probabilities,fps=fps,threshold=threshold)


def training_samples(example):
    labels,hard,index,windows,keep = example
    known = keep & (labels[index]>=0)
    selected = (index[0][known],index[1][known])
    return windows[known],torch.tensor(labels[selected],dtype=torch.float32),torch.from_numpy(hard[selected]),selected


def run(cache,pilot,bwf_run,output,*,practice,extra_sources=(),soft_boundaries=False,
        bwf_labels=None,ranking_weight=0.,temporal_augmentation=False,boundary_weight=1.,
        reviewed_sources=(),epochs=40,hard_negative_weight=2.):
    if not np.isfinite(boundary_weight) or boundary_weight<=0:
        raise ValueError('Use a positive finite boundary weight.')
    if epochs < 1:
        raise ValueError('Use at least one training epoch.')
    torch.set_num_threads(1)
    store = ObservationStore.open(bwf_run)
    sequence = collect_player_poses(store,store.load_calibration()['H_court_to_image'])
    steps = (.8,1.25) if temporal_augmentation else ()
    context = int(np.ceil(HALF_WINDOW*max((1.,*steps))))
    labels,split,hard = load_labels(bwf_labels or pilot/'bwf-correction-target-labels.csv',
                                   sequence.frame_ids,train_context=context)
    samples = [_examples(sequence,labels,split,hard,'train',frame_steps=steps)]
    targets = [np.where(split=='train',labels,-2)]
    original_sources = [(cache,name) for name in ('finebadminton-0057','finebadminton-0070')]
    if practice:
        original_sources.extend((cache,f'mendeley-defence-{index}') for index in range(1,5))
    original_sources.extend(extra_sources)
    train_sources = [(path,name,'train') for path,name in original_sources]
    train_sources.extend((path,name,'validation') for path,name in reviewed_sources)
    names = [name for _,name,_ in train_sources]
    if len(set(names))!=len(names):
        raise ValueError('Give each training source only once.')
    if 'finebadminton-0063' in names:
        raise ValueError('FineBadminton 0063 is reserved for the training report.')
    for path,source,label_split in train_sources:
        example = load_examples(path,source,label_split,soft_boundaries=soft_boundaries,frame_steps=steps)
        samples.append(training_samples(example))
        targets.append(example[0])
    train = tuple(torch.cat(tensors) for tensors in zip(*(sample[:3] for sample in samples)))
    positive_weights = None
    if boundary_weight!=1.:
        boundary = np.concatenate([airborne_boundary_frames(target,sample[3])
                                   for target,sample in zip(targets,samples)])
        positive_weights = torch.as_tensor(np.where(boundary,boundary_weight,1.),dtype=torch.float32)
    holdout = load_examples(cache,'finebadminton-0063','validation')
    fps = pose_cache_fps(cache,'finebadminton-0063')
    results = []
    output.mkdir(parents=True)
    for method in ('scratch','mgmae-body','mgmae-relationships'):
        for seed in (7,17,27):
            torch.manual_seed(seed)
            model = JumpSegmenter()
            initial = None
            if method != 'scratch':
                state = torch.load(pilot/method/f'motion-{seed}.pt',weights_only=True,map_location='cpu')
                model.load_state_dict(state)
                initial = score(model,holdout,fps=fps)
            fit_classifier(model,*train,epochs=epochs,ranking_weight=ranking_weight,
                           positive_weights=positive_weights,hard_negative_weight=hard_negative_weight)
            results.append({'model':method,'seed':seed,'initial':initial,
                            'fine_tuned':score(model,holdout,fps=fps)})
            torch.save(model.state_dict(),output/f'{method}-{seed}.pt')
    report = {'train_sources':['bwf-pilot',*names],
              'reviewed_training_sources':[name for _,name in reviewed_sources],
              'positive_training_frames':int((train[1]==1).sum()),
              'negative_training_frames':int((train[1]==0).sum()),
              'hard_negative_training_frames':int(train[2].sum()),
              'hard_negative_weight':hard_negative_weight,
              'soft_boundary_frames':int(((train[1]>0)&(train[1]<1)).sum()),
              'epochs':epochs,
              'ranking_weight':ranking_weight,'temporal_step_factors':[1.,*steps],
              'boundary_weight':boundary_weight,
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
    parser.add_argument('--fit-reviewed-source',action='append',nargs=2,default=[],metavar=('CACHE','SOURCE'),
                        help='Fit using an earlier evaluation CSV. Exclude the source from subsequent evaluation.')
    parser.add_argument('--bwf-labels',type=Path,help='Use a different reviewed BWF label CSV.')
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--hard-negative-weight',type=float,default=2.,
                        help='Sampling weight for reviewed shuffles, lunges, and planted strokes.')
    parser.add_argument('--ranking-weight',type=float,default=0.,
                        help='Penalize high negative logits that outrank low positive logits.')
    parser.add_argument('--temporal-augmentation',action='store_true',
                        help='Sample faster and slower views of each labeled pose window.')
    parser.add_argument('--boundary-weight',type=float,default=1.,
                        help='Weight the first and last two verified airborne frames within positives.')
    parser.add_argument('--soft-boundaries',action='store_true',
                        help='Supervise ambiguous takeoff and landing intervals with soft targets.')
    args = parser.parse_args()
    extra = [(Path(path),source) for path,source in args.extra_source]
    reviewed = [(Path(path),source) for path,source in args.fit_reviewed_source]
    run(args.pose_cache,args.pilot,args.bwf_run,args.output,practice=args.practice,
        extra_sources=extra,soft_boundaries=args.soft_boundaries,bwf_labels=args.bwf_labels,
        ranking_weight=args.ranking_weight,temporal_augmentation=args.temporal_augmentation,
        boundary_weight=args.boundary_weight,reviewed_sources=reviewed,epochs=args.epochs,
        hard_negative_weight=args.hard_negative_weight)


if __name__ == '__main__':
    main()
