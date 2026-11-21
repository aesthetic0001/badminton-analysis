"""Compare pose probabilities with and without pixel-level shoe motion."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch

from badminton_analysis.jump_benchmark import evaluate_jump_scores, jump_operating_points
from badminton_analysis.jumps import JumpSegmenter, predict_airborne
from pose_transfer import load_examples, load_sequence


def paired_mean(values,axis):
    usable = np.isfinite(values)
    count = usable.sum(axis=axis)
    total = np.where(usable,values,0).sum(axis=axis)
    return np.divide(total,count,out=np.full(total.shape,np.nan),where=count>0)


def foot_extreme(values,function,fill):
    finite = np.isfinite(values)
    result = function(np.where(finite,values,fill),axis=1)
    return np.where(finite.any(axis=1),result,np.nan)


def optical_features(flow):
    """Represent motion of the least mobile shoe before and after the center."""
    speed = np.log1p(80*np.linalg.norm(flow,axis=-1))
    before = paired_mean(speed[:,:,:2],axis=2)
    after = paired_mean(speed[:,:,2:],axis=2)
    held = np.maximum(before,after)
    vertical = paired_mean(flow[:,:,:,1],axis=2)
    return np.column_stack((
        foot_extreme(before,np.min,np.inf),foot_extreme(after,np.min,np.inf),
        foot_extreme(held,np.min,np.inf),foot_extreme(held,np.max,-np.inf),
        foot_extreme(after-before,np.min,np.inf),
        foot_extreme(after-before,np.max,-np.inf),
        foot_extreme(vertical,np.min,np.inf),foot_extreme(vertical,np.max,-np.inf),
        np.isfinite(speed).mean(axis=(1,2)),
    ))


def source_inputs(cache,source,flow_cache,models):
    sequence = load_sequence(cache,source)
    target,_,index,_,_ = load_examples(cache,source,'validation')
    pose = []
    for model in models:
        probabilities = predict_airborne(model,sequence)
        logits = logit(probabilities.astype(float).clip(1e-5,1-1e-5))
        offsets = np.arange(-2,3)
        rows = index[0][:,None]+offsets
        valid = (rows>=0)&(rows<len(target))
        neighbors = np.where(valid,logits[rows.clip(0,len(target)-1),index[1][:,None]],np.nan)
        pose.extend((neighbors[:,2],paired_mean(neighbors[:,:2],axis=1),
                     paired_mean(neighbors[:,3:],axis=1)))
    pose = np.column_stack(pose)

    with np.load(flow_cache/f'{source}.npz',allow_pickle=False) as data:
        ids = data['frames']
        frames = sequence.frame_ids[index[0]]
        match = np.searchsorted(ids,frames)
        if np.any(match>=len(ids)) or not np.array_equal(ids[match],frames):
            raise ValueError(f'Shoe features and reviewed frames do not match for {source}.')
        pixels = optical_features(data['motion'][match,index[1]])
    available = np.isfinite(pose[:,::3]).any(axis=1)
    return source,target,(index[0][available],index[1][available]),pose[available],pixels[available]


def inputs(data,include_flow):
    _,target,index,pose,flow = data
    return (np.column_stack((pose,flow)) if include_flow else pose),target[index]


def design(data,center,scale):
    finite = np.isfinite(data)
    normalized = np.clip(np.where(finite,(data-center)/scale,0.),-5,5)
    return np.column_stack((np.ones(len(data)),normalized,~finite))


def fit(sources,include_flow,penalty):
    matrices,labels = [],[]
    for source in sources:
        X,y = inputs(source,include_flow)
        known = y>=0
        matrices.append(X[known])
        labels.append(y[known])
    X,y = np.concatenate(matrices),np.concatenate(labels)
    center = np.nanmedian(X,axis=0)
    scale = np.nanstd(X,axis=0)
    center = np.nan_to_num(center)
    scale = np.where(np.isfinite(scale),scale,1.).clip(1e-4)
    matrix = design(X,center,scale)

    def loss(weights):
        logits = matrix@weights
        penalty_weights = np.r_[0.,weights[1:]]
        value = (np.logaddexp(0.,logits)-y*logits).sum()+penalty/2*np.dot(penalty_weights,penalty_weights)
        gradient = matrix.T@(expit(logits)-y)+penalty*penalty_weights
        return value,gradient

    model = minimize(loss,np.zeros(matrix.shape[1]),method='L-BFGS-B',jac=True)
    if not model.success:
        raise RuntimeError(model.message)
    return center,scale,model.x


def probabilities(source,parameters,include_flow):
    center,scale,weights = parameters
    X,_ = inputs(source,include_flow)
    p = np.full(source[1].shape,np.nan)
    p[source[2]] = expit(design(X,center,scale)@weights)
    return p


def operating_points(sources,scores):
    positive = np.concatenate([p[data[1]==1] for data,p in zip(sources,scores)])
    negative = np.concatenate([p[data[1]==0] for data,p in zip(sources,scores)])
    return jump_operating_points(positive,negative)


def compare(development,holdout,include_flow):
    trials = []
    for penalty in (2.,8.,32.):
        validation = []
        for i,example in enumerate(development):
            fitting = development[:i]+development[i+1:]
            validation.append(probabilities(example,fit(fitting,include_flow,penalty),include_flow))
        operating = operating_points(development,validation)
        trials.append((penalty,operating))
    def rank(item):
        _,metrics = item
        true = metrics['true_positive_at_fp_limit']
        false = metrics['false_positive_at_recall_target']
        return (true if true is not None else -1,-false if false is not None else -np.inf)
    penalty,operating = max(trials,key=rank)
    report = {'penalty':penalty,'development_leave_one_match_out':operating}
    if holdout is not None:
        p = probabilities(holdout,fit(development,include_flow,penalty),include_flow)
        labels = holdout[1]
        report['holdout_strict'] = {'fixed_half':evaluate_jump_scores(labels,p,fps=30)}
        for target in ('specificity','recall'):
            key = f'{target}_threshold'
            cutoff = operating[key]
            report['holdout_strict'][f'development_{key}'] = (
                evaluate_jump_scores(labels,p,fps=30,threshold=cutoff) if cutoff is not None else None
            )
        report['holdout_threshold_diagnostic'] = operating_points([holdout],[p])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('flow_cache',type=Path)
    parser.add_argument('checkpoints',nargs='+',type=Path)
    parser.add_argument('--dev',action='append',nargs=2,required=True,metavar=('POSE_CACHE','SOURCE'))
    parser.add_argument('--holdout',nargs=2,metavar=('POSE_CACHE','SOURCE'))
    args = parser.parse_args()
    names = [name for _,name in args.dev]
    if len(names)<2 or len(set(names))!=len(names) or (args.holdout and args.holdout[1] in names):
        parser.error('Supply distinct development sources and a separate holdout source.')
    torch.set_num_threads(1)
    models = [JumpSegmenter() for _ in args.checkpoints]
    for model,checkpoint in zip(models,args.checkpoints):
        weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(weights)
    development = [source_inputs(Path(cache),name,args.flow_cache,models) for cache,name in args.dev]
    holdout = source_inputs(Path(args.holdout[0]),args.holdout[1],args.flow_cache,models) if args.holdout else None
    report = {'development':names,'holdout':holdout[0] if holdout is not None else None}
    for label,include in [('pose_only',False),('pose_and_shoe_flow',True)]:
        report[label] = compare(development,holdout,include)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
