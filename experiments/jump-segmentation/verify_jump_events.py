"""Use the score pattern across a proposed jump to refine pose predictions."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch

from badminton_analysis.jump_benchmark import evaluate_jump_scores, jump_operating_points
from badminton_analysis.jumps import JumpSegmenter, jump_segments, predict_airborne
from pose_transfer import load_examples, load_sequence, pose_cache_fps


def load_source(cache,source,model,window_fps=None):
    target = load_examples(cache,source,'validation')[0]
    sequence = load_sequence(cache,source)
    fps = pose_cache_fps(cache,source)
    step = fps/window_fps if window_fps is not None else 1.
    return target,predict_airborne(model,sequence,frame_step=step),fps


def propose(scores,fps,threshold):
    spans = list(jump_segments(scores,threshold))
    scale = 30./fps
    features = []
    for player,start,stop in spans:
        values = scores[start:stop,player]
        logits = logit(values.clip(1e-5,1-1e-5))
        features.append([np.log((stop-start)*scale),logits.max(),logits.mean(),
                         np.count_nonzero(values>=.8)*scale,np.argmax(values)/max(1,stop-start-1)])
    return spans,np.asarray(features,dtype=float).reshape(-1,5)


def fit_verifier(examples,penalty):
    features,labels = [],[]
    for (targets,_,_),(spans,values) in examples:
        # A proposed span is useful if it covers any confirmed airborne frame.
        for (player,start,stop),row in zip(spans,values):
            reference = targets[start:stop,player]
            if (reference>=0).any():
                features.append(row)
                labels.append(int((reference==1).any()))
    X,y = np.asarray(features),np.asarray(labels)
    if len(np.unique(y))!=2:
        raise ValueError('Event verification needs both true and false proposed jumps.')
    center,scale = X.mean(axis=0),X.std(axis=0).clip(1e-5)
    design = np.column_stack((np.ones(len(X)),np.clip((X-center)/scale,-5,5)))

    def objective(weights):
        z = design@weights
        regular = np.r_[0.,weights[1:]]
        value = (np.logaddexp(0,z)-y*z).sum()+.5*penalty*(regular@regular)
        return value,design.T@(expit(z)-y)+penalty*regular

    result = minimize(objective,np.zeros(design.shape[1]),method='L-BFGS-B',jac=True)
    if not result.success:
        raise RuntimeError(result.message)
    return center,scale,result.x


def refine(source,candidates,parameters,weight,extension):
    _,scores,fps = source
    spans,X = candidates
    center,scale,weights = parameters
    adjustments = weights[0]+np.clip((X-center)/scale,-5,5)@weights[1:]
    observed = np.isfinite(scores)
    evidence = np.where(observed,0.,np.nan)
    inside = np.zeros(scores.shape,dtype=bool)
    for (player,start,stop),value in zip(spans,adjustments):
        evidence[start:stop,player] = value
        inside[start:stop,player] = True
    maximum = round(extension*fps/30)
    for (player,start,stop),value in zip(spans,adjustments):
        if value <= 0:
            continue
        for first,step in ((start-1,-1),(stop,1)):
            for distance in range(maximum):
                frame = first+distance*step
                if not 0<=frame<len(scores) or not observed[frame,player] or inside[frame,player]:
                    break
                evidence[frame,player] = max(value,evidence[frame,player])
    return expit(logit(scores.astype(float).clip(1e-6,1-1e-6))+weight*evidence)


def operating_points(sources,probabilities):
    return jump_operating_points(
        np.concatenate([p[labels==1] for (labels,_,_),p in zip(sources,probabilities)]),
        np.concatenate([p[labels==0] for (labels,_,_),p in zip(sources,probabilities)]),
    )


def report(development,validation_scores,holdout,p):
    operating = operating_points(development,validation_scores)
    targets,_,fps = holdout
    evaluation = {}
    for key in ('specificity_threshold','recall_threshold'):
        cutoff = operating[key]
        evaluation[key] = evaluate_jump_scores(targets,p,fps=fps,threshold=cutoff) if cutoff is not None else None
    return {'development':operating,'holdout_at_development_cutoffs':evaluation,
            'holdout_diagnostic':operating_points([holdout],[p])}


def compare(development,holdout,proposal,penalty,weight,extension):
    examples = [(source,propose(source[1],source[2],proposal)) for source in development]
    validation = []
    for i,example in enumerate(examples):
        trained = fit_verifier(examples[:i]+examples[i+1:],penalty)
        validation.append(refine(*example,trained,weight,extension))
    trained = fit_verifier(examples,penalty)
    candidates = propose(holdout[1],holdout[2],proposal)
    evaluation = report(development,validation,holdout,
                        refine(holdout,candidates,trained,weight,extension))
    return {'proposal_threshold':proposal,'penalty':penalty,'weight':weight,'extension_frames_at_30fps':extension,
            **evaluation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--dev',action='append',nargs=2,required=True,metavar=('POSE_CACHE','SOURCE'))
    parser.add_argument('--holdout',nargs=2,required=True,metavar=('POSE_CACHE','SOURCE'))
    parser.add_argument('--window-fps',type=float,
                        help='Interpolate pose inputs at the model training frame rate.')
    args = parser.parse_args()
    names = [name for _,name in args.dev]
    if len(names)<2 or len(set(names))!=len(names) or args.holdout[1] in names:
        parser.error('Supply distinct development matches and a separate holdout match.')
    if args.window_fps is not None and (not np.isfinite(args.window_fps) or args.window_fps<=0):
        parser.error('Use a positive finite model frame rate.')
    torch.set_num_threads(1)
    model = JumpSegmenter()
    model.load_state_dict(torch.load(args.checkpoint,weights_only=True,map_location='cpu'))
    development = [load_source(Path(cache),name,model,args.window_fps) for cache,name in args.dev]
    holdout = load_source(Path(args.holdout[0]),args.holdout[1],model,args.window_fps)
    result = {'development':names,'holdout':args.holdout[1],'holdout_fps':holdout[2],
              'window_fps':args.window_fps,
              'pose_only':report(development,[row[1] for row in development],holdout,holdout[1]),
              'higher_weight':compare(development,holdout,.58,8.,4.,0),
              'lower_weight':compare(development,holdout,.30,2.,.5,1)}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
