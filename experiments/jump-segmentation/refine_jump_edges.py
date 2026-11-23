"""Fit takeoff and landing distributions around jumps from a frozen verifier."""

import argparse
from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, logsumexp, softmax
import torch

from badminton_analysis.jump_benchmark import _boundary_range, evaluate_jump_scores
from badminton_analysis.jump_decode import recover_jump_boundaries
from badminton_analysis.jumps import HIP_INDICES, JumpSegmenter, jump_segments, predict_airborne
from pose_transfer import load_examples, load_sequence, pose_cache_fps, remove_repeated_frames
from verify_jump_events import JumpEnsemble, fit_verifier, operating_points, propose, refine


@dataclass
class Match:
    targets: np.ndarray
    scores: np.ndarray
    fps: float
    hips: np.ndarray
    shoes: np.ndarray
    heights: np.ndarray

    @property
    def verifier_input(self):
        return self.targets,self.scores,self.fps


@dataclass
class Edge:
    frames: np.ndarray
    features: np.ndarray
    allowed: np.ndarray


def _landmark_mean(points,confidence):
    usable = (confidence>=.35)&np.isfinite(points).all(axis=-1)
    count = usable.sum(axis=-1)
    total = np.where(usable[...,None],points,0.).sum(axis=-2)
    return np.divide(total,count[...,None],out=np.full(total.shape,np.nan),
                     where=count[...,None]>0)


def load_match(cache,name,model,*,repeat_cadence=None):
    targets = load_examples(cache,name,'validation')[0]
    sequence = load_sequence(cache,name)
    pts,confidence = sequence.points,sequence.scores
    hips = _landmark_mean(pts[:,:,HIP_INDICES],confidence[:,:,HIP_INDICES])
    shoes = np.stack([_landmark_mean(pts[:,:,a:b],confidence[:,:,a:b])
                      for a,b in ((8,11),(11,14))],axis=2)
    heights = sequence.boxes[:,:,3]-sequence.boxes[:,:,1]
    if repeat_cadence is None:
        scores = predict_airborne(model,sequence)
    else:
        compact, source_to_compact = remove_repeated_frames(sequence,*repeat_cadence)
        scores = predict_airborne(model,compact)[source_to_compact]
    return Match(targets,scores,pose_cache_fps(cache,name),hips,shoes,heights)


def verified_scores(match,candidates,parameters):
    scores = refine(match.verifier_input,candidates,parameters,.5,1)
    return recover_jump_boundaries(scores,match.scores,fps=match.fps,before=2,after=1,min_score=.4)


def fit_verification(matches):
    examples = [(match.verifier_input,propose(match.scores,match.fps,.3)) for match in matches]
    out_of_fold = []
    for i,(_,candidates) in enumerate(examples):
        verifier = fit_verifier(examples[:i]+examples[i+1:],2.)
        out_of_fold.append(verified_scores(matches[i],candidates,verifier))
    calibration = operating_points([m.verifier_input for m in matches],out_of_fold)
    cutoff = calibration['specificity_threshold']
    if cutoff is None:
        raise ValueError('The verifier has no development cutoff at the false-positive target.')
    return fit_verifier(examples,2.),cutoff,out_of_fold


def _acceptable_interval(labels,start,stop,landing):
    first,last = _boundary_range(labels,start,stop)
    if landing:
        return (stop,last) if last<len(labels) and labels[last]==0 else None
    return (first,start) if first and labels[first-1]==0 else None


def _motion_features(match,player,frames,height):
    present = (frames>=0)&(frames<len(match.scores))
    frames = frames.clip(0,len(match.scores)-1)
    present &= np.isfinite(match.scores[frames,player])
    h = np.where(present[:,None],match.hips[frames,player],np.nan)
    f = np.where(present[:,None,None],match.shoes[frames,player],np.nan)
    delta = np.diff(h,axis=0)/height
    shoe_speed = np.linalg.norm(np.diff(f,axis=0),axis=2)/height
    foot = np.fmin(shoe_speed[:,0],shoe_speed[:,1])
    return [*delta[:,1],*foot]


def event_edges(match,scores,cutoff):
    """Yield one ordered list of candidate transitions for each accepted event."""
    spans = list(jump_segments(scores,cutoff))
    truth = list(jump_segments(match.targets==1))
    radius = max(1,round(3*match.fps/30))
    for n,(player,start,stop) in enumerate(spans):
        length = len(scores)
        previous = spans[n-1] if n and spans[n-1][0]==player else None
        following = spans[n+1] if n+1<len(spans) and spans[n+1][0]==player else None
        left_limit = (previous[2]+start)//2+1 if previous else 0
        right_limit = (stop+following[1])//2 if following else length
        finite = np.isfinite(match.scores[:,player])
        left = start
        while left>max(left_limit,start-radius) and finite[left-1]:
            left -= 1
        right = stop
        while right<min(right_limit,stop+radius) and finite[right]:
            right += 1
        top = start+int(np.argmax(scores[start:stop,player]))
        found = [(min(e,stop)-max(s,start),s,e) for p,s,e in truth
                 if p==player and min(e,stop)>max(s,start)]
        reference = max(found)[1:] if found else None
        heights = match.heights[start:stop,player]
        heights = heights[np.isfinite(heights)&(heights>0)]
        height = np.median(heights) if len(heights) else np.nan
        edges = []
        for landing,original,lo,hi in ((False,start,left,min(top,start+radius)),
                                      (True,stop,max(top+1,stop-radius),right)):
            frames = np.arange(lo,hi+1)
            feature = []
            for frame in frames:
                indices = frame+np.arange(-2,2)
                if landing:
                    indices = indices[::-1]
                inside = (indices>=0)&(indices<length)
                inputs = match.scores[indices.clip(0,length-1),player]
                logits = logit(np.where(inside,inputs,np.nan).clip(1e-4,1-1e-4))
                offset = (frame-original)*30/match.fps
                motion = _motion_features(match,player,indices,height)
                feature.append([abs(offset),offset,*logits,*motion])
            interval = _acceptable_interval(match.targets[:,player],*reference,landing) if reference else None
            allowed = ((frames>=interval[0])&(frames<=interval[1])) if interval else np.zeros(len(frames),bool)
            edges.append(Edge(frames,np.asarray(feature),allowed))
        yield player,tuple(edges)


def _standardize(X,center,scale):
    finite = np.isfinite(X)
    return np.column_stack((np.where(finite,np.clip((X-center)/scale,-5,5),0.),~finite))


def fit_edges(events,use_motion,penalty=6.):
    models = []
    features = 12 if use_motion else 6
    for side in range(2):
        edges = [edge for _,pair in events if (edge := pair[side]).allowed.any() and len(edge.frames)>1]
        if not edges:
            raise ValueError('No verified examples bracket this type of contact transition.')
        pooled = np.concatenate([e.features[:,:features] for e in edges])
        center = np.nan_to_num(np.nanmean(pooled,axis=0))
        scale = np.nan_to_num(np.nanstd(pooled,axis=0),nan=1.).clip(1e-4)
        inputs = [_standardize(e.features[:,:features],center,scale) for e in edges]

        def objective(w):
            cost,grad = penalty/2*np.dot(w,w),penalty*w
            for X,edge in zip(inputs,edges):
                z = X@w
                a = edge.allowed
                cost += logsumexp(z)-logsumexp(z[a])
                grad += X.T@softmax(z)-X[a].T@softmax(z[a])
            return cost,grad

        result = minimize(objective,np.zeros(2*features),jac=True,method='L-BFGS-B')
        if not result.success:
            raise RuntimeError(result.message)
        models.append((center,scale,result.x))
    return models


def decode(match,events,models,takeoff,landing):
    decisions = np.where(np.isfinite(match.scores),0.,np.nan)
    for player,edges in events:
        bounds = []
        for i,(edge,(center,scale,w)) in enumerate(zip(edges,models)):
            mass = softmax(_standardize(edge.features[:,:len(center)],center,scale)@w)
            level = takeoff if i==0 else 1-landing
            at = min(len(edge.frames)-1,np.searchsorted(np.cumsum(mass),level))
            bounds.append(edge.frames[at])
        decisions[bounds[0]:bounds[1],player] = 1.
    return decisions


def measure(match,scores,cutoff=.5):
    return evaluate_jump_scores(match.targets,scores,fps=match.fps,threshold=cutoff)


def cross_validate(development):
    evidence = []
    for i,match in enumerate(development):
        training = development[:i]+development[i+1:]
        params,cutoff,out_of_fold = fit_verification(training)
        score = verified_scores(match,propose(match.scores,match.fps,.3),params)
        edges = list(event_edges(match,score,cutoff))
        learned = [e for m,s in zip(training,out_of_fold) for e in event_edges(m,s,cutoff)]
        evidence.append((match,score,cutoff,edges,
                         {motion:fit_edges(learned,motion) for motion in (False,True)}))
    baseline = [measure(match,score,cutoff) for match,score,cutoff,_,_ in evidence]
    return evidence,baseline


def _totals(rows):
    return {key:sum(r[key] for r in rows) for key in (
        'positive_frames','negative_frames','true_positive_frames','false_positive_frames',
        'missing_positive_scores','missing_negative_scores','matched_events_with_tolerance','reference_events')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints',nargs='+',type=Path)
    parser.add_argument('--dev',action='append',nargs=2,required=True,metavar=('POSE_CACHE','SOURCE'))
    parser.add_argument('--holdout',nargs=2,metavar=('POSE_CACHE','SOURCE'))
    args = parser.parse_args()
    names = [name for _,name in args.dev]
    if len(names)<3 or len(set(names))!=len(names) or (args.holdout and args.holdout[1] in names):
        parser.error('Supply at least three distinct development matches and a separate holdout.')
    torch.set_num_threads(1)
    models = []
    for checkpoint in args.checkpoints:
        model = JumpSegmenter()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        models.append(model)
    model = models[0] if len(models)==1 else JumpEnsemble(models)
    development = [load_match(Path(cache),name,model) for cache,name in args.dev]
    evidence,baseline = cross_validate(development)
    report = {'development':names,'baseline':_totals(baseline),'variants':[]}
    levels = (.3,.4,.5,.6,.7,.8)
    for motion,takeoff,landing in product((False,True),levels,levels):
        rows = [measure(match,decode(match,edges,params[motion],takeoff,landing))
                for match,_,_,edges,params in evidence]
        report['variants'].append({'motion':motion,'takeoff_quantile':takeoff,
                                   'landing_quantile':landing,'development':_totals(rows)})
    trials = report['variants']

    def rank(row):
        n = row['development']
        meets = n['false_positive_frames'] < .005*(n['negative_frames']-n['missing_negative_scores'])
        return (meets,n['true_positive_frames'] if meets else -n['false_positive_frames'],
                n['true_positive_frames'],-n['false_positive_frames'])

    selected = max(trials,key=rank)
    report['selected'] = selected
    if args.holdout:
        cache,name = args.holdout
        verifier,cutoff,oof = fit_verification(development)
        training = [e for match,p in zip(development,oof) for e in event_edges(match,p,cutoff)]
        model_edges = fit_edges(training,selected['motion'])
        holdout = load_match(Path(cache),name,model)
        p = verified_scores(holdout,propose(holdout.scores,holdout.fps,.3),verifier)
        edges = list(event_edges(holdout,p,cutoff))
        result = decode(holdout,edges,model_edges,selected['takeoff_quantile'],
                        selected['landing_quantile'])
        report['holdout'] = {'source':name,'fps':holdout.fps,'baseline':measure(holdout,p,cutoff),
                             'refined':measure(holdout,result)}
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
