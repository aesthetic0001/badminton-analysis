"""Extend confirmed jump segments to recover takeoff and landing frames."""

import numpy as np

from badminton_analysis.jumps import jump_segments


def decode_jump_scores(probabilities, *, peak_threshold, takeoff_threshold, landing_threshold, fps=30.):
    """Return 0/1 decisions with NaNs retained where pose scores are unavailable."""
    scores = np.asarray(probabilities,dtype=float)
    if (scores.ndim != 2 or not np.isfinite(fps) or fps <= 0
            or not .5 <= peak_threshold <= 1
            or not 0 <= takeoff_threshold <= .5 or not 0 <= landing_threshold <= .5):
        raise ValueError('Expected frame-by-player scores, valid thresholds, and a positive frame rate.')
    if np.any((scores < 0)|(scores > 1)):
        raise ValueError('Scores must be probabilities or NaN.')
    decoded = np.where(np.isfinite(scores),0.,np.nan)
    extension = max(1,round(2*fps/30))
    for player,start,stop in jump_segments(scores):
        if scores[start:stop,player].max() < peak_threshold:
            continue
        for _ in range(extension):
            if start > 0 and scores[start-1,player] >= takeoff_threshold:
                start -= 1
            if stop < len(scores) and scores[stop,player] >= landing_threshold:
                stop += 1
        decoded[start:stop,player] = 1.
    return decoded


def recover_jump_boundaries(verified,base,*,fps,before,after,min_score):
    """Transfer verifier confidence to nearby frames with supporting pose scores."""
    scores = np.asarray(verified,dtype=float)
    base = np.asarray(base,dtype=float)
    if (scores.ndim!=2 or scores.shape!=base.shape or not np.isfinite(fps) or fps<=0
            or not 0<=min_score<=1 or not np.isfinite([before,after]).all()
            or before<0 or after<0):
        raise ValueError('Expected aligned scores, nonnegative margins, and a positive frame rate.')
    result = scores.copy()
    supported = np.isfinite(scores)&np.isfinite(base)&(base>=min_score)
    for sign,margin in ((-1,before),(1,after)):
        reach = min(len(scores)-1,round(margin*fps/30))
        continuous = supported.copy()
        for distance in range(1,reach+1):
            shift = sign*distance
            neighbor = np.roll(scores,shift,axis=0)
            continuous &= np.roll(supported,shift,axis=0)
            if sign<0:
                continuous[-distance:] = False
            else:
                continuous[:distance] = False
            result[continuous] = np.maximum(result[continuous],neighbor[continuous])
    return result
