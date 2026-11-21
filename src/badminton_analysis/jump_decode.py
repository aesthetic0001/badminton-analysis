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
