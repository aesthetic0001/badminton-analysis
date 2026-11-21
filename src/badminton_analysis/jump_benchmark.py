"""Strict frame metrics and one-to-one, duration-aware jump event scoring."""

from bisect import bisect_right
import math

import numpy as np

from badminton_analysis.jumps import jump_segments


def _allowance(duration, fps):
    """Allow one frame for short jumps and two for longer jumps at 30 fps."""
    at_30_fps = 1 if duration / fps <= .2 else 2
    return max(1, round(at_30_fps * fps / 30))


def _boundary_range(labels, start, stop):
    first, last = start, stop
    while first and labels[first-1] == -1:
        first -= 1
    while last < len(labels) and labels[last] == -1:
        last += 1
    return first, last


def _match_events(truth, predictions, labels, fps):
    by_player = [
        [(index,start,stop) for index,(slot,start,stop) in enumerate(predictions) if slot == player]
        for player in range(labels.shape[1])
    ]
    stop_frames = [[stop for _,_,stop in events] for events in by_player]
    edges = {}
    for i,(player,start,stop) in enumerate(truth):
        lo, hi = _boundary_range(labels[:,player], start, stop)
        gap = _allowance(stop-start,fps)
        centers = []
        first_candidate = bisect_right(stop_frames[player],start)
        for index,begin,end in by_player[player][first_candidate:]:
            if begin >= stop:
                break
            onset_error = max(lo-begin, begin-start, 0)
            landing_error = max(stop-end,end-hi,0)
            if onset_error <= gap and landing_error <= gap:
                centers.append((onset_error+landing_error,index))
        edges[i] = [index for _,index in sorted(centers)]

    matched = {}
    def assign(reference, visited):
        for prediction in edges[reference]:
            if prediction in visited:
                continue
            visited.add(prediction)
            if prediction not in matched or assign(matched[prediction],visited):
                matched[prediction] = reference
                return True
        return False

    for reference in edges:
        assign(reference,set())
    return len(matched)


def evaluate_jump_scores(targets, scores, *, fps, threshold=.5):
    """Score contiguous (frame, player) arrays.

    Targets are -2 for unreviewed, -1 for uncertain, 0 for no correction, and
    1 for a jump requiring correction. Missing scores are counted explicitly.
    """
    targets,scores = np.asarray(targets),np.asarray(scores,dtype=float)
    if (targets.ndim != 2 or targets.shape != scores.shape or not math.isfinite(fps)
            or fps <= 0 or not 0 <= threshold <= 1 or not np.isin(targets,[-2,-1,0,1]).all()):
        raise ValueError('Use aligned (frame, player) targets and scores, a valid fps and threshold.')
    if np.isinf(scores).any() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('Scores must be probabilities or NaN.')
    finite = np.isfinite(scores)
    positive, negative = targets == 1, targets == 0
    flagged = finite & (scores >= threshold)
    true_positive = int((flagged & positive).sum())
    false_positive = int((flagged & negative).sum())
    positives, negatives = int(positive.sum()), int(negative.sum())
    missing_negative = int((negative & ~finite).sum())
    reference = list(jump_segments(positive))
    predictions = list(jump_segments(np.where(targets != -2,scores,np.nan),threshold))
    matched = _match_events(reference,predictions,targets,fps)
    return {
        'threshold': threshold,
        'positive_frames': positives,
        'negative_frames': negatives,
        'true_positive_frames': true_positive,
        'false_positive_frames': false_positive,
        'missing_positive_scores': int((positive & ~finite).sum()),
        'missing_negative_scores': missing_negative,
        'strict_frame_recall': true_positive/positives if positives else None,
        'frame_false_positive_rate': false_positive/(negatives-missing_negative)
        if negatives > missing_negative else None,
        'reference_events': len(reference),
        'predicted_events': len(predictions),
        'matched_events_with_tolerance': matched,
        'event_recall_with_tolerance': matched/len(reference) if reference else None,
    }


def jump_operating_points(positive_scores,negative_scores,*,recall_target=.99,fp_target=.005):
    """Find the cutoffs at the frame targets. Choose cutoffs on development footage."""
    positive = np.asarray(positive_scores,dtype=float).ravel()
    negative = np.asarray(negative_scores,dtype=float).ravel()
    if not 0 < recall_target <= 1 or not 0 < fp_target <= 1:
        raise ValueError('Recall and false-positive targets must be in (0, 1].')
    if any(np.any((scores<0)|(scores>1)) for scores in (positive,negative)):
        raise ValueError('Scores must be probabilities or NaN.')
    # Unscored positives are misses. Compute FPR on scored negatives and report coverage.
    negative = np.sort(negative[np.isfinite(negative)])
    if not len(positive) or not len(negative):
        raise ValueError('Operating points need positives and scored negatives.')
    positive = np.sort(np.nan_to_num(positive,nan=-1.))
    allowed = math.ceil(fp_target*len(negative))-1
    # The score threshold is inclusive. Move above the entire tied score group.
    specificity = float(np.nextafter(negative[-1-allowed],np.inf))
    sensitivity = float(positive[-math.ceil(recall_target*len(positive))])
    return {
        'specificity_threshold':specificity if specificity<=1 else None,
        'recall_threshold':sensitivity if sensitivity>=0 else None,
        'true_positive_at_fp_limit':int((positive>=specificity).sum()) if specificity<=1 else None,
        'false_positive_at_recall_target':int((negative>=sensitivity).sum()) if sensitivity>=0 else None,
        'positive_frames':len(positive),
        'scored_negative_frames':len(negative),
    }
