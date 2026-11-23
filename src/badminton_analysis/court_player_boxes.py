"""Associate singles player detections with the calibrated court."""

import cv2
import numpy as np

from badminton_analysis.positions import court_length_m, court_width_m


def select_court_players(boxes, scores, court_to_image, previous=None):
    """Return near and far player boxes, leaving unmatched slots empty.

    The image of a raised foot can fall outside the court plane. Allow a margin
    around the playing area and use the previous location to retain the track.
    """
    boxes, scores = np.asarray(boxes), np.asarray(scores)
    selected = np.full((2,4),np.nan)
    if not len(boxes):
        return selected
    H = np.asarray(court_to_image,dtype=float)
    across = court_width_m/2
    ends = cv2.perspectiveTransform(np.array([[[across,0.],[across,court_length_m]]]),H)[0]
    near_is_high_y = ends[1,1] > ends[0,1]
    bottom = np.column_stack(((boxes[:,0]+boxes[:,2])/2,boxes[:,3]))
    position = cv2.perspectiveTransform(bottom[None].astype(float),np.linalg.inv(H))[0]
    if not near_is_high_y:
        position[:,1] = court_length_m-position[:,1]
    x,y = position.T
    height = boxes[:,3]-boxes[:,1]
    x_outside = np.maximum.reduce((-x,x-court_width_m,np.zeros(len(x))))
    y_outside = np.maximum.reduce((-y,y-court_length_m,np.zeros(len(y))))
    eligible = (x_outside<=1.6)&(y_outside<=2.3)&(height>0)&np.isfinite(position).all(axis=1)
    side = (y<court_length_m/2).astype(int)
    score = scores-.18*x_outside-.13*y_outside
    for slot in range(2):
        candidates = np.flatnonzero(eligible & (side==slot))
        if not len(candidates):
            continue
        rank = score[candidates]
        prior = None if previous is None else previous[slot]
        if prior is not None and np.isfinite(prior).all():
            prior_foot = np.array([(prior[0]+prior[2])/2,prior[3]])
            displacement = np.linalg.norm(bottom[candidates]-prior_foot,axis=1)/(prior[3]-prior[1])
            rank = rank-.65*displacement
        selected[slot] = boxes[candidates[np.argmax(rank)]]
    return selected
