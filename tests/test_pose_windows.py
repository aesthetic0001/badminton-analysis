import unittest

import numpy as np

from badminton_analysis.jumps import HALF_WINDOW, LANDMARKS, PoseSequence, pose_windows
from badminton_analysis.train_jumps import airborne_boundary_frames


class PoseWindowTest(unittest.TestCase):
    def test_fractional_spacing_preserves_center_and_does_not_cross_cuts(self):
        joints = len(LANDMARKS)
        ids = np.arange(17)
        court = np.ones(len(ids),bool)
        court[5] = False
        points = np.zeros((len(ids),2,joints,2))
        points[...,0] = ids[:,None,None]
        confidence = np.ones(points.shape[:-1])
        boxes = np.broadcast_to([0.,0.,10.,10.],(len(ids),2,4))
        sequence = PoseSequence(ids,court,points,confidence,boxes)

        interpolated = pose_windows(sequence,[8],[0],frame_step=.5)[0]
        self.assertTrue(interpolated[HALF_WINDOW,-1])
        self.assertAlmostEqual(interpolated[HALF_WINDOW+1,2*joints],.05)
        self.assertFalse(interpolated[0,-1])
        self.assertFalse(interpolated[1,-1])

        subsampled = pose_windows(sequence,[8,5],[0,0],frame_step=2.)
        self.assertTrue(subsampled[0,HALF_WINDOW-1,-1])
        self.assertFalse(subsampled[0,HALF_WINDOW-2,-1])
        self.assertFalse(subsampled[1].any())

    def test_boundary_training_weights_use_verified_frame_edges(self):
        targets = np.array([0,-1,1,1,1,1,1,1,1,-1,0,1,1,0])[:,None]
        index = (np.arange(len(targets)),np.zeros(len(targets),dtype=int))
        selected = np.flatnonzero(airborne_boundary_frames(targets,index))
        np.testing.assert_array_equal(selected,[2,3,7,8,11,12])
