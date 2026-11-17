"""Regression checks for partial-foot projection and independent player gaps."""
import unittest

import cv2
import numpy as np

from badminton_analysis.positions import (
    motion_gate, position_heatmaps, positions_from_observations,
    wholebody_court_positions,
)
from badminton_analysis.replay import _candidate_positions


class PartialPositionsTest(unittest.TestCase):
    def setUp(self):
        self.points = np.zeros((1, 133, 2))
        self.points[0, 17:23] = [[1, 2], [1.2, 2], [1.1, 1.6],
                                [2, 2], [2.2, 2], [2.1, 1.6]]
        self.scores = np.ones((1, 133))
        self.H = np.array([[100., 5., 30.], [2., 50., 40.], [.01, .02, 1.]])
        self.pixels = cv2.perspectiveTransform(self.points, self.H)

    def test_complete_and_partial_feet(self):
        feet, centre = wholebody_court_positions(self.pixels, self.scores, self.H)
        np.testing.assert_allclose(feet, [[[1.1, 1.8], [2.1, 1.8]]])
        np.testing.assert_allclose(centre, [[1.6, 1.8]])
        self.scores[0, 19] = 0  # No left heel; use available toes.
        np.testing.assert_allclose(
            wholebody_court_positions(self.pixels, self.scores, self.H)[1], [[1.6, 1.9]])
        self.scores[0, 20:23] = 0  # No right foot; no fabricated radius.
        feet, centre = wholebody_court_positions(self.pixels, self.scores, self.H)
        self.assertTrue(np.isnan(feet[0, 1]).all())
        np.testing.assert_allclose(centre, [[1.1, 2.]])
        self.scores[0, 18] = 0  # A lone reliable toe is still usable.
        np.testing.assert_allclose(
            wholebody_court_positions(self.pixels, self.scores, self.H)[1], [[1., 2.]])
        self.scores[0, 17] = np.nan
        self.assertTrue(np.isnan(wholebody_court_positions(
            self.pixels, self.scores, self.H)[1]).all())

    def test_legacy_and_empty(self):
        self.scores[0, 19] = 0
        self.assertTrue(np.isnan(wholebody_court_positions(
            self.pixels, self.scores, self.H, allow_partial=False)[1]).all())
        feet, centre = wholebody_court_positions(
            np.empty((0, 133, 2)), np.empty((0, 133)), self.H)
        self.assertEqual(feet.shape, (0, 2, 2))
        self.assertEqual(centre.shape, (0, 2))

    def test_independent_halves_and_radius(self):
        points = np.repeat(self.points, 3, axis=0)
        points[2, :, 1] += 8
        scores = np.ones((3, 133))
        scores[2, 20:23] = 0
        obs = dict(frame_ids=np.arange(3), court_scores=np.array([1., 1., 0.]),
                   pose_offsets=np.array([0, 1, 3, 3]),
                   keypoints_px=points, keypoint_scores=scores)
        positions, radii, eligible = positions_from_observations(obs, np.eye(3))
        np.testing.assert_array_equal(eligible, [True, True, False])
        self.assertTrue(np.isnan(positions[0, 1]).all())
        self.assertTrue(np.isfinite(positions[1]).all())
        self.assertTrue(np.isnan(radii[1, 1]))
        self.assertAlmostEqual(radii[0, 0], .5)
        # Two candidates in the near half must not suppress the unique far half.
        obs['frame_ids'] = np.arange(1)
        obs['pose_offsets'] = np.array([0, 3])
        obs['court_scores'] = np.ones(1)
        positions, _, eligible = positions_from_observations(obs, np.eye(3))
        self.assertTrue(eligible[0])
        self.assertTrue(np.isnan(positions[0, 0]).all())
        self.assertTrue(np.isfinite(positions[0, 1]).all())
        settings = dict(foot_min_score=.4, court_margin_m=1., position_policy='partial-feet-v3')
        np.testing.assert_allclose(_candidate_positions(obs, np.eye(3), settings), positions)

    def test_replay_of_old_analysis_keeps_strict_feet(self):
        self.scores[0, 20:23] = 0
        obs = dict(frame_ids=np.arange(1), pose_offsets=np.array([0, 1]),
                   keypoints_px=self.points, keypoint_scores=self.scores)
        settings = dict(foot_min_score=.4, court_margin_m=1.)
        self.assertTrue(np.isnan(_candidate_positions(obs, np.eye(3), settings)).all())
        settings['position_policy'] = 'partial-feet-v3'
        points = _candidate_positions(obs, np.eye(3), settings)
        np.testing.assert_allclose(points[0, 0], [1.1, 1.8])
        self.assertTrue(np.isnan(points[0, 1]).all())

    def test_motion_with_missing_players_and_gaps(self):
        positions = np.full((8, 2, 2), np.nan)
        positions[:, 0] = np.column_stack([np.arange(8)*.1, np.ones(8)])
        shuttle = np.full((8, 2), np.nan)
        valid = np.ones(8, bool)
        self.assertTrue(motion_gate(positions, shuttle, valid).all())
        positions[:, 1] = [3., 10.]  # Visible stationary opponent still vetoes fallback.
        self.assertFalse(motion_gate(positions, shuttle, valid).any())
        shuttle[:] = np.column_stack([np.arange(8)*20, np.ones(8)])
        self.assertTrue(motion_gate(positions, shuttle, valid).all())
        positions[:, 1] = np.nan
        positions[:3, 0] = [1., 1.]
        positions[3:5, 0] = np.nan
        positions[5:, 0] = [4., 1.]
        shuttle[:] = np.nan
        self.assertFalse(motion_gate(positions, shuttle, valid).any())
        valid[4] = False
        shuttle[:] = np.column_stack([np.arange(8)*20, np.ones(8)])
        keep = motion_gate(positions, shuttle, valid)
        self.assertFalse(keep[3:5].any())
        self.assertTrue(keep[:3].all())
        self.assertTrue(keep[5:].all())

    def test_heatmap_counts_only_observed_players(self):
        positions = np.full((3, 2, 2), np.nan)
        positions[:, 0] = [2., 2.]
        positions[0, 1] = [3., 10.]
        counts = position_heatmaps(positions, 30.)['frame_counts'].sum(axis=(1, 2))
        np.testing.assert_array_equal(counts, [3, 1])


if __name__ == '__main__':
    unittest.main()
