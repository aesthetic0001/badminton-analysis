import unittest

import numpy as np

from badminton_analysis.jumps import collect_player_poses, rectify_jumps


class JumpPositionTest(unittest.TestCase):
    def test_projected_shoes_cross_net_without_creating_a_far_player(self):
        projection = np.array([[1., 0, .1, 0], [0, 1., -.7, 0], [0, 0, 0, 1.]])
        frame_ids = np.arange(13)
        times = (frame_ids - 1) / 30
        expected = np.column_stack((3.05 + .15 * times, 6.4 + .1 * times))
        z = 1. + 1.7 * times - 4.905 * times ** 2
        keypoints = np.zeros((len(frame_ids), 133, 2))
        keypoints[:, 11:13] = (expected + z[:, None] * [.1, -.7])[:, None]
        keypoints[:, 17:23] = expected[:, None]
        keypoints[1:-1, 17:23, 1] += .8
        chunk = {
            'frame_ids': frame_ids,
            'court_scores': np.ones(len(frame_ids)),
            'keypoints_px': keypoints,
            'keypoint_scores': np.ones(keypoints.shape[:2]),
            'boxes_xyxy': np.tile([2., 4., 4., 7.], (len(frame_ids), 1)),
            'pose_offsets': np.arange(len(frame_ids) + 1),
        }

        class Store:
            def iter_chunks(self):
                yield chunk

        sequence, stances, radii = collect_player_poses(Store(), np.eye(3), include_stance=True)
        np.testing.assert_array_equal(sequence.present, np.tile([True, False], (len(frame_ids), 1)))
        self.assertTrue(np.all(stances[1:-1, 0, 1] > 6.7))
        self.assertTrue(np.isnan(stances[:, 1]).all())
        self.assertTrue(np.isnan(radii[:, 1]).all())

        probabilities = np.zeros((len(frame_ids), 2))
        probabilities[1:-1, 0] = .99
        corrected, _, events = rectify_jumps(sequence, stances, probabilities, projection, 30)
        self.assertIsNotNone(events[0]['median_reprojection_px'])
        np.testing.assert_allclose(corrected[:, 0], expected, atol=1e-5)
        self.assertTrue(np.isnan(corrected[:, 1]).all())
