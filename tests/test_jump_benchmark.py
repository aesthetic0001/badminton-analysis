import unittest

import numpy as np

from badminton_analysis.jump_benchmark import evaluate_jump_scores


class JumpBenchmarkTest(unittest.TestCase):
    def test_duration_aware_matching_keeps_frame_false_positives(self):
        targets = np.zeros((40, 1), dtype=int)
        targets[7:11] = 1
        targets[20:30] = 1
        scores = np.zeros_like(targets, dtype=float)
        scores[5:11] = 1  # Two frames early for a four-frame jump.
        scores[18:32] = 1  # Two frames off each boundary of a longer jump.
        report = evaluate_jump_scores(targets, scores, fps=30)
        self.assertEqual(report['matched_events_with_tolerance'], 1)
        self.assertEqual(report['false_positive_frames'], 6)
        self.assertEqual(report['strict_frame_recall'], 1)

    def test_uncertain_boundaries_missing_scores_and_extra_events(self):
        targets = np.array([[-2], [0], [-1], [-1], [1], [1], [-1], [0], [0], [0]])
        scores = np.array([[1], [np.nan], [1], [1], [1], [1], [1], [0], [1], [0]])
        report = evaluate_jump_scores(targets, scores, fps=30)
        self.assertEqual(report['matched_events_with_tolerance'], 1)
        self.assertEqual(report['predicted_events'], 2)
        self.assertEqual(report['missing_negative_scores'], 1)
        self.assertAlmostEqual(report['frame_false_positive_rate'], 1/3)


if __name__ == '__main__':
    unittest.main()
