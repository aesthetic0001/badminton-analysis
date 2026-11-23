import unittest

import numpy as np

from badminton_analysis.jump_benchmark import evaluate_jump_scores, jump_operating_points


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

    def test_cutoff_obeys_strict_false_positive_limit_and_score_ties(self):
        negative = np.r_[np.full(198,.2),.8,.8,.9]
        point = jump_operating_points([.7,.85,.97],negative)
        self.assertGreater(point['specificity_threshold'],.8)
        self.assertEqual(point['true_positive_at_fp_limit'],2)
        self.assertEqual(point['false_positive_at_recall_target'],3)

        exactly_two_hundred = jump_operating_points([.9],negative[1:])
        self.assertGreater(exactly_two_hundred['specificity_threshold'],.9)

        scores = negative.astype(np.float32)[:,None]
        point = jump_operating_points([.9],scores)
        report = evaluate_jump_scores(np.zeros_like(scores),scores,fps=30,
                                      threshold=point['specificity_threshold'])
        self.assertEqual(report['false_positive_frames'],1)

    def test_cutoff_reports_unattainable_targets(self):
        point = jump_operating_points([.9,np.nan],[.2,1.,np.nan])
        self.assertIsNone(point['specificity_threshold'])
        self.assertIsNone(point['recall_threshold'])
        self.assertEqual(point['positive_frames'],2)
        self.assertEqual(point['scored_negative_frames'],2)


if __name__ == '__main__':
    unittest.main()
