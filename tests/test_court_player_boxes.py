import unittest

import numpy as np

from badminton_analysis.court_player_boxes import select_court_players


class CourtPlayerBoxesTest(unittest.TestCase):
    def test_keeps_players_and_excludes_judge_behind_baseline(self):
        camera = np.array([[60., 0., 150.], [0., 24., 180.], [0., 0., 1.]])
        far, judge, near = np.array([[310., 102., 378., 234.],
                                    [280., 54., 330., 115.],
                                    [270., 300., 370., 478.]])
        selected = select_court_players([far,judge,near],[.69,.99,.85],camera)
        np.testing.assert_array_equal(selected,[near,far])

        reverse = np.array([[-1., 0., 6.1], [0., -1., 13.4], [0., 0., 1.]])
        selected = select_court_players([far,judge,near],[.69,.99,.85],camera@reverse)
        np.testing.assert_array_equal(selected,[near,far])

    def test_recent_track_breaks_ambiguity_on_same_court_half(self):
        camera = np.array([[60., 0., 150.], [0., 24., 180.], [0., 0., 1.]])
        current = np.array([310., 102., 378., 234.])
        distractor = np.array([231., 95., 281., 237.])
        previous = [np.full(4,np.nan),current+[1., 2., 3., 4.]]
        selected = select_court_players([current,distractor],[.7,.9],camera,previous)
        self.assertTrue(np.isnan(selected[0]).all())
        np.testing.assert_array_equal(selected[1],current)
