import unittest

import numpy as np

from badminton_analysis.jump_decode import decode_jump_scores


class JumpDecodeTest(unittest.TestCase):
    def test_extends_confirmed_event_and_removes_unconfirmed_flag(self):
        scores = np.array([[.2], [.2], [.2], [.6], [.99], [.4], [.3],
                           [0.], [.7], [.8], [0.]])
        decoded = decode_jump_scores(scores,peak_threshold=.98,takeoff_threshold=.15,
                                     landing_threshold=.3)
        np.testing.assert_array_equal(decoded[:,0],[0,1,1,1,1,1,1,0,0,0,0])

    def test_keeps_confirmed_short_event_and_stops_at_missing_data(self):
        scores = np.array([[.2, .2], [np.nan, .1], [.99, .1], [.3, .1], [.2, .1]])
        decoded = decode_jump_scores(scores,peak_threshold=.98,takeoff_threshold=.1,
                                     landing_threshold=.3)
        np.testing.assert_array_equal(decoded[:,0],[0,np.nan,1,1,0])
        np.testing.assert_array_equal(decoded[:,1],np.zeros(len(scores)))
