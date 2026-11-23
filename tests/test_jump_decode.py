import unittest

import numpy as np

from badminton_analysis.jump_decode import decode_jump_scores, recover_jump_boundaries


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

    def test_boundary_recovery_is_bounded_and_requires_contiguous_support(self):
        verified = np.array([[.2],[.2],[.2],[.99],[.2],[np.nan],[.2],[.98]])
        base = np.array([[.5],[.5],[.5],[.95],[.5],[np.nan],[.3],[.95]])
        result = recover_jump_boundaries(verified,base,fps=30,before=2,after=1,min_score=.4)
        np.testing.assert_array_equal(result[:,0],[.2,.99,.99,.99,.99,np.nan,.2,.98])
