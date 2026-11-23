import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from badminton_analysis.train_rgb_mgmae import CENTER, _pretrain_centers


class RGBPretrainingTest(unittest.TestCase):
    def test_short_clip_keeps_only_complete_context_windows(self):
        with TemporaryDirectory() as directory:
            intervals = Path(directory) / 'rallies.json'
            intervals.write_text(json.dumps([{'start_frame': 4, 'end_frame': 32}]))
            centers = _pretrain_centers(intervals, 34, 30, {'train': {}, 'validation': {}}, 7)
        np.testing.assert_array_equal(centers, 4 + CENTER + np.arange(6)*3)
