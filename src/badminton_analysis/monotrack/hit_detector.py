# Adapted from MonoTrack; see LICENSE-MonoTrack.
import numpy as np
from .pose import Pose

class HitDetector(object):
    def __init__(self, court, poses, trajectory):
        self.court = court
        self.poses = poses
        self.trajectory = trajectory

    # Returns the hits in the given trajectory as well as who hit it
    # Output is are two lists of values:
    #   - List of frame ids where things are hit
    #   - 0 (no hit), 1 (bottom player hits), 2 (top player hits)
    def detect_hits(self):
        pass

class AdhocHitDetector(HitDetector):
    def __init__(self, poses, trajectory):
        super().__init__(None, poses, trajectory)

    def _detect_hits_1d(self, z, thresh=4, window=8):
        # For a hit to be registered, the point must be a local max / min and
        # the slope must exceed thresh on either the left or right side
        # The slope is averaged by the window parameter to remove noise
        z = np.array(z)
        bpts = []
        for i in range(window+1, len(z)-window-1):
            if not np.isfinite(z[i-window:i+window+1]).all():
                continue
            if (z[i]-z[i-1]) * (z[i]-z[i+1]) < 0:
                continue

            # This is a local opt
            left = abs(np.median(z[i-window+1:i+1] - z[i-window:i]))
            right = abs(np.median(z[i+1:i+window+1] - z[i:i+window]))
            if max(left, right) > thresh:
                bpts.append(i)
        return bpts

    def _merge_hits(self, x, y, closeness=2):
        bpt = []
        for t in sorted(x + y):
            if len(bpt) == 0 or bpt[-1] < t - closeness:
                bpt.append(t)
        return bpt

    def _detect_hits(self, x, y, thresh=10, window=7, closeness=15):
        return self._merge_hits(
            self._detect_hits_1d(x, thresh, window),
            self._detect_hits_1d(y, thresh, window),
            closeness
        )

    def detect_hits(self, fps=25):
        Xb, Yb = self.trajectory.X, self.trajectory.Y
        result = self._detect_hits(Xb, Yb)

        # Filter hits by pose
        is_hit = []
        last_hit = -1
        # Filter hits by velocity
        avg_hit = np.average(np.diff(result)) if len(result) > 1 else np.inf
        for i, fid in enumerate(result):
            if i+1 < len(result) and result[i+1] - fid > 1.6 * avg_hit:
                is_hit.append(0)
                continue

            if fid >= self.poses[0].values.shape[0]:
                break

            c = np.array([Xb[fid], Yb[fid]])
            reached_by = 0
            dist_reached = 1e99
            for j in range(2):
                xy = self.poses[j].iloc[fid].to_list()
                pose = Pose()
                pose.init_from_kparray(xy)
                if pose.can_reach(c):
                    pdist = np.linalg.norm(c - pose.get_centroid())
                    if not reached_by or pdist < dist_reached:
                        reached_by = j + 1
                        dist_reached = pdist

            if reached_by:
                last_hit = reached_by
            is_hit.append(reached_by)

        return result, is_hit
