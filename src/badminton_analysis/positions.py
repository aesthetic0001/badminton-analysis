"""Court search and foot projection shared by the position notebooks."""
from itertools import combinations

import cv2
import numpy as np


def deduplicate_hough_lines(
    raw_lines, image_shape, angle_tolerance_deg=2, distance_tolerance_px=10,
    max_lines=40,
):
    """Return distinct (rho, theta) lines, retaining stronger Hough peaks first."""
    if raw_lines is None:
        raise ValueError("No Hough lines found; inspect the mask and Hough threshold.")

    height, width = image_shape[:2]
    centre = np.array([width / 2, height / 2])
    angle_tolerance = np.deg2rad(angle_tolerance_deg)
    kept = []
    kept_normals = []

    for rho, theta in np.asarray(raw_lines).reshape(-1, 2):
        normal = np.array([np.cos(theta), np.sin(theta)])
        if kept:
            normals = np.asarray(kept_normals)
            alignment = np.where(normals @ normal >= 0, 1.0, -1.0)
            aligned_normals = normals * alignment[:, None]
            aligned_rhos = np.asarray(kept)[:, 0] * alignment
            angles = np.arccos(np.clip(aligned_normals @ normal, -1, 1))
            separation = np.abs(
                (normal - aligned_normals) @ centre - rho + aligned_rhos
            )
            if np.any((angles < angle_tolerance) & (separation < distance_tolerance_px)):
                continue

        kept.append((rho, theta))
        kept_normals.append(normal)
        if len(kept) >= max_lines:
            break

    return np.asarray(kept, dtype=np.float64).reshape(-1, 2)


def partition_hough_lines(lines, epsilon=0.01, n_starts=20, seed=0):
    """Greedily maximize cross-family weights; return two (rho, theta) arrays."""
    lines = np.asarray(lines, dtype=np.float64).reshape(-1, 2)
    if len(lines) < 2:
        raise ValueError("At least two distinct lines are needed for partitioning.")
    if epsilon <= 0 or n_starts < 1:
        raise ValueError("epsilon and n_starts must be positive.")

    # Normal-angle differences also measure line-angle differences.
    # abs(cos(delta)) treats a line as undirected: theta and theta + pi agree.
    theta = lines[:, 1]
    delta = theta[:, None] - theta[None, :]
    angle = np.arccos(np.clip(np.abs(np.cos(delta)), 0, 1))
    weights = 1.0 / (np.abs(angle - np.pi / 2) + epsilon)**2
    np.fill_diagonal(weights, 0.0)

    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best_labels = None

    for _ in range(n_starts):
        labels = rng.integers(0, 2, size=len(lines)).astype(bool)
        while True:
            same_family = labels[:, None] == labels[None, :]
            # Moving i gains its within-family edges and loses its cross-family edges.
            gains = np.sum(weights * np.where(same_family, 1.0, -1.0), axis=1)
            move = int(np.argmax(gains))
            if gains[move] <= 1e-8:
                break
            labels[move] = ~labels[move]

        # Each undirected edge occurs twice in the symmetric weight matrix.
        score = weights[labels[:, None] != labels[None, :]].sum() / 2
        if score > best_score:
            best_score = score
            best_labels = labels.copy()

    return lines[~best_labels], lines[best_labels]


# Court-plane coordinates: x across the court, y along its length (metres).
court_width_m = 6.10
court_length_m = 13.40
court_model_corners = np.float32([
    [0, 0], [court_width_m, 0],
    [court_width_m, court_length_m], [0, court_length_m],
])

court_line_names = [
    "Doubles sideline 1", "Singles sideline 1",
    "Singles sideline 2", "Doubles sideline 2",
    "Back boundary 1", "Doubles long service 1", "Short service 1",
    "Short service 2", "Doubles long service 2", "Back boundary 2",
    "Centre service 1", "Centre service 2",
]
court_model_segments = np.float32(
    [[[x, 0], [x, court_length_m]] for x in [0, 0.46, 5.64, court_width_m]]
    + [[[0, y], [court_width_m, y]] for y in [0, 0.76, 4.72, 8.68, 12.64, court_length_m]]
    + [[[3.05, 0], [3.05, 4.72]], [[3.05, 8.68], [3.05, court_length_m]]]
)


def search_court_layout(
    line_mask, family_a, family_b, model_corners, model_segments,
    tolerance_px=3.0, samples_per_line=80, min_area_fraction=0.1,
):
    """Search outer-boundary hypotheses; return the best court-to-image mapping."""
    if len(family_a) < 2 or len(family_b) < 2:
        raise ValueError("Court search needs at least two lines in each family.")
    if tolerance_px <= 0 or samples_per_line < 2:
        raise ValueError("Use positive tolerance and at least two samples per line.")

    if not 0 <= min_area_fraction <= 1:
        raise ValueError("min_area_fraction must be between zero and one.")

    line_mask = np.asarray(line_mask, dtype=bool)
    if not line_mask.any():
        raise ValueError("No candidate pixels to score; inspect the line mask first.")
    height, width = line_mask.shape
    image_bounds = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    # Candidate pixels are zeros, so distances measure proximity to those pixels.
    distances = cv2.distanceTransform((~line_mask).astype(np.uint8), cv2.DIST_L2, 5)
    model_corners = np.asarray(model_corners, dtype=np.float32)
    model_segments = np.asarray(model_segments, dtype=np.float32)
    corners_h = np.column_stack([model_corners, np.ones(4)])
    endpoints_h = np.column_stack([
        model_segments.reshape(-1, 2), np.ones(model_segments.shape[0] * 2),
    ])
    t = np.linspace(0, 1, samples_per_line)[None, :, None]

    def homogeneous_lines(family):
        rho, theta = np.asarray(family).T
        return np.column_stack([np.cos(theta), np.sin(theta), -rho])

    def intersect(first, second):
        point = np.cross(first, second)
        if abs(point[2]) < 1e-8:  # Parallel lines have no finite intersection.
            return None
        return point[:2] / point[2]

    lines_a = homogeneous_lines(family_a)
    lines_b = homogeneous_lines(family_b)
    best = None
    hypotheses_scored = 0

    for a_indices in combinations(range(len(lines_a)), 2):
        a0, a1 = lines_a[list(a_indices)]
        for b_indices in combinations(range(len(lines_b)), 2):
            b0, b1 = lines_b[list(b_indices)]
            points = [intersect(a0, b0), intersect(a0, b1),
                      intersect(a1, b1), intersect(a1, b0)]
            if any(point is None for point in points):
                continue
            quad = np.asarray(points, dtype=np.float32)
            if not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
                continue
            visible_area, _ = cv2.intersectConvexConvex(quad, image_bounds)
            if visible_area < min_area_fraction * width * height:
                continue

            # Try which family defines width vs length. Reflection and 180-degree
            # alternatives have the same support because this template is symmetric.
            for shift in (0, 1):
                image_corners = np.roll(quad, shift, axis=0)
                H = cv2.getPerspectiveTransform(model_corners, image_corners)
                if not np.isfinite(H).all():
                    continue
                denominators = (corners_h @ H.T)[:, 2]
                if not (np.all(denominators > 1e-8) or np.all(denominators < -1e-8)):
                    continue  # Reject a projection that crosses infinity within the court.

                projected = endpoints_h @ H.T
                segments_px = (projected[:, :2] / projected[:, 2:]).reshape(-1, 2, 2)
                samples = segments_px[:, 0, None] * (1 - t) + segments_px[:, 1, None] * t
                pixels = np.rint(samples).astype(int)
                x, y = pixels[..., 0], pixels[..., 1]
                valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
                support = np.zeros(x.shape)
                support[valid] = np.exp(
                    -0.5 * (distances[y[valid], x[valid]] / tolerance_px)**2
                )
                sample_counts = valid.sum(axis=1)
                visible = sample_counts > 0
                if not visible.any():
                    continue
                line_support = np.divide(
                    support.sum(axis=1), sample_counts,
                    out=np.full(len(model_segments), np.nan), where=visible,
                )
                score = float(line_support[visible].mean())
                hypotheses_scored += 1

                if best is None or score > best["score"]:
                    best = {
                        "H_court_to_image": H,
                        "corners_px": image_corners.copy(),
                        "segments_px": segments_px,
                        "score": score,
                        "line_support": line_support,
                        "family_a_indices": a_indices,
                        "family_b_indices": b_indices,
                    }

    if best is None:
        raise ValueError(
            "No valid court hypothesis. Inspect the Hough lines and families; "
            "this search requires four detected outer boundary lines and sufficient visible court surface area."
        )
    best["hypotheses_scored"] = hypotheses_scored
    return best


def wholebody_court_positions(keypoints, scores, H_court_to_image, min_score=0.4):
    """Return foot centres (N, 2, 2) and stance centres (N, 2), in metres."""
    keypoints = np.asarray(keypoints, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    H = np.asarray(H_court_to_image, dtype=np.float64)
    if keypoints.ndim != 3 or keypoints.shape[1:] != (133, 2):
        raise ValueError('Expected COCO-WholeBody keypoints with shape (N, 133, 2).')
    if scores.shape != keypoints.shape[:2]:
        raise ValueError('Scores must have shape (N, 133).')
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError('Expected a finite 3x3 court-to-image homography.')
    if not np.isfinite(min_score) or min_score < 0:
        raise ValueError('Use a finite nonnegative landmark score threshold.')
    inverse = np.linalg.inv(H)
    feet_px = keypoints[:, 17:23].reshape(-1, 2, 3, 2)
    foot_scores = scores[:, 17:23].reshape(-1, 2, 3)
    homogeneous = np.concatenate([feet_px, np.ones((*feet_px.shape[:-1], 1))], axis=-1)
    projected = homogeneous @ inverse.T
    denominators = projected[..., 2]
    valid = (
        np.isfinite(projected).all(axis=-1)
        & np.isfinite(foot_scores) & (foot_scores >= min_score)
        & (np.abs(denominators) > 1e-8)
    )
    court_landmarks = np.full(feet_px.shape, np.nan)
    np.divide(projected[..., :2], denominators[..., None],
              out=court_landmarks, where=valid[..., None])
    toe_centres = court_landmarks[:, :, :2].mean(axis=2)
    foot_centres = (toe_centres + court_landmarks[:, :, 2]) / 2
    foot_centres[~valid.all(axis=2)] = np.nan
    return foot_centres, foot_centres.mean(axis=1)


def court_line_mask(frame):
    """Extract court stripe pixels using the notebook's intensity/tensor tests."""
    scale = frame.shape[0] / 1080
    blue = frame[:, :, 0].astype(np.float32)
    t = max(1, round(16 * scale))
    if 2 * t >= min(blue.shape):
        raise ValueError("Frame is too small for court detection.")
    centre = blue[t:-t, t:-t]
    mask = np.zeros(blue.shape, dtype=bool)
    mask[t:-t, t:-t] = (centre >= 80) & (
        ((centre - blue[t:-t, :-2*t] > 20) & (centre - blue[t:-t, 2*t:] > 20))
        | ((centre - blue[:-2*t, t:-t] > 20) & (centre - blue[2*t:, t:-t] > 20))
    )
    value = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    value = cv2.GaussianBlur(value, (5, 5), 0)
    gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
    size = max(3, round(21 * scale) | 1)
    jxx, jxy, jyy = [cv2.boxFilter(a, -1, (size, size)) for a in (gx*gx, gx*gy, gy*gy)]
    difference = np.sqrt((jxx-jyy)**2 + 4*jxy**2)
    mask &= (jxx+jyy+difference)/2 > 8*np.maximum((jxx+jyy-difference)/2, 0)
    return mask


def detect_court(frame, max_lines=40):
    """Apply the existing notebook's stripe, tensor and layout-search detector."""
    mask = court_line_mask(frame)
    scale = frame.shape[0] / 1080
    raw = cv2.HoughLines(mask.astype(np.uint8)*255, 1, np.pi/1080, max(20, round(200*scale)))
    lines = deduplicate_hough_lines(
        raw, frame.shape, distance_tolerance_px=max(1, 10*scale), max_lines=max_lines,
    )
    family_a, family_b = partition_hough_lines(lines)
    return search_court_layout(
        mask, family_a, family_b, court_model_corners, court_model_segments,
        tolerance_px=max(1, 3*scale),
    )


def court_agreement(frame, H_court_to_image):
    """Score support for the proposed court's markings without searching new layouts."""
    H = np.asarray(H_court_to_image, dtype=float)
    if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        raise ValueError("Expected a finite, nonsingular 3x3 court homography.")
    corners_h = np.column_stack([court_model_corners, np.ones(4)])
    denominators = (corners_h @ H.T)[:, 2]
    if not (np.all(denominators > 0) or np.all(denominators < 0)):
        raise ValueError("Court projection must not cross infinity.")
    mask = court_line_mask(frame)
    if not mask.any():
        return 0.
    # Match the layout search's per-marking distance score and sample count.
    distances = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    segments = cv2.perspectiveTransform(court_model_segments, H)
    t = np.linspace(0, 1, 80)[None, :, None]
    samples = segments[:, 0, None]*(1-t) + segments[:, 1, None]*t
    pixels = np.rint(samples).astype(int)
    x, y = pixels[..., 0], pixels[..., 1]
    height, width = mask.shape
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    counts = valid.sum(axis=1)
    if not counts.any():
        return 0.
    tolerance_px = max(1, 3*height/1080)
    support = np.zeros(x.shape)
    support[valid] = np.exp(-.5*(distances[y[valid], x[valid]]/tolerance_px)**2)
    return float(np.mean(support.sum(axis=1)[counts > 0]/counts[counts > 0]))


def align_court(layout, reference_corners):
    """Resolve court reflections/end swaps and return H plus corner RMS in pixels."""
    corners = np.asarray(layout["corners_px"], dtype=np.float32)
    # Only court symmetries: swapping width and length would change physical units.
    alternatives = corners[[[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]]
    errors = np.sqrt(np.mean(np.sum((alternatives-reference_corners)**2, axis=2), axis=1))
    best = int(np.argmin(errors))
    H = cv2.getPerspectiveTransform(court_model_corners, alternatives[best])
    return H, float(errors[best])


def motion_gate(positions, shuttle, valid, window=8, shuttle_deviation_px=20.,
                player_displacement_m=.05):
    """Keep valid frames when the shuttle OR both stances move in a trailing window."""
    positions, shuttle = np.asarray(positions, float), np.asarray(shuttle, float)
    valid = np.asarray(valid, bool)
    if positions.shape != (len(valid), 2, 2) or shuttle.shape != (len(valid), 2):
        raise ValueError("Expected positions (N, 2, 2), shuttle (N, 2), valid (N,).")
    if window < 2 or min(shuttle_deviation_px, player_displacement_m) <= 0:
        raise ValueError("Use a window >=2 and positive motion thresholds.")
    accepted = np.zeros(len(valid), dtype=bool)
    run_start = 0
    for i in range(len(valid)):
        if not valid[i] or not np.isfinite(positions[i]).all():
            run_start = i + 1
            continue
        start = max(run_start, i-window+1)
        if i-start+1 < window:
            continue
        stances = positions[start:i+1]
        player_motion = np.linalg.norm(stances-stances[0], axis=2).max(axis=0)
        detections = shuttle[start:i+1]
        detections = detections[np.isfinite(detections).all(axis=1)]
        shuttle_moves = (len(detections) >= 2 and np.linalg.norm(
            detections-detections.mean(axis=0), axis=1,
        ).max() >= shuttle_deviation_px)
        accepted[i] = shuttle_moves or (player_motion >= player_displacement_m).all()
    return accepted


def position_heatmaps(positions, fps, radius_m=.15, cell_m=.05, margin_m=1.):
    """Return raw bins and circular-presence counts/seconds on the court grid."""
    from scipy.ndimage import convolve

    positions = np.asarray(positions, float)
    if positions.ndim != 3 or positions.shape[1:] != (2, 2):
        raise ValueError("Expected positions with shape (N, 2, 2).")
    if not np.isfinite([fps, radius_m, cell_m, margin_m]).all() or min(fps, radius_m, cell_m) <= 0 or margin_m < 0:
        raise ValueError("Use finite positive FPS, radius and cell size, and a nonnegative margin.")
    edges = [np.linspace(-margin_m, length+margin_m,
                         int(np.ceil((length+2*margin_m)/cell_m))+1)
             for length in (court_width_m, court_length_m)]
    dx, dy = [np.diff(edge)[0] for edge in edges]
    rx, ry = int(np.ceil(radius_m/dx)), int(np.ceil(radius_m/dy))
    x, y = np.meshgrid(np.arange(-rx, rx+1)*dx, np.arange(-ry, ry+1)*dy)
    disk = (x*x+y*y <= radius_m**2+1e-12).astype(float)
    counts = []
    for player in range(2):
        points = positions[:, player]
        points = points[np.isfinite(points).all(axis=1)]
        counts.append(np.histogram2d(points[:, 0], points[:, 1], bins=edges)[0].T)
    counts = np.asarray(counts)
    presence = np.stack([convolve(count, disk, mode="constant") for count in counts])
    return {"x_edges_m": edges[0], "y_edges_m": edges[1], "frame_counts": counts,
            "presence_frames": presence, "dwell_seconds": presence/fps}
