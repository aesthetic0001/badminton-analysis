# Court3D adapted from MonoTrack; see LICENSE-MonoTrack.
import cv2
import numpy as np
from .court import Court
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

class Court3D:
    def __init__(self, corners, pole_tips):
        self.court2d = Court(corners)

        # The net
        self.left_post = [[0, 13.4/2, 0], [0, 13.4/2, 1.55]]
        self.right_post = [[6.1, 13.4/2, 0], [6.1, 13.4/2, 1.55]]
        self.net_line_top = [[0, 13.4/2, 1.55], [6.1, 13.4/2, 1.55]]
        self.net_line_bot = [[0, 13.4/2, 1.55 / 2.], [6.1, 13.4/2, 1.55 / 2.]]
        self.boundary = [[0, 0, 0], [6.1, 0, 0], [0, 13.4, 0], [6.1, 13.4, 0]]
        self.net = [self.left_post, self.right_post, self.net_line_top, self.net_line_bot]

        # Last two are poles
        image_points = list(corners) + list(pole_tips)
        object_points = self.boundary + self.net_line_top

        N = len(image_points)
        A = np.zeros((2*N, 11))
        y = np.zeros((2*N, 1))
        for i in range(N):
            for j in range(2):
                st = 4 if j > 0 else 0
                z = np.array(object_points[i])
                A[2*i+j, st:st+3] = z
                A[2*i+j, st+3] = 1
                A[2*i+j, -3:] = -image_points[i][j] * z
                y[2*i+j] = image_points[i][j]

        c, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
        if rank < 11:
            raise ValueError("Degenerate court/net calibration; supply six distinct, accurate points")
        C = np.zeros((3, 4))
        for i in range(3):
            for j in range(4):
                if i == 2 and j == 3:
                    C[i,j] = 1
                else:
                    C[i,j] = c[4 * i + j, 0]

        self.camProj = np.array(C)

    def pixel_to_court(self, coord):
        return self.court2d.pixel_to_court(coord)

    def project_uv(self, p3d):
        Z = self.camProj @ np.array([x for x in p3d] + [1])
        return Z[:2] / Z[2]

    def draw_lines(self, image):
        cimg = self.court2d.draw_lines(image)
        colour = (0, 0, 255)
        thickness = 3
        for line in [self.left_post, self.right_post, self.net_line_top]:
            z0, z1 = self.project_uv(np.array(line[0])), self.project_uv(np.array(line[1]))
            p0, p1 = tuple(z0.astype(int)), tuple(z1.astype(int))
            cimg = cv2.line(cimg, p0, p1, colour, thickness)
        return cimg


def simulate(parameters, times):
    """MonoTrack's quadratic-drag dynamics, in metres and actual seconds.

    parameters = [x0, y0, z0, vx0, vy0, vz0, drag]. Drag has units 1/m.
    An adaptive integrator replaces upstream's inconsistent frame adjustments.
    """
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Times must start at zero and strictly increase")
    parameters = np.asarray(parameters, dtype=float)

    def dynamics(t, state):
        velocity = state[3:]
        acceleration = np.array([0., 0., -9.8]) - parameters[6] * np.linalg.norm(velocity) * velocity
        return np.r_[velocity, acceleration]

    solution = solve_ivp(dynamics, (0, times[-1]), parameters[:6], t_eval=times,
                         rtol=1e-7, atol=1e-8)
    if not solution.success or solution.y.shape[1] != len(times):
        raise RuntimeError("Flight integration failed")
    return solution.y[:3].T


def project(camera, points):
    homogeneous = np.c_[points, np.ones(len(points))] @ camera.camProj.T
    depth = homogeneous[:, 2:3]
    # Keep optimizer residuals finite close to the camera plane.
    depth = np.where(np.abs(depth) < 1e-7, 1e-7, depth)
    return homogeneous[:, :2] / depth


def fit_shot(camera, observations, frame_ids, fps, start_xy, end_xy):
    """Adapt MonoTrack's court/player-constrained single-flight fit.

    Uses current SciPy least_squares with the same gravity/quadratic-drag model,
    actual frame timing, missing-data masks and explicit fit diagnostics. The
    soft endpoint priors allow for distance between a player's feet and racket.
    The result is a model estimate, not a measured 3D location.
    """
    observations = np.asarray(observations, dtype=float)
    frame_ids = np.asarray(frame_ids, dtype=float)
    start_xy, end_xy = np.asarray(start_xy, float), np.asarray(end_xy, float)
    if observations.shape != (len(frame_ids), 2) or start_xy.shape != (2,) or end_xy.shape != (2,):
        raise ValueError("Expected one x/y observation per frame and two x/y endpoint locations")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("The actual capture FPS must be positive")
    if len(frame_ids) < 8 or not np.isfinite(frame_ids).all() or np.any(np.diff(frame_ids) <= 0):
        raise ValueError("A flight needs at least eight ordered frames")
    visible = np.isfinite(observations).all(axis=1)
    if visible.sum() < 8 or not np.isfinite(np.r_[start_xy, end_xy]).all():
        raise ValueError("A flight needs eight visible shuttle positions and both player locations")
    times = (frame_ids - frame_ids[0]) / fps
    duration = times[-1]
    if duration > 10:
        raise ValueError("Flights longer than ten seconds need hit segmentation")
    lower = np.array([-1., -2., .1, -150., -150., -150., 0.])
    upper = np.array([7.1, 15.4, 6., 150., 150., 150., .4])
    velocity = np.r_[(end_xy - start_xy) / duration, 9.8 * duration / 2]
    initial = np.r_[start_xy, 1.7, velocity, .2]
    initial = np.clip(initial, lower + 1e-5, upper - 1e-5)

    def residual(parameters):
        points = simulate(parameters, times)
        pixels = project(camera, points)
        return np.r_[((pixels - observations)[visible] / 2).ravel(),
                     (points[0, :2] - start_xy) / .75,
                     (points[-1, :2] - end_xy) / .75,
                     np.minimum(points[:, 2], 0) * 10]

    fit = least_squares(residual, initial, bounds=(lower, upper), max_nfev=400,
                        ftol=1e-9, xtol=1e-9, gtol=1e-9)
    points = simulate(fit.x, times)
    rms = float(np.sqrt(np.mean(np.sum((project(camera, points)[visible] - observations[visible]) ** 2, axis=1))))
    diagnostics = {"optimizer_success": bool(fit.success), "message": fit.message,
                   "evaluations": fit.nfev, "reprojection_rms_px": rms,
                   "visible_observations": int(visible.sum()), "fps": fps,
                   "parameters": fit.x.tolist(),
                   "interpretation": "Physics-model estimate; reprojection error does not validate 3D accuracy"}
    return points, diagnostics

