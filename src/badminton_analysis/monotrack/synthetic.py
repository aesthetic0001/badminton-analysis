"""Known-truth numerical check; this is not a real-world 3D accuracy benchmark."""
import numpy as np

from .reconstruction import Court3D, fit_shot, project, simulate


def fixture(fps):
    # An independent pinhole projection, not a camera estimated from test footage.
    projection = np.array([[800., 320., -50., 1200.],
                           [0., 180., -650., 4500.], [0., 1., 0., 12.]])
    world = np.array([[0, 0, 0], [6.1, 0, 0], [0, 13.4, 0], [6.1, 13.4, 0],
                      [0, 6.7, 1.55], [6.1, 6.7, 1.55]])
    pixels = np.c_[world, np.ones(6)] @ projection.T
    pixels = pixels[:, :2] / pixels[:, 2:]
    camera = Court3D(pixels[:4].tolist(), pixels[4:].tolist())
    frame_ids = np.arange(round(1.2 * fps) + 1)
    parameters = np.array([2., 2., 1.7, 2., 16., 8., .12])
    # Generate truth with an independent small-step RK4 integrator, rather than
    # the solve_ivp integration used by the fitter.
    state = parameters[:6].copy()
    step = 1 / (fps * 40)
    def derivative(s):
        v = s[3:]
        return np.r_[v, np.array([0, 0, -9.8]) - parameters[-1] * np.linalg.norm(v) * v]
    points = [state[:3].copy()]
    for i in range(len(frame_ids) - 1):
        for _ in range(40):
            a = derivative(state)
            b = derivative(state + a * step / 2)
            c = derivative(state + b * step / 2)
            d = derivative(state + c * step)
            state += (a + 2*b + 2*c + d) * step / 6
        points.append(state[:3].copy())
    points = np.array(points)
    return camera, projection, frame_ids, points, project(camera, points)


def validate():
    results = []
    for fps in (30, 60):
        camera, projection, ids, truth, observations = fixture(fps)
        estimate, diagnostics = fit_shot(camera, observations, ids, fps, truth[0, :2], truth[-1, :2])
        error = float(np.max(np.linalg.norm(estimate - truth, axis=1)))
        camera_error = float(np.max(np.abs(camera.camProj - projection / projection[-1, -1])))
        if not diagnostics["optimizer_success"] or error > .001 or camera_error > 1e-7:
            raise AssertionError(f"Synthetic reconstruction failed at {fps} FPS: {error}m")
        results.append({"fps": fps, "frames": len(ids), "max_3d_error_m": error,
                        "camera_max_coefficient_error": camera_error, **diagnostics})
    return {"passed": True, "kind": "synthetic known-camera/known-endpoint quadratic-drag flight",
            "limitation": "Validates implementation and timing only; not real footage accuracy or paper reproduction",
            "cases": results}
