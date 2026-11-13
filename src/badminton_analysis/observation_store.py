"""Persistent, chunked model observations for the player-position notebook.

NPZ files contain only numeric arrays (never pickle). A manifest identifies the
source and extraction settings; analysis thresholds intentionally live elsewhere.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np


def file_fingerprint(path, *, content=False):
    """Identify a video by path/stat, or hash small code/model files in full."""
    path = Path(path).resolve()
    stat = path.stat()
    if content:
        with path.open("rb") as stream:
            return {"path": str(path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def settings_id(settings):
    encoded = json.dumps(settings, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()[:24]


def _atomic_write(path, writer):
    """Publish a complete file only after flushing it, even on interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_npz(path, arrays):
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Observation files must not contain object arrays.")
    _atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays))


def _load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


class ObservationStore:
    """One extraction run, with immutable calibration and resumable chunks.

    Required metadata: first_frame_id, stop_frame_id, chunk_frames, fps.
    Include the video fingerprint, model identities and extraction settings too.
    """
    schema_version = 1

    def __init__(self, root, metadata):
        if metadata.get("schema_version", self.schema_version) != self.schema_version:
            raise ValueError("Unsupported observation schema version.")
        self.manifest = json.loads(json.dumps({"schema_version": self.schema_version, **metadata}, allow_nan=False))
        first, stop, chunk = (self.manifest[key] for key in
                              ("first_frame_id", "stop_frame_id", "chunk_frames"))
        if not all(isinstance(value, int) for value in (first, stop, chunk)) or first < 0 or stop <= first or chunk < 1:
            raise ValueError("Use integer frame bounds and a positive chunk size.")
        if not np.isfinite(self.manifest["fps"]) or self.manifest["fps"] <= 0:
            raise ValueError("FPS must be finite and positive.")
        self.run_id = settings_id(self.manifest)
        self.directory = Path(root) / self.run_id
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != self.manifest:
                raise ValueError(f"Observation manifest mismatch: {manifest_path}")
        else:
            payload = (json.dumps(self.manifest, indent=2, sort_keys=True) + "\n").encode()
            _atomic_write(manifest_path, lambda stream: stream.write(payload))

    @classmethod
    def open(cls, directory):
        """Replay an existing run without its source video, models or code files."""
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if settings_id(manifest) != directory.name:
            raise ValueError("Observation manifest does not match its run directory.")
        return cls(directory.parent, manifest)

    def validate_resume(self, metadata):
        """Allow backend/code updates while preserving input and model identity.

        Explicit resumption may change chunk size or mix CPU/GPU observations.
        New chunks refer to their own extraction metadata; legacy chunks use
        the run manifest.
        """
        required = ("video", "fps", "source_size_px", "first_frame_id", "stop_frame_id",
                    "processing_width", "context_frames_each_side",
                    "court_max_lines", "extraction_revision")
        self._chunk_size(metadata["chunk_frames"])
        mismatches = [name for name in required if self.manifest.get(name) != metadata.get(name)]
        old_models, new_models = self.manifest["models"], metadata["models"]
        for name in (set(old_models) | set(new_models)) - {"rtmpose_device", "inference_device"}:
            if old_models.get(name) != new_models.get(name):
                mismatches.append(f"models.{name}")
        if mismatches:
            raise ValueError("Selected raw run is incompatible with these extraction settings: "
                             + ", ".join(sorted(mismatches)))

    def record_extraction(self, metadata):
        """Record small provenance metadata referenced by new binary chunks."""
        self.validate_resume(metadata)
        extraction_id = settings_id(metadata)
        path = self.directory / "extractions" / f"{extraction_id}.json"
        if path.exists():
            if json.loads(path.read_text()) != metadata:
                raise ValueError(f"Extraction metadata mismatch: {path}")
        else:
            payload = (json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            _atomic_write(path, lambda stream: stream.write(payload))
        return extraction_id

    def _chunk_size(self, size=None):
        size = self.manifest["chunk_frames"] if size is None else size
        if not isinstance(size, int) or size < 1:
            raise ValueError("chunk_frames must be a positive integer.")
        return size

    def saved_chunks(self):
        """Index committed frame intervals, independently of the original grid."""
        bounds = []
        for path in (self.directory / "chunks").glob("*.npz"):
            match = re.fullmatch(r"(\d+)-(\d+)\.npz", path.name)
            if match is None:
                raise ValueError(f"Invalid chunk filename: {path}")
            first, stop = map(int, match.groups())
            self._chunk_path(first, stop)
            bounds.append((first, stop))
        bounds.sort()
        for previous, current in zip(bounds, bounds[1:]):
            if current[0] < previous[1]:
                raise ValueError(f"Overlapping saved chunks: {previous} and {current}")
        return bounds

    def chunk_bounds(self, chunk_frames=None):
        """Return saved intervals plus the proposed batches for missing frames."""
        return sorted(self.saved_chunks() + self.missing_chunks(chunk_frames))

    def _chunk_path(self, first, stop):
        start, end = (self.manifest[key] for key in ("first_frame_id", "stop_frame_id"))
        if not start <= first < stop <= end:
            raise ValueError("Chunk bounds must lie inside this extraction run.")
        return self.directory / "chunks" / f"{first:09d}-{stop:09d}.npz"

    def missing_chunks(self, chunk_frames=None):
        """Partition uncovered intervals using the requested new batch size."""
        size = self._chunk_size(chunk_frames)
        cursor, end = (self.manifest[key] for key in ("first_frame_id", "stop_frame_id"))
        missing = []
        for first, stop in self.saved_chunks() + [(end, end)]:
            missing.extend((start, min(start + size, first))
                           for start in range(cursor, first, size))
            cursor = stop
        return missing

    def save_calibration(self, *, reference_image, reference_H, reference_corners, reference_score):
        path = self.directory / "calibration.npz"
        if path.exists():
            raise FileExistsError(path)
        _save_npz(path, {"reference_image_bgr": reference_image,
                         "reference_frame_id": np.asarray(self.manifest["first_frame_id"]),
                         "source_frame_start": np.asarray(self.manifest["first_frame_id"]),
                         "source_frame_stop": np.asarray(self.manifest["stop_frame_id"]),
                         "H_court_to_image": reference_H,
                         "reference_corners_px": reference_corners,
                         "reference_score": np.asarray(reference_score)})

    def load_calibration(self):
        path = self.directory / "calibration.npz"
        return _load_npz(path) if path.is_file() else None

    def _validate_chunk(self, data, first, stop):
        n = stop - first
        for name, expected in (("chunk_frame_start", first), ("chunk_frame_stop", stop),
                               ("chunk_frame_count", n)):
            if name in data:
                value = data[name]
                if value.shape != () or value.dtype.kind not in "iu" or int(value) != expected:
                    raise ValueError(f"{name} does not match this chunk's frame interval.")
        ids = data["frame_ids"]
        offsets = data["pose_offsets"]
        if ids.dtype.kind not in "iu" or not np.array_equal(ids, np.arange(first, stop)):
            raise ValueError("Raw observations must retain every absolute source frame ID.")
        if (offsets.shape != (n + 1,) or offsets.dtype.kind not in "iu"
                or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("Invalid ragged pose offsets.")
        people = int(offsets[-1])
        shapes = {"keypoints_px": (people, 133, 2), "keypoint_scores": (people, 133),
                  "boxes_xyxy": (people, 4), "box_scores": (people,),
                  "shuttle_pixels": (n, 2), "shuttle_detected": (n,),
                  "court_scores": (n,), "timestamps_seconds": (n,)}
        for name, shape in shapes.items():
            if data[name].shape != shape or data[name].dtype.kind not in "biuf":
                raise ValueError(f"{name} must be a numeric array with shape {shape}.")
        np.testing.assert_allclose(data["timestamps_seconds"], ids / self.manifest["fps"])

    def save_chunk(self, first, stop, **data):
        path = self._chunk_path(first, stop)
        if path.exists():
            raise FileExistsError(path)
        for saved_first, saved_stop in self.saved_chunks():
            if first < saved_stop and saved_first < stop:
                raise ValueError(f"New chunk overlaps saved frames [{saved_first}, {saved_stop}).")
        data = {name: np.asarray(value) for name, value in data.items()}
        for name, value in (("chunk_frame_start", first), ("chunk_frame_stop", stop),
                            ("chunk_frame_count", stop - first)):
            data.setdefault(name, np.asarray(value, dtype=np.int64))
        self._validate_chunk(data, first, stop)
        _save_npz(path, data)

    def load_chunk(self, first, stop):
        path = self._chunk_path(first, stop)
        if not path.is_file():
            return None
        data = _load_npz(path)
        self._validate_chunk(data, first, stop)
        # Normalize old files in memory without rewriting their saved arrays.
        for name, value in (("chunk_frame_start", first), ("chunk_frame_stop", stop),
                            ("chunk_frame_count", stop - first)):
            data.setdefault(name, np.asarray(value, dtype=np.int64))
        return data

    def iter_chunks(self):
        """Read in source order; never silently replay an incomplete run."""
        cursor = self.manifest["first_frame_id"]
        for first, stop in self.saved_chunks():
            if first != cursor:
                raise FileNotFoundError(f"Missing raw frames [{cursor}, {first}); resume extraction first.")
            data = self.load_chunk(first, stop)
            if data is None:
                raise FileNotFoundError(f"Missing raw frames [{first}, {stop}); resume extraction first.")
            yield data
            cursor = stop
        if cursor != self.manifest["stop_frame_id"]:
            raise FileNotFoundError(f"Missing raw frames [{cursor}, {self.manifest['stop_frame_id']}); resume extraction first.")
