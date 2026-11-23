"""Apply the existing court-agreement threshold to a reviewed pose cache."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.positions import court_agreement, detect_court
from badminton_analysis.train_rgb_mgmae import _source_frames


def check(video, pose_cache, reference, output, minimum):
    if output.exists():
        raise FileExistsError(output)
    if not np.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError('The court score threshold must be between zero and one.')
    with np.load(pose_cache, allow_pickle=False) as archive:
        data = dict(archive)
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 3])
    try:
        if not capture.isOpened():
            raise FileNotFoundError(video)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        size = (960, round(height * 960 / width))
        image = next(_source_frames(capture, [reference]))
        court = detect_court(cv2.resize(image, size, interpolation=cv2.INTER_AREA))['H_court_to_image']
        scores = np.array([court_agreement(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), court)
                           for frame in _source_frames(capture, data['frames'])], dtype=np.float32)
    finally:
        capture.release()
    data['court_scores'] = scores
    data['court_valid'] = scores >= minimum
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(f'{pose_cache.stem}: {data["court_valid"].sum()}/{len(scores)} frames agree with the reference court.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('pose_cache', type=Path)
    parser.add_argument('reference', type=int)
    parser.add_argument('output', type=Path)
    parser.add_argument('--minimum', type=float, default=.45)
    args = parser.parse_args()
    cv2.setNumThreads(2)
    check(args.video, args.pose_cache, args.reference, args.output, args.minimum)


if __name__ == '__main__':
    main()
