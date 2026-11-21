"""Cache five-frame RGB views of both shoes on fixed-camera match footage."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.train_rgb_mgmae import _source_frames
from refine_jump_edges import _landmark_mean


PATCH_SIZE = (32, 24)
OFFSETS = (-2, -1, 0, 1, 2)


def patch_transforms(points, scores, boxes):
    shoes = np.stack([_landmark_mean(points[..., start:stop, :], scores[..., start:stop])
                      for start, stop in ((8, 11), (11, 14))], axis=2)
    heights = boxes[..., 3] - boxes[..., 1]
    valid_box = np.isfinite(heights) & (heights > 0)
    visible = np.isfinite(shoes).all(axis=-1) & valid_box[..., None]
    transform = np.zeros((*visible.shape, 2, 3), np.float32)
    width, height = PATCH_SIZE
    # Leave more room below the shoe to include the court and its shadow.
    transform[..., 0, 0] = .56 * heights[..., None] / (width - 1)
    transform[..., 1, 1] = .40 * heights[..., None] / (height - 1)
    transform[..., 0, 2] = shoes[..., 0] - .28 * heights[..., None]
    transform[..., 1, 2] = shoes[..., 1] - .15 * heights[..., None]
    return transform, visible


def extract(cache, video, destination):
    cv2.setNumThreads(2)
    with np.load(cache, allow_pickle=False) as data:
        frames = data['frames']
        transforms, visible = patch_transforms(data['points'], data['scores'], data['boxes'])
    width, height = PATCH_SIZE
    patches = np.zeros((*visible.shape, len(OFFSETS), height, width, 3), dtype=np.uint8)
    locations = [np.argwhere(row) for row in visible]

    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 3])
    try:
        if not capture.isOpened():
            raise FileNotFoundError(video)
        for index, bgr in enumerate(_source_frames(capture, frames)):
            for column, offset in enumerate(OFFSETS):
                center = index - offset
                if not 0 <= center < len(frames) or frames[index] - frames[center] != offset:
                    continue
                # The same image coordinates are used across time to preserve motion.
                for player, foot in locations[center]:
                    patch = cv2.warpAffine(bgr, transforms[center, player, foot], PATCH_SIZE,
                                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                           borderMode=cv2.BORDER_REPLICATE)
                    patches[center, player, foot, column] = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, frames=frames, patches=patches, visible=visible)
    print(f'{destination.stem}: {visible.sum()} of {visible.size} shoe tracks', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose_cache', type=Path)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    extract(args.pose_cache, args.video, args.output)


if __name__ == '__main__':
    main()
