"""Render source footage beside cached court positions, without model inference."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import wave
from fractions import Fraction

import cv2
import numpy as np

from .observation_store import ObservationStore
from .positions import (
    court_length_m, court_width_m, court_model_segments, wholebody_court_positions,
)


PLAYER_COLOURS = ((240, 210, 45), (40, 155, 255))  # BGR: cyan, orange.


def _video_encoding(encoder):
    if encoder == 'h264_nvenc':
        return ['-c:v', encoder, '-preset', 'p4', '-rc', 'vbr', '-cq', '23', '-b:v', '0']
    if encoder == 'libx264':
        return ['-c:v', encoder, '-preset', 'veryfast', '-crf', '23', '-threads', '4']
    raise ValueError('Choose libx264 or h264_nvenc.')


def _text(image, message, xy, colour=(230, 230, 230), size=.48):
    cv2.putText(image, message, xy, cv2.FONT_HERSHEY_SIMPLEX, size,
                colour, 1, cv2.LINE_AA)


def _candidate_positions(chunk, homography, settings):
    """Show unique candidates per half even if the other player is missing."""
    _, stances = wholebody_court_positions(
        chunk['keypoints_px'], chunk['keypoint_scores'], homography, settings['foot_min_score'],
        allow_partial=settings.get('position_policy') == 'partial-feet-v3')
    margin = settings['court_margin_m']
    near_court = (np.isfinite(stances).all(axis=1) & (stances >= -margin).all(axis=1)
                  & (stances <= [court_width_m+margin, court_length_m+margin]).all(axis=1))
    result = np.full((len(chunk['frame_ids']), 2, 2), np.nan)
    for i, (first, stop) in enumerate(zip(chunk['pose_offsets'], chunk['pose_offsets'][1:])):
        candidates = np.flatnonzero(near_court[first:stop]) + first
        for slot in (0, 1):
            indices = candidates[(stances[candidates, 1] >= court_length_m/2) == bool(slot)]
            if len(indices) == 1:
                result[i, slot] = stances[indices[0]]
    return result


class _ReplayView:
    def __init__(self, width, height, homography, settings):
        self.width, self.height = width, height
        self.homography, self.settings = homography, settings
        self.sidebar, self.header = 360, 64
        self.shape = (height + self.header, width + self.sidebar, 3)
        self.margin = settings['court_margin_m']
        self.scale = min((self.sidebar - 48) / (court_width_m + 2*self.margin),
                         (self.shape[0] - 164) / (court_length_m + 2*self.margin))
        self.origin = np.array([width + (self.sidebar - court_width_m*self.scale)/2,
                                self.shape[0] - 50 - self.margin*self.scale])
        self.trails = [deque(maxlen=20), deque(maxlen=20)]
        self.streak = 0
        self.base = np.full(self.shape, (25, 23, 21), np.uint8)
        _text(self.base, 'COURT POSITIONS', (width+20, 25), size=.62)
        _text(self.base, 'Far end', (width+140, 82))
        for segment in court_model_segments:
            a, b = [self.court_pixel(point) for point in segment]
            cv2.line(self.base, a, b, (120, 140, 120), 1, cv2.LINE_AA)
        y = court_length_m/2
        cv2.line(self.base, self.court_pixel([0, y]), self.court_pixel([court_width_m, y]),
                 (220, 220, 220), 2, cv2.LINE_AA)
        for slot, name in enumerate(['Near player', 'Far player']):
            _text(self.base, name, (width+20+slot*165, self.shape[0]-14), PLAYER_COLOURS[slot])
        self.projected_lines = cv2.perspectiveTransform(court_model_segments, homography)

    def court_pixel(self, xy):
        return tuple(np.rint(self.origin + np.asarray(xy)*[self.scale, -self.scale]).astype(int))

    def render(self, image, frame_id, fps, court_score, points, eligible, accepted, shuttle,
               reconstructed=(False, False)):
        canvas = self.base.copy()
        camera = image.copy()
        court_ok = court_score >= self.settings['court_min_score']
        self.streak = self.streak + 1 if eligible else 0
        if not court_ok:
            state, detail = 'COURT CHECK REJECTED', 'Projected positions are provisional'
            colour = (90, 100, 255)
        elif not eligible:
            detail = ('Need a unique usable player in at least one half'
                      if self.settings.get('position_policy') == 'partial-feet-v3'
                      else 'Need one confident player in each half')
            state = 'PLAYER CHECK REJECTED'
            colour = (70, 170, 255)
        elif accepted:
            state, detail = 'ACCEPTED', 'Included in heatmaps'
            colour = (100, 225, 120)
        elif (not self.settings.get('motion_policy', '').startswith('offline-window-coverage-')
              and self.streak < self.settings['motion_window']):
            state = 'MOTION WINDOW WARMUP'
            detail = f"{self.streak}/{self.settings['motion_window']} consecutive eligible frames"
            colour = (70, 170, 255)
        else:
            state = 'MOTION CHECK REJECTED'
            detail = ('No confirmed movement in available valid observations'
                      if self.settings.get('motion_policy', '').startswith('offline-window-coverage-')
                      else 'Movement below the configured thresholds')
            colour = (70, 170, 255)

        # A faint fixed-court overlay makes calibration mismatches visible.
        overlay = camera.copy()
        for segment in self.projected_lines:
            a, b = np.rint(segment).astype(int)
            cv2.line(overlay, tuple(a), tuple(b), colour, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, .35, camera, .65, 0, camera)
        for slot, point in enumerate(points):
            if not np.isfinite(point).all():
                self.trails[slot].clear()
                continue
            player_colour = PLAYER_COLOURS[slot]
            xy = self.court_pixel(point)
            if court_ok:
                self.trails[slot].append(xy)
                if len(self.trails[slot]) > 1:
                    cv2.polylines(canvas, [np.asarray(self.trails[slot], np.int32)],
                                  False, player_colour, 1, cv2.LINE_AA)
                cv2.circle(canvas, xy, 7, player_colour, -1 if accepted else 2, cv2.LINE_AA)
            else:
                self.trails[slot].clear()
                cv2.drawMarker(canvas, xy, player_colour, cv2.MARKER_TILTED_CROSS, 13, 2)
            pixel = cv2.perspectiveTransform(np.asarray(point, float)[None, None], self.homography)[0, 0]
            pixel = tuple(np.rint(pixel).astype(int))
            if reconstructed[slot]:
                cv2.drawMarker(camera, pixel, player_colour, cv2.MARKER_DIAMOND, 15, 2)
                _text(camera, 'Jump fit', (pixel[0] + 9, pixel[1] - 7), player_colour)
            else:
                cv2.circle(camera, pixel, 6, player_colour, 2, cv2.LINE_AA)
        if np.isfinite(shuttle).all():
            cv2.circle(camera, tuple(np.rint(shuttle).astype(int)), 7, (220, 60, 240), 2, cv2.LINE_AA)
        canvas[self.header:, :self.width] = camera
        seconds = frame_id/fps
        _text(canvas, f'Source {int(seconds//60):02d}:{seconds%60:05.2f} | Frame {frame_id} | {state}',
              (14, 24), colour, .55)
        _text(canvas, detail, (14, 49))
        _text(canvas, f"Court support {court_score:.2f} / {self.settings['court_min_score']:.2f}",
              (self.width+20, 49), colour)
        return canvas


def render_position_replay(run_directory, analysis_directory=None, output=None, *,
                           encoder='libx264', max_frames=None, preview_frame=0):
    """Render every selected frame, including rejected observations and gaps.

    The source video supplies images/audio only; positions, scores and shuttle
    observations come from the binary cache. Rejected-frame positions are shown
    where recoverable, with explicit labels when calibration is unverified.
    """
    store = ObservationStore.open(run_directory)
    if analysis_directory is None:
        candidates = list((store.directory / 'derived').glob('*/settings.json'))
        if not candidates:
            raise FileNotFoundError('Generate the notebook analysis before rendering a replay.')
        analysis_directory = max(candidates, key=lambda path: path.stat().st_mtime).parent
    analysis_directory = Path(analysis_directory)
    settings = json.loads((analysis_directory / 'settings.json').read_text())
    if settings['raw_run_id'] != store.run_id:
        raise ValueError('The analysis belongs to a different raw run.')
    with np.load(analysis_directory / 'analysis.npz', allow_pickle=False) as data:
        ids, eligible, accepted = (data[key] for key in ['frame_ids', 'eligible', 'accepted'])
        jump_xyz = data['jump_xyz_m'] if 'jump_xyz_m' in data else None
        jump_probabilities = data['jump_probabilities'] if jump_xyz is not None else None
        jump_positions = data['jump_positions_m'] if 'jump_positions_m' in data else None
    expected = np.arange(store.manifest['first_frame_id'], store.manifest['stop_frame_id'])
    if not np.array_equal(ids, expected):
        raise ValueError('Analysis must cover the complete raw frame interval in order.')
    n = len(ids) if max_frames is None else min(max_frames, len(ids))
    if n < 1:
        raise ValueError('Need at least one frame to render.')
    video = Path(store.manifest['video']['path'])
    fps, width = settings['fps'], settings['processing_width']
    source_width, source_height = store.manifest['source_size_px']
    height = round(source_height*width/source_width)
    homography = np.asarray(settings['reference_H_court_to_image'], float)
    view = _ReplayView(width, height, homography, settings)
    output = Path(output) if output else analysis_directory / 'positions-replay.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + '.partial.mp4')
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(video)
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(ids[0]))
    encoding = _video_encoding(encoder)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{view.shape[1]}x{view.shape[0]}',
               '-r', f'{fps:.12f}', '-i', 'pipe:0',
               '-ss', f'{ids[0]/fps:.9f}', '-i', str(video),
               '-map', '0:v:0', '-map', '1:a:0?', *encoding, '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-b:a', '128k', '-t', f'{n/fps:.9f}',
               '-shortest', '-movflags', '+faststart', str(temporary)]
    started = time.monotonic()
    rendered = 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for chunk in store.iter_chunks():
                # Recover candidate positions even when the fixed court check failed.
                points = _candidate_positions(chunk, homography, settings)
                for offset, frame_id in enumerate(chunk['frame_ids']):
                    if rendered >= n:
                        break
                    if frame_id != ids[rendered]:
                        raise ValueError('Raw observation frame IDs do not match the analysis.')
                    ok, image = capture.read()
                    if not ok:
                        raise ValueError(f'Video decoding failed at source frame {frame_id}.')
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                    reconstructed = (False, False)
                    frame_points = points[offset]
                    if jump_xyz is not None:
                        court_valid = chunk['court_scores'][offset] >= settings['court_min_score']
                        if jump_positions is not None and court_valid:
                            frame_points = jump_positions[rendered]
                        elif jump_positions is None:
                            airborne = jump_probabilities[rendered] >= settings['jumps']['threshold']
                            frame_points[airborne] = jump_xyz[rendered, airborne, :2]
                        reconstructed = np.isfinite(jump_xyz[rendered]).all(axis=1)
                    frame = view.render(image, int(frame_id), fps, chunk['court_scores'][offset],
                                        frame_points, eligible[rendered], accepted[rendered],
                                        chunk['shuttle_pixels'][offset], reconstructed)
                    if rendered == preview_frame:
                        cv2.imwrite(str(output.with_suffix('.jpg')), frame)
                    process.stdin.write(frame.tobytes())
                    rendered += 1
                    if rendered % 600 == 0:
                        elapsed = time.monotonic()-started
                        print(f'Rendered {rendered:,}/{n:,} frames ({rendered/n:.1%}); '
                              f'{rendered/elapsed:.1f} frames/s', flush=True)
                if rendered >= n:
                    break
            if rendered != n:
                raise ValueError(f'Only rendered {rendered} of {n} expected frames.')
            process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace'))
            os.replace(temporary, output)
        except BaseException as error:
            process.kill()
            process.wait()
            temporary.unlink(missing_ok=True)
            if isinstance(error, BrokenPipeError):
                errors.seek(0)
                raise RuntimeError(errors.read().decode(errors='replace')) from error
            raise
        finally:
            capture.release()
    print(f'Saved {output} ({rendered:,} frames; {rendered/fps:.2f} s).', flush=True)
    return output


def split_position_replay(analysis_directory, source=None, *, encoder='libx264',
                          rejected_court_passed_only=False):
    """Partition a rendered replay using the saved per-frame acceptance mask.

    Preserve source IDs burned into the full replay, remove time gaps in each
    output, and cut audio at the same frame boundaries at sample precision.
    Set rejected_court_passed_only to export just rejected frames whose cached
    court support meets the analysis threshold. No model inference is run.
    """
    directory = Path(analysis_directory)
    settings = json.loads((directory / 'settings.json').read_text())
    with np.load(directory / 'analysis.npz', allow_pickle=False) as data:
        ids, accepted = data['frame_ids'], data['accepted'].astype(bool)
        court_scores = data['court_scores'] if 'court_scores' in data.files else None
    source = Path(source) if source else directory / 'positions-replay.mp4'
    if not source.is_file():
        raise FileNotFoundError('Render the complete positions replay before splitting it.')
    if accepted.shape != ids.shape or ids.ndim != 1:
        raise ValueError('Acceptance flags must match the source frame IDs.')
    if rejected_court_passed_only:
        if court_scores is None:
            store = ObservationStore.open(directory.parent.parent)
            cached = [(chunk['frame_ids'], chunk['court_scores']) for chunk in store.iter_chunks()]
            if not np.array_equal(ids, np.concatenate([frames for frames, _ in cached])):
                raise ValueError('Cached court scores do not match the analysis frame IDs.')
            court_scores = np.concatenate([scores for _, scores in cached])
        masks = [(~accepted) & (court_scores >= settings['court_min_score'])]
        names = ['rejected-court-passed']
    else:
        masks, names = [accepted, ~accepted], ['accepted', 'rejected']
    counts = [int(mask.sum()) for mask in masks]
    if not all(counts):
        raise ValueError('Each requested replay selection must contain at least one frame.')
    fps = Fraction(settings['fps']).limit_denominator(1000000)
    selections = []
    for mask in masks:
        boundaries = np.r_[0, np.flatnonzero(mask[1:] != mask[:-1])+1, len(mask)]
        selections.append([(int(first), int(stop)) for first, stop in zip(boundaries, boundaries[1:])
                           if mask[first]])
    condition = '+'.join(f'between(n,{first},{stop-1})' for first, stop in selections[0])
    timing = f'setpts=N*{fps.denominator}/({fps.numerator}*TB)'
    if len(masks) == 2:
        graph = (f"[0:v]select=outputs=2:expr='if({condition},1,2)'[keep][drop];"
                 f'[keep]{timing}[out0];[drop]{timing}[out1]')
    else:
        graph = f"[0:v]select=expr='{condition}',{timing}[out0]"
    encoding = _video_encoding(encoder)
    outputs = [directory / f'{name}-replay.mp4' for name in names]
    info = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,nb_frames',
        '-of', 'json', str(source)], text=True))
    video_stream = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
    if int(video_stream['nb_frames']) != len(ids):
        raise ValueError('The full replay frame count does not match this analysis.')
    has_audio = any(stream['codec_type'] == 'audio' for stream in info['streams'])
    with tempfile.TemporaryDirectory(prefix='replay-split-', dir=directory) as temporary:
        temporary = Path(temporary)
        videos = [temporary / f'{name}.mp4' for name in names]
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                   '-filter_complex', graph]
        for index, (output, count) in enumerate(zip(videos, counts)):
            command.extend(['-map', f'[out{index}]', '-an', *encoding, '-pix_fmt', 'yuv420p',
                            '-r', str(fps), '-fps_mode', 'cfr', '-frames:v', str(count), str(output)])
        print('Exporting cached frames: ' + ', '.join(f'{name}={count:,}'
              for name, count in zip(names, counts)), flush=True)
        subprocess.run(command, check=True)
        audio_paths = [temporary / f'{name}.wav' for name in names]
        if has_audio:
            print('Cutting matching audio at the saved frame boundaries.', flush=True)
            pcm_path = temporary / 'source.pcm'
            sample_rate, channels = 48000, 2
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                            '-i', str(source), '-map', '0:a:0', '-vn', '-f', 's16le',
                            '-ar', str(sample_rate), '-ac', str(channels), str(pcm_path)], check=True)
            pcm = np.memmap(pcm_path, dtype='<i2', mode='r').reshape(-1, channels)
            for audio_path, ranges in zip(audio_paths, selections):
                with wave.open(str(audio_path), 'wb') as stream:
                    stream.setnchannels(channels)
                    stream.setsampwidth(2)
                    stream.setframerate(sample_rate)
                    for first, stop in ranges:
                        start_sample = round(first*sample_rate/float(fps))
                        stop_sample = round(stop*sample_rate/float(fps))
                        stream.writeframesraw(pcm[start_sample:stop_sample].tobytes())
                        missing = stop_sample - max(start_sample, min(stop_sample, len(pcm)))
                        if missing > 0:
                            stream.writeframesraw(bytes(missing*channels*2))
            del pcm
        for index, (video, output, count) in enumerate(zip(videos, outputs, counts)):
            staged_output = temporary / output.name
            command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video)]
            if has_audio:
                command.extend(['-i', str(audio_paths[index]), '-map', '0:v:0', '-map', '1:a:0',
                                '-c:a', 'aac', '-b:a', '128k'])
            command.extend(['-c:v', 'copy', '-t', f'{count/float(fps):.9f}',
                            '-movflags', '+faststart', str(staged_output)])
            subprocess.run(command, check=True)
            os.replace(staged_output, output)
            mask = masks[index]
            extra = ({'court_scores': court_scores[mask],
                      'court_min_score': np.asarray(settings['court_min_score'])}
                     if rejected_court_passed_only else {})
            np.savez_compressed(output.with_suffix('.npz'), source_frame_ids=ids[mask],
                                output_frame_ids=np.arange(count), fps=np.asarray(float(fps)), **extra)
            print(f'Saved {output} ({count/float(fps):.2f} s).', flush=True)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--analysis-directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--encoder', choices=['libx264', 'h264_nvenc'], default='libx264')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--preview-frame', type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--split-only', action='store_true', help='Split an existing full replay by saved acceptance flags')
    mode.add_argument('--rejected-court-passed-only', action='store_true',
                      help='Export rejected frames that passed the cached court check')
    args = parser.parse_args()
    if args.split_only or args.rejected_court_passed_only:
        directory = args.analysis_directory
        if directory is None:
            directory = max((args.run_directory / 'derived').glob('*/settings.json'),
                            key=lambda path: path.stat().st_mtime).parent
        split_position_replay(directory, source=args.output, encoder=args.encoder,
                              rejected_court_passed_only=args.rejected_court_passed_only)
    else:
        render_position_replay(args.run_directory, args.analysis_directory, args.output,
                               encoder=args.encoder, max_frames=args.max_frames,
                               preview_frame=args.preview_frame)
