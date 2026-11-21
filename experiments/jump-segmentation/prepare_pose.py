"""Extract WholeBody pose windows from reviewed clips for the transfer experiment."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from rtmlib import RTMPose
import torch
from ultralytics import YOLO

from badminton_analysis.jumps import LANDMARKS
from badminton_analysis.prepare_rgb_player_crops import SOURCE_SIZE, _detect
from badminton_analysis.rgb_mgmae import CENTER, FRAMES, resize_frame
from badminton_analysis.train_rgb_mgmae import _decode


def extract(video, frames, detector, estimator, output, *, practice=False):
    capture = cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,3])
    try:
        if not capture.isOpened():
            raise FileNotFoundError(video)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frames is None:
            frames = np.arange(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        images = np.empty((len(frames),height,width,3),dtype=np.uint8)
        _decode(capture,frames,images,lambda frame: frame)
    finally:
        capture.release()

    if practice:
        sampled, detections = practice_boxes(detector,images)
    else:
        resized = np.array([resize_frame(frame,SOURCE_SIZE) for frame in images])
        sampled,detections = _detect(detector,resized)
        w,h = SOURCE_SIZE
        scale = min(w/width,h/height)
        left,top = (w-round(width*scale))//2,(h-round(height*scale))//2
        detections = (detections-np.array([left,top,left,top]))/scale
    points = np.full((len(frames),2,len(LANDMARKS),2),np.nan,dtype=np.float32)
    scores = np.zeros(points.shape[:-1],dtype=np.float32)
    boxes = np.full((len(frames),2,4),np.nan,dtype=np.float32)
    detected_frames = frames[sampled]
    for i,(frame,image) in enumerate(zip(frames,images)):
        nearby = np.flatnonzero(abs(detected_frames-frame)<=4)
        for slot in range(2):
            usable = nearby[np.isfinite(detections[nearby,slot]).all(axis=1)]
            if len(usable):
                closest = usable[np.argmin(abs(detected_frames[usable]-frame))]
                boxes[i,slot] = detections[closest,slot]
        slots = np.flatnonzero(np.isfinite(boxes[i]).all(axis=1))
        if len(slots):
            keypoints,confidence = estimator(image,boxes[i,slots])
            points[i,slots] = keypoints[:,LANDMARKS]
            scores[i,slots] = confidence[:,LANDMARKS]
    np.savez_compressed(output,frames=frames,points=points,scores=scores,boxes=boxes)
    print(f'{output.stem}: {np.isfinite(boxes).all(axis=2).sum()} poses in {len(frames)} frames',flush=True)


def practice_boxes(detector,images):
    """Select the foreground player in the labeled portrait practice clips."""
    height,width = images.shape[1:3]
    sampled = np.arange(0,len(images),3)
    predictions = detector.predict([images[i] for i in sampled],imgsz=704,classes=[0],
                                   device='cpu',verbose=False)
    detections = np.full((len(sampled),2,4),np.nan)
    for i,prediction in enumerate(predictions):
        box = prediction.boxes.xyxy.numpy()
        good = ((box[:,3] > .56*height) & (box[:,3] < .89*height)
                & (box[:,0] > .03*width) & (box[:,2] < .97*width))
        eligible = box[good]
        if len(eligible):
            detections[i,0] = eligible[np.argmax(eligible[:,3]-eligible[:,1])]
    return sampled,detections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('rgb_caches',type=Path,nargs='+')
    parser.add_argument('--detector',type=Path,default=Path('yolo26m.pt'))
    parser.add_argument('--pose-model',type=Path,required=True)
    parser.add_argument('--practice-directory',type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    detector = YOLO(args.detector)
    estimator = RTMPose(str(args.pose_model),model_input_size=(288,384))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    estimator.session = ort.InferenceSession(args.pose_model,sess_options=options,
                                             providers=['CPUExecutionProvider'])
    for directory in args.rgb_caches:
        for name in json.loads((directory/'sources.json').read_text()):
            output = args.output/f'{name}.npz'
            if output.exists():
                raise FileExistsError(output)
            metadata = json.loads((directory/f'{name}.json').read_text())
            with np.load(directory/f'{name}.npz',allow_pickle=False) as data:
                frames = np.unique(data['centers'][:,None]+np.arange(FRAMES)-CENTER)
            extract(metadata['video']['path'],frames,detector,estimator,output)
    if args.practice_directory:
        for video in sorted(args.practice_directory.glob('defence-*.mp4')):
            output = args.output/f'mendeley-{video.stem}.npz'
            if output.exists():
                raise FileExistsError(output)
            extract(video,None,detector,estimator,output,practice=True)


if __name__ == '__main__':
    main()
