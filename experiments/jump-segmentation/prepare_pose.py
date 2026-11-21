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

from badminton_analysis.court_player_boxes import select_court_players
from badminton_analysis.jumps import LANDMARKS
from badminton_analysis.positions import detect_court
from badminton_analysis.prepare_rgb_player_crops import SOURCE_SIZE, _detect
from badminton_analysis.rgb_mgmae import CENTER, FRAMES, resize_frame
from badminton_analysis.train_rgb_mgmae import _decode, _labels


def extract(video, frames, detector, estimator, output, *, practice=False, reference_frame=None,
            labels=None):
    capture = cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,3])
    try:
        if not capture.isOpened():
            raise FileNotFoundError(video)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'No valid frame rate for {video}.')
        court = None
        if reference_frame is not None:
            reference = np.empty((1,height,width,3),dtype=np.uint8)
            _decode(capture,[reference_frame],reference,lambda frame: frame)
            size = (960,round(height*960/width))
            reduced = cv2.resize(reference[0],size,interpolation=cv2.INTER_AREA)
            court = np.diag([width/size[0],height/size[1],1.]) @ detect_court(reduced)['H_court_to_image']
        if labels is not None:
            reviewed = _labels(labels,count)
            centers = np.unique([index for partition in reviewed.values() for index in partition])
            if not len(centers):
                raise ValueError(f'No reviewed frames in {labels}.')
            frames = np.unique(centers[:,None]+np.arange(FRAMES)-CENTER)
        elif frames is None:
            frames = np.arange(count)
        images = np.empty((len(frames),height,width,3),dtype=np.uint8)
        _decode(capture,frames,images,lambda frame: frame)
    finally:
        capture.release()

    if practice:
        sampled, detections = practice_boxes(detector,images)
    elif court is not None:
        sampled, detections = calibrated_boxes(detector,images,frames,court)
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
    np.savez_compressed(output,frames=frames,points=points,scores=scores,boxes=boxes,fps=fps)
    print(f'{output.stem}: {np.isfinite(boxes).all(axis=2).sum()} poses in {len(frames)} frames',flush=True)


def calibrated_boxes(detector,images,frames,court):
    sampled = np.arange(0,len(images),2)
    detections = []
    previous = np.full((2,4),np.nan)
    last = np.full(2,frames[0]-100)
    for batch in np.array_split(sampled,int(np.ceil(len(sampled)/8))):
        predictions = detector.predict([images[i] for i in batch],imgsz=960,classes=[0],
                                       conf=.18,device='cpu',verbose=False)
        for i,result in zip(batch,predictions):
            previous[frames[i]-last>6] = np.nan
            boxes = select_court_players(result.boxes.xyxy.numpy(),result.boxes.conf.numpy(),court,previous)
            detections.append(boxes)
            observed = np.isfinite(boxes).all(axis=1)
            previous[observed] = boxes[observed]
            last[observed] = frames[i]
    return sampled,np.array(detections)


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
    parser.add_argument('rgb_caches',type=Path,nargs='*')
    parser.add_argument('--sources',type=Path,action='append',default=[],
                        help='Prepare reviewed pose windows from a video and label manifest.')
    parser.add_argument('--detector',type=Path,default=Path('yolo26m.pt'))
    parser.add_argument('--pose-model',type=Path,required=True)
    parser.add_argument('--practice-directory',type=Path)
    parser.add_argument('--court-reference-frames',type=Path,
                        help='JSON mapping a fixed-camera source name to its court calibration frame.')
    args = parser.parse_args()
    if not (args.rgb_caches or args.sources or args.practice_directory):
        parser.error('Supply an RGB cache, a source manifest, or practice clips.')
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
    references = json.loads(args.court_reference_frames.read_text()) if args.court_reference_frames else {}
    for directory in args.rgb_caches:
        for name in json.loads((directory/'sources.json').read_text()):
            output = args.output/f'{name}.npz'
            if output.exists():
                raise FileExistsError(output)
            metadata = json.loads((directory/f'{name}.json').read_text())
            with np.load(directory/f'{name}.npz',allow_pickle=False) as data:
                frames = np.unique(data['centers'][:,None]+np.arange(FRAMES)-CENTER)
            extract(metadata['video']['path'],frames,detector,estimator,output,
                    reference_frame=references.get(name))
    for manifest in args.sources:
        for item in json.loads(manifest.read_text())['sources']:
            name = item['name']
            if not name or Path(name).name!=name or name in ('.','..'):
                raise ValueError('Use a file-safe source name.')
            output = args.output/f'{name}.npz'
            if output.exists():
                raise FileExistsError(output)
            extract(manifest.parent/item['video'],None,detector,estimator,output,
                    reference_frame=references.get(name),labels=manifest.parent/item['labels'])
    if args.practice_directory:
        for video in sorted(args.practice_directory.glob('defence-*.mp4')):
            output = args.output/f'mendeley-{video.stem}.npz'
            if output.exists():
                raise FileExistsError(output)
            extract(video,None,detector,estimator,output,practice=True)


if __name__ == '__main__':
    main()
