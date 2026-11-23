"""Build single-player RGB clips from broadcast frames and person detections."""

import argparse
import json
import tempfile
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.observation_store import file_fingerprint
from badminton_analysis.rgb_mgmae import CENTER, FRAMES, PATCH, flow_masks, resize_frame
from badminton_analysis.train_rgb_mgmae import PLAYERS, UNREVIEWED, _decode


SOURCE_SIZE = (640, 384)
CROP_SIZE = 160


def _player_boxes(boxes, scores):
    """Find the near/far pair in the central court region of a singles broadcast."""
    width,height = SOURCE_SIZE
    center_x = (boxes[:,0]+boxes[:,2])/2
    foot_y = boxes[:,3]
    heights = boxes[:,3]-boxes[:,1]
    inside = (center_x>=.19*width)&(center_x<=.81*width)&(foot_y>=.31*height)&(foot_y<=.96*height)
    inside &= (heights>=.075*height)&(heights<=.48*height)
    ranked = []
    for a,b in combinations(np.flatnonzero(inside),2):
        near,far = (a,b) if foot_y[a]>foot_y[b] else (b,a)
        separation = (foot_y[near]-foot_y[far])/height
        if separation<.08 or foot_y[near]<.50*height or foot_y[far]>.66*height:
            continue
        if heights[near]<.87*heights[far]:
            continue
        score = scores[near]+scores[far]+min(separation,.4)+.15*min(heights[near]/heights[far]-1,1)
        ranked.append((score,near,far))
    if not ranked:
        return np.full((2,4),np.nan)
    _,near,far = max(ranked)
    return boxes[[near,far]]


def crop_player_window(rgb,box):
    """Use one square for all 13 frames, keeping margin below the bounding box."""
    left,top,right,bottom = box
    height = bottom-top
    size = max(1.,height*1.5,(right-left)*1.35)
    x = (left+right-size)/2
    y = (top+bottom-size)/2+.10*height
    scale = CROP_SIZE/size
    transform = np.array([[scale,0,-x*scale],[0,scale,-y*scale]])
    return np.stack([cv2.warpAffine(frame,transform,(CROP_SIZE,CROP_SIZE),flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE) for frame in rgb])


def _detect(detector,pixels):
    sample = np.arange(0,len(pixels),4)
    detections = []
    for batch in np.array_split(sample,int(np.ceil(len(sample)/12))):
        images = [cv2.cvtColor(pixels[i],cv2.COLOR_RGB2BGR) for i in batch]
        predictions = detector.predict(images,imgsz=640,classes=[0],conf=.23,device='cpu',verbose=False)
        detections.extend(_player_boxes(result.boxes.xyxy.numpy(),result.boxes.conf.numpy())
                          for result in predictions)
    return sample,np.array(detections)


def _write_player(directory,name,video_name,data,pixels,boxes,detected,slot,flow):
    train, validation = (data[f'{key}_index'][data[f'{key}_labels'][:,slot] != UNREVIEWED]
                         for key in ('train','validation'))
    indices = np.unique(np.r_[train,validation,data['pretrain']])
    if not len(indices):
        return False
    mapping = np.arange(len(indices)*FRAMES).reshape(-1,FRAMES)
    video = np.lib.format.open_memmap(directory/f'{name}.npy',mode='w+',dtype=np.uint8,
                                      shape=(len(indices)*FRAMES,CROP_SIZE,CROP_SIZE,3))
    available = np.zeros(len(indices),bool)
    for row,index in enumerate(indices):
        middle = data['mapping'][index,CENTER]
        close = np.abs(detected-middle)<=4
        candidate_boxes = boxes[close,slot]
        candidate_boxes = candidate_boxes[np.isfinite(candidate_boxes).all(axis=1)]
        if len(candidate_boxes):
            video[mapping[row]] = crop_player_window(pixels[data['mapping'][index]],np.median(candidate_boxes,axis=0))
            available[row] = True
    video.flush()
    pretrain = np.searchsorted(indices,data['pretrain'])
    pretrain = pretrain[available[pretrain]]
    rng = np.random.default_rng(7+slot)
    grid = CROP_SIZE//PATCH
    masks = np.empty((len(pretrain),2,FRAMES,grid,grid),dtype=bool)
    for i,row in enumerate(pretrain):
        masks[i] = flow_masks(video[mapping[row]],rng,flow)
    arrays = {}
    for key in ('train','validation'):
        select = data[f'{key}_labels'][:,slot] != UNREVIEWED
        arrays[f'{key}_index'] = np.searchsorted(indices,data[f'{key}_index'][select])
        arrays[f'{key}_labels'] = data[f'{key}_labels'][select,slot,None]
    np.savez_compressed(directory/f'{name}.npz',mapping=mapping,centers=data['centers'][indices],
                        pretrain=pretrain,masks=masks,available=available,
                        fps=data.get('fps',30.),video_name=video_name,**arrays)
    print(f'{name}: cropped {available.sum()}/{len(available)} windows.',flush=True)
    return True


def prepare(source,output,weights):
    from ultralytics import YOLO
    import torch

    cv2.setNumThreads(2)
    torch.set_num_threads(5)
    output.mkdir(parents=True,exist_ok=True)
    detector = YOLO(weights)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    names = []
    for name in json.loads((source/'sources.json').read_text()):
        signature = {'input':file_fingerprint(source/f'{name}.npz',content=True),
                     'source':file_fingerprint(source/f'{name}.json',content=True),
                     'detector':file_fingerprint(weights,content=True),
                   'source_size':list(SOURCE_SIZE),'crop_size':CROP_SIZE}
        marker = output/f'{name}-crop-source.json'
        if marker.is_file():
            previous = json.loads(marker.read_text())
            if previous['signature']!=signature:
                raise ValueError(f'Prepare {name} in a new crop directory.')
            names.extend(previous['players'])
            continue
        metadata = json.loads((source/f'{name}.json').read_text())
        with np.load(source/f'{name}.npz',allow_pickle=False) as file:
            data = dict(file)
        frames = np.unique(data['centers'][:,None]+np.arange(FRAMES)-CENTER)
        with tempfile.TemporaryDirectory(dir=output) as temp:
            original = np.memmap(Path(temp)/'rgb',mode='w+',dtype=np.uint8,
                                 shape=(len(frames),SOURCE_SIZE[1],SOURCE_SIZE[0],3))
            video = cv2.VideoCapture(metadata['video']['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,3])
            try:
                if not video.isOpened():
                    raise FileNotFoundError(metadata['video']['path'])
                _decode(video,frames,original,lambda frame:resize_frame(frame,SOURCE_SIZE))
            finally:
                video.release()
            positions,boxes = _detect(detector,original)
            players = []
            for slot,player in enumerate(PLAYERS):
                key = f'{name}-{player}'
                if _write_player(output,key,name,data,original,boxes,positions,slot,flow):
                    players.append(key)
        marker.write_text(json.dumps({'signature':signature,'players':players},indent=2)+'\n')
        names.extend(players)
    (output/'sources.json').write_text(json.dumps(names)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rgb_cache',type=Path)
    parser.add_argument('player_cache',type=Path)
    parser.add_argument('--weights',type=Path,default=Path('yolo26m.pt'))
    args = parser.parse_args()
    prepare(args.rgb_cache,args.player_cache,args.weights)


if __name__ == '__main__':
    main()
