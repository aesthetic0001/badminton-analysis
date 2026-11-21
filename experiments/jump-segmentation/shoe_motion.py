"""Track shoe pixels in fixed-camera footage for contact experiments."""

import argparse
from pathlib import Path

import cv2
import numpy as np

from badminton_analysis.train_rgb_mgmae import _decode


OFFSETS = (-2,-1,1,2)


def foot_corners(frame,points,confidence,height):
    shoe = points[(confidence>=.35)&np.isfinite(points).all(axis=1)]
    if len(shoe)<2:
        return np.empty((0,2),np.float32)
    x,y = shoe.mean(axis=0)
    radius = np.clip(.043*height,3,10)
    left,right = max(0,int(x-radius)),min(frame.shape[1],int(x+radius)+1)
    top,bottom = max(0,int(y-radius)),min(frame.shape[0],int(y+radius)+1)
    crop = frame[top:bottom,left:right]
    if min(crop.shape)<4:
        return np.empty((0,2),np.float32)
    corners = cv2.goodFeaturesToTrack(crop,maxCorners=8,qualityLevel=.025,minDistance=2,
                                     blockSize=3)
    if corners is None:
        return np.empty((0,2),np.float32)
    return corners.reshape(-1,2)+np.float32([left,top])


def measure_flow(image,adjacent,points):
    """Reject a track when the reverse optical flow misses its starting point."""
    if len(points)==0:
        return points,np.empty(0,bool)
    kwargs = {'winSize':(13,13),'maxLevel':2}
    start = points.reshape(-1,1,2)
    forward,status,_ = cv2.calcOpticalFlowPyrLK(image,adjacent,start,None,**kwargs)
    backward,reversed_status,_ = cv2.calcOpticalFlowPyrLK(adjacent,image,forward,None,**kwargs)
    error = np.linalg.norm((backward-start).reshape(-1,2),axis=1)
    valid = (status[:,0]!=0)&(reversed_status[:,0]!=0)&(error<1.25)
    return (forward-start).reshape(-1,2),valid


def extract(cache,video,output):
    cv2.setNumThreads(2)
    with np.load(cache,allow_pickle=False) as data:
        frames,points,confidence,boxes = (data[key] for key in ('frames','points','scores','boxes'))
    capture = cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,3])
    try:
        if not capture.isOpened():
            raise FileNotFoundError(video)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        images = np.empty((len(frames),height,width),np.uint8)
        _decode(capture,frames,images,lambda bgr: cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()

    motion = np.full((len(frames),2,2,len(OFFSETS),2),np.nan,np.float32)
    count = np.zeros(motion.shape[:-1],np.uint8)
    lookup = dict(zip(frames,range(len(frames))))
    for i,frame in enumerate(frames):
        corners,keys = [],[]
        for player,box in enumerate(boxes[i]):
            tall = box[3]-box[1]
            if not np.isfinite(tall) or tall<=0:
                continue
            for foot in (0,1):
                joints = slice(-6+3*foot, -3 if foot==0 else None)
                pixels = foot_corners(images[i],points[i,player,joints],confidence[i,player,joints],tall)
                if len(pixels):
                    corners.append(pixels)
                    keys.append((player,foot,tall))
        if not corners:
            continue
        sizes = np.cumsum([len(values) for values in corners])
        pixels = np.concatenate(corners)
        for column,distance in enumerate(OFFSETS):
            other = lookup.get(frame+distance)
            if other is None:
                continue
            delta,valid = measure_flow(images[i],images[other],pixels)
            start = 0
            for (player,foot,tall),stop in zip(keys,sizes):
                tracks = delta[start:stop][valid[start:stop]]
                start = stop
                if len(tracks)>=2:
                    motion[i,player,foot,column] = np.median(tracks,axis=0)/(tall*distance)
                    count[i,player,foot,column] = len(tracks)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,frames=frames,motion=motion,tracks=count)
    print(f'{output.name}: {(count>0).sum()} of {count.size} flow estimates',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pose_cache',type=Path)
    parser.add_argument('video',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    extract(args.pose_cache,args.video,args.output)


if __name__=='__main__':
    main()
