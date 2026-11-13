from pathlib import Path
import json
import cv2
import numpy as np
import torch
import onnxruntime as ort
from rtmlib import RTMPose,draw_skeleton
from badminton_analysis.observation_store import ObservationStore
from badminton_analysis.monotrack.tracking import require_cuda_device

root=Path('/home/aesthetic/code/badminton-analysis')
store=ObservationStore.open(root/'data/player-position-heatmap/f7d755110dc74ca7f882b074')
torch.set_num_threads(4)
device=require_cuda_device('cuda:0')
ort.preload_dlls()
pose=RTMPose(str(root/'.cache/rtmpose/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728.onnx'),
             model_input_size=(288,384),to_openpose=False,backend='onnxruntime',device=str(device))
assert pose.session.get_providers()[0]=='CUDAExecutionProvider'
pose.session.disable_fallback()

def colour(value):
    return (70,225,95) if value>=.6 else ((0,190,255) if value>=.4 else (70,75,255))
def text(img,message,xy,c=(235,235,235),size=.55):
    cv2.putText(img,message,xy,cv2.FONT_HERSHEY_SIMPLEX,size,c,1,cv2.LINE_AA)

bounds=store.saved_chunks()
capture=cv2.VideoCapture(store.manifest['video']['path'])
try:
 for frame_id in [13699]:
    for first,stop in bounds:
        if first<=frame_id<stop:
            data=store.load_chunk(first,stop);i=frame_id-first
            a,b=data['pose_offsets'][i:i+2]
            boxes=data['boxes_xyxy'][a:b];box_scores=data['box_scores'][a:b]
            saved_points=data['keypoints_px'][a:b];saved_scores=data['keypoint_scores'][a:b]
            break
    else:raise RuntimeError('No cached detection boxes')
    capture.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
    ok,source=capture.read()
    if not ok:raise RuntimeError(f'Cannot decode source frame {frame_id}')
    width=store.manifest['processing_width']
    image=cv2.resize(source,(width,round(source.shape[0]*width/source.shape[1])),interpolation=cv2.INTER_AREA)
    points,scores=pose(image,bboxes=boxes)
    output=store.directory/'inspections'/f'frame-{frame_id}'
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'rtmpose-raw.npz',frame_id=np.asarray(frame_id),
        timestamp_seconds=np.asarray(frame_id/store.manifest['fps']),image_size_px=np.asarray(image.shape[:2][::-1]),
        boxes_xyxy=boxes,box_scores=box_scores,keypoints_px=points,keypoint_scores=scores,
        saved_keypoints_px=saved_points,saved_keypoint_scores=saved_scores,device=np.asarray(str(device)))
    cv2.imwrite(str(output/'source-frame.png'),source)
    cv2.imwrite(str(output/'processing-frame.png'),image)
    targets=sorted(np.argsort(-box_scores)[:2],key=lambda i:boxes[i,3],reverse=True)
    if len(targets)!=2:raise RuntimeError('Need two detected players for this inspection')
    canvas=np.full((960,1440,3),(24,22,20),np.uint8)
    seconds=frame_id/store.manifest['fps']
    text(canvas,f'RTMPose WholeBody | Source frame {frame_id} | {int(seconds//60):02d}:{seconds%60:05.2f} | CUDA',(16,25),size=.65)
    text(canvas,'Cached person boxes and 960px input. Skeleton shows scores >= 0.4; all six foot predictions are marked.',(16,49),size=.48)
    overlay=draw_skeleton(image.copy(),points[targets],scores[targets],kpt_thr=.4,radius=2,line_width=1)
    for slot,person in enumerate(targets):
        a,b=np.rint(boxes[person].reshape(2,2)).astype(int)
        cv2.rectangle(overlay,tuple(a),tuple(b),(180,180,180),1)
        text(overlay,f'{["Near","Far"][slot]} / P{person}',tuple(a+[0,-8]),size=.48)
        for index in range(17,23):
            xy=tuple(np.rint(points[person,index]).astype(int))
            cv2.circle(overlay,xy,4,colour(scores[person,index]),2,cv2.LINE_AA)
    canvas[60:600,:960]=overlay
    text(canvas,'FOOT LANDMARK SCORES',(980,92),size=.65)
    for label,x in [('Landmark',980),('GPU',1210),('Cached',1321)]:text(canvas,label,(x,119),size=.48)
    names=['L big toe','L small toe','L heel','R big toe','R small toe','R heel']
    for slot,(person,start) in enumerate(zip(targets,[153,350])):
        text(canvas,f'{["Near","Far"][slot]} player / P{person}',(980,start),size=.58)
        for offset,index in enumerate(range(17,23)):
            y=start+26+offset*22
            text(canvas,f'{index}: {names[offset]}',(980,y),size=.51)
            text(canvas,f'{scores[person,index]:.3f}',(1210,y),colour(scores[person,index]),.53)
            text(canvas,f'{saved_scores[person,index]:.3f}',(1321,y),colour(saved_scores[person,index]),.53)
    text(canvas,'Green >= 0.6    Amber 0.4-0.6',(980,553),size=.48)
    text(canvas,'Red < 0.4 (rejected by foot filter)',(980,577),(70,75,255),.48)
    source_scale=source.shape[1]/image.shape[1]
    for slot,(person,x0) in enumerate(zip(targets,[0,720])):
        p=points[person]
        lower=p[[13,14,15,16,17,18,19,20,21,22]]
        low=np.floor(lower.min(axis=0)-[18,12]).astype(int)
        high=np.ceil(lower.max(axis=0)+[18,18]).astype(int)
        low=np.maximum(low,[0,0]);high=np.minimum(high,[image.shape[1],image.shape[0]])
        aa=np.rint(low*source_scale).astype(int);bb=np.rint(high*source_scale).astype(int)
        crop=source[aa[1]:bb[1],aa[0]:bb[0]]
        scale=min(684/crop.shape[1],277/crop.shape[0])
        shape=(round(crop.shape[1]*scale),round(crop.shape[0]*scale))
        zoom=cv2.resize(crop,shape,interpolation=cv2.INTER_CUBIC)
        px=x0+(720-shape[0])//2;py=646
        canvas[py:py+shape[1],px:px+shape[0]]=zoom
        text(canvas,f'{["Near","Far"][slot]} player: knees, ankles and feet',(x0+18,626),size=.57)
        coords=(p*source_scale-aa)*scale+[px,py]
        for u,v in [(13,15),(14,16),(15,17),(15,18),(15,19),(16,20),(16,21),(16,22)]:
            cv2.line(canvas,tuple(np.rint(coords[u]).astype(int)),tuple(np.rint(coords[v]).astype(int)),(160,160,160),1,cv2.LINE_AA)
        for index in [13,14,15,16,17,18,19,20,21,22]:
            xy=tuple(np.rint(coords[index]).astype(int))
            cv2.circle(canvas,xy,5,colour(scores[person,index]),2,cv2.LINE_AA)
            shift=([8,-8] if index in [13,15,17,19,22] else [-31,15])
            text(canvas,str(index),tuple(np.asarray(xy)+shift),colour(scores[person,index]),.5)
    text(canvas,'13/14 knees | 15/16 ankles | 17/18 left toes, 19 left heel | 20/21 right toes, 22 right heel',(18,947),size=.52)
    cv2.imwrite(str(output/'rtmpose-result.png'),canvas)
    print(json.dumps({'frame':frame_id,'timestamp_seconds':seconds,'player_indices':list(map(int,targets)),
        'gpu_foot_scores':scores[targets,17:23].round(4).tolist(),
        'cached_foot_scores':saved_scores[targets,17:23].round(4).tolist(),
        'result':str(output/'rtmpose-result.png')}),flush=True)
finally:
 capture.release()