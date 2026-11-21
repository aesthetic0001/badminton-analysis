"""Prepare broadcast clips and train an RGB jump detector with optional MGMAE pretraining."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from badminton_analysis.jumps import jump_segments
from badminton_analysis.jump_benchmark import evaluate_jump_scores
from badminton_analysis.observation_store import file_fingerprint
from badminton_analysis.rgb_mgmae import (
    CENTER, FRAMES, GRID, HEIGHT, WIDTH, RGBJumpSegmenter, flow_masks, patch_rgb_targets, resize_frame,
)


STATES = {'grounded': 0, 'no_correction': 0, 'airborne': 1, 'uncertain': -1}
PLAYERS = ('near', 'far')
UNREVIEWED = -2


def _labels(path, frame_count):
    labels = {split: {} for split in ('train','validation')}
    used = set()
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            player = PLAYERS.index(row['player'])
            start, stop = int(row['start_frame']),int(row['stop_frame'])
            if not CENTER <= start < stop <= frame_count-CENTER:
                raise ValueError(f'Label interval lacks a full RGB window: {row}')
            target = STATES[row['state']]
            frames = labels[row['split']]
            for frame in range(start,stop):
                if (frame,player) in used:
                    raise ValueError(f'Overlapping labels at frame {frame}.')
                used.add((frame,player))
                frames.setdefault(frame,[UNREVIEWED,UNREVIEWED])[player] = target
    a,b = (np.array(sorted(labels[key]),dtype=int) for key in ('train','validation'))
    if len(a) and len(b):
        nearest = np.searchsorted(b,a)
        before = b[(nearest-1).clip(0,len(b)-1)]
        after = b[nearest.clip(0,len(b)-1)]
        if np.any(np.minimum(abs(a-before),abs(a-after)) <= 2*CENTER):
            raise ValueError('Training and validation windows share source frames.')
    return labels


def _pretrain_centers(path, count, fps, labels, seed, *, on_court=False, groups=24):
    """Choose full-window RGB samples from rally footage, away from validation clips."""
    if not labels['train'] and labels['validation']:
        raise ValueError('Keep validation-only sources out of pretraining.')
    offsets = np.arange(8)*3
    end = offsets[-1] + 2*CENTER + 1
    if on_court:
        with np.load(path,allow_pickle=False) as data:
            indices = data['frame_ids']
            accepted = data['court_scores'] >= .45
        if not np.all(np.diff(indices)==1):
            raise ValueError('Court-valid input must cover contiguous frames.')
        intervals = np.flatnonzero(np.diff(np.r_[False,accepted,False])).reshape(-1,2)+indices[0]
    else:
        rallies = json.loads(path.read_text())
        intervals = [(r['start_frame'],r['end_frame']+1) for r in rallies]
    candidates = {}
    for start, stop in intervals:
        start, stop = max(0,start), min(count,stop)
        if stop-start < FRAMES:
            continue
        if stop-start < end:
            candidates.setdefault(start,offsets[offsets+FRAMES <= stop-start])
        else:
            for group_start in range(start,stop-end+1,end):
                candidates[group_start] = offsets
    starts = np.array(sorted(candidates),dtype=int)
    if labels['validation']:
        val = np.array(sorted(labels['validation']))
        pad = round(30*fps)
        insertion = np.searchsorted(val,starts-pad)
        safe = (insertion>=len(val)) | (val[insertion.clip(0,len(val)-1)]>starts+end+pad)
        starts = starts[safe]
    if not len(starts):
        raise ValueError('No pretraining windows remain outside validation.')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(starts,size=min(groups,len(starts)),replace=False)
    return np.unique(np.concatenate([start+candidates[start]+CENTER for start in chosen]))


def _decode(capture, indices, destination, resize=resize_frame):
    # Seek between nearby clips and decode overlapping windows only once.
    frame = None
    for i, frame_id in enumerate(indices):
        frame_id = int(frame_id)
        if frame is None or frame_id-frame>18:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES,frame_id):
                raise ValueError(f'Cannot seek to source frame {frame_id}.')
            frame = frame_id-1
        while frame+1 < frame_id:
            if not capture.grab():
                raise ValueError(f'Cannot advance to source frame {frame_id}.')
            frame += 1
        ok,bgr = capture.read()
        if not ok:
            raise ValueError(f'Cannot decode source frame {frame_id}.')
        destination[i] = resize(bgr)
        frame = frame_id


def prepare(config_path, directory):
    config_path,directory = Path(config_path).resolve(),Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    seed = 7
    cv2.setNumThreads(2)
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    prepared = []
    seen_videos = set()
    for item in json.loads(config_path.read_text())['sources']:
        name = item['name']
        if not name or Path(name).name != name or name in ('.','..') or name in prepared:
            raise ValueError('Use a unique file-safe name for each source.')
        video = (config_path.parent / item['video']).resolve()
        annotations = (config_path.parent / item['labels']).resolve() if 'labels' in item else None
        if video in seen_videos:
            raise ValueError('Give each source video once to keep split checks effective.')
        seen_videos.add(video)
        upstream = (config_path.parent / item['pretraining']).resolve() if 'pretraining' in item else None
        groups = int(item.get('pretraining_groups', 24))
        if groups < 1 or (annotations is None and upstream is None):
            raise ValueError('Provide labels or pretraining footage and a positive group count.')
        signature = {'video':file_fingerprint(video), 'labels':file_fingerprint(annotations,content=True) if annotations else None,
                     'pretraining':file_fingerprint(upstream) if upstream else None,
                     'pretraining_kind':item.get('pretraining_kind'),'shape':[FRAMES,HEIGHT,WIDTH],
                     'patch_size':int(WIDTH/GRID[1]),'sampling_seed':seed,
                     'unreviewed_label':UNREVIEWED}
        if groups != 24:
            signature['pretraining_groups'] = groups
        marker = directory/f'{name}.json'
        if marker.exists():
            if json.loads(marker.read_text()) != signature:
                raise ValueError(f'Cached source {name} has different inputs. Use another cache directory.')
            if all((directory/f'{name}.{ext}').is_file() for ext in ('npy','npz')):
                print(f'Reusing {name}.',flush=True)
                prepared.append(name)
                continue
        capture = cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,3])
        if not capture.isOpened():
            raise FileNotFoundError(video)
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not 29.5 <= fps <= 30.5:
                raise ValueError(f'This pilot expects approximately 30 fps, got {fps} for {video}.')
            labels = _labels(annotations,count) if annotations else {'train': {}, 'validation': {}}
            pretrain = (_pretrain_centers(upstream,count,fps,labels,seed,
                                          on_court=item.get('pretraining_kind')=='court_scores',
                                          groups=groups)
                        if upstream else np.empty(0,dtype=int))
            centers = np.unique(np.r_[pretrain,*[list(part) for part in labels.values()]]).astype(int)
            frames = centers[:,None]+np.arange(FRAMES)-CENTER
            indices = np.unique(frames)
            mapping = np.searchsorted(indices,frames)
            pretrain_index = np.searchsorted(centers,pretrain)
            pixels = np.lib.format.open_memmap(directory/f'{name}.npy',mode='w+',dtype=np.uint8,
                                               shape=(len(indices),HEIGHT,WIDTH,3))
            _decode(capture,indices,pixels)
            pixels.flush()
            masks = np.empty((len(pretrain),2,FRAMES,*GRID),dtype=bool)
            rng = np.random.default_rng(seed)
            for i,index in enumerate(pretrain_index):
                masks[i] = flow_masks(pixels[mapping[index]],rng,flow)
            arrays = {'fps':np.asarray(fps)}
            for split, rows in labels.items():
                ordered = sorted(rows)
                arrays[split+'_index'] = np.searchsorted(centers,ordered)
                arrays[split+'_labels'] = np.array([rows[c] for c in ordered],dtype=np.int8).reshape(-1,2)
            np.savez_compressed(directory/f'{name}.npz',centers=centers,mapping=mapping,
                                pretrain=pretrain_index,masks=masks,**arrays)
            marker.write_text(json.dumps(signature,indent=2)+'\n')
            print(f'{name}: {len(indices)} RGB frames, {len(pretrain)} pretraining clips.',flush=True)
            prepared.append(name)
        finally:
            capture.release()
    (directory/'sources.json').write_text(json.dumps(prepared)+'\n')


class _Source:
    def __init__(self,directory,name):
        self.name = name
        self.pixels = np.load(directory/f'{name}.npy',mmap_mode='r',allow_pickle=False)
        with np.load(directory/f'{name}.npz',allow_pickle=False) as data:
            self.data = dict(data)

    def clips(self,indices):
        return torch.from_numpy(self.pixels[self.data['mapping'][indices]])


def _prepare_training(sources):
    batches, targets = [], []
    for src in sources:
        labels = src.data['train_labels']
        indices = src.data['train_index']
        known = (labels >= 0).any(axis=1)
        if 'available' in src.data:
            known &= src.data['available'][indices]
        batches.append(src.clips(indices[known]))
        targets.append(torch.from_numpy(labels[known]))
    return torch.cat(batches),torch.cat(targets)


def _pretrain(model,sources,epochs,seed):
    rng = np.random.default_rng(seed)
    items = [(source,i) for source in sources for i in range(len(source.data['pretrain']))]
    if not items:
        raise ValueError('No pretraining clips. Prepare training footage first.')
    decoder = nn.Conv3d(model.mask_token.shape[1],48,1)
    optimizer = torch.optim.AdamW([*model.embedding.parameters(),*model.encoder.parameters(),
                                  model.mask_token,*decoder.parameters()],lr=.002)
    history = []
    for epoch in range(epochs):
        loss_sum = 0.
        for batch in np.array_split(rng.permutation(len(items)),int(np.ceil(len(items)/8))):
            pixels, masks = [], []
            for j in batch:
                src,i = items[j]
                pixels.append(src.clips([src.data['pretrain'][i]])[0])
                masks.append(src.data['masks'][i,epoch%2])
            video = torch.stack(pixels)
            mask = torch.from_numpy(np.stack(masks))
            estimate = decoder(model.encode(video,mask)).permute(0,2,3,4,1)
            loss = F.smooth_l1_loss(estimate[mask],patch_rgb_targets(video)[mask],beta=.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(batch)
        history.append(loss_sum/len(items))
        print(f'Pretrain {epoch+1}/{epochs}: {history[-1]:.4f}',flush=True)
    return history


def _pretraining_inventory(sources):
    frames_by_video = {}
    for source in sources:
        data = source.data
        centers = data['centers'][data['pretrain']]
        if len(centers):
            name = str(data.get('video_name',source.name))
            fps = float(data.get('fps',30))
            frames = (centers[:,None]+np.arange(FRAMES)-CENTER).ravel()
            frames_by_video.setdefault((name,fps),[]).append(frames)
    videos = [
        {'source':name,'unique_frames':count,'unique_seconds':count/fps}
        for (name,fps),parts in frames_by_video.items()
        for count in [len(np.unique(np.concatenate(parts)))]
    ]
    return {'windows':sum(len(source.data['pretrain']) for source in sources),
            'unique_seconds':sum(video['unique_seconds'] for video in videos),'videos':videos}


def _train_classifier(model,sources,epochs,seed):
    videos, targets = _prepare_training(sources)
    known = targets>=0
    positives = (targets==1).sum().item()
    negatives = (targets==0).sum().item()
    if not positives or not negatives:
        raise ValueError('Need both positive and negative supervised targets.')
    weights = torch.where((targets==1).any(axis=1),negatives/positives,1.)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0008)
    torch.manual_seed(seed+1)
    for epoch in range(epochs):
        loss_sum = 0.
        for batch in torch.multinomial(weights,len(weights),replacement=True).split(8):
            clip = videos[batch]
            # Apply one spatial flip to the entire window to preserve temporal motion.
            augmented = torch.rand(len(clip))<.5
            clip[augmented] = clip[augmented].flip(dims=(3,))
            logits = model(clip)
            loss = F.binary_cross_entropy_with_logits(logits[known[batch]],targets[batch][known[batch]].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(batch)
        print(f'Fine-tune {epoch+1}/{epochs}: {loss_sum/len(videos):.4f}',flush=True)
    return {'positive_player_frames':positives,'negative_player_frames':negatives,'clips':len(videos)}


def _evaluate(model,source,threshold=.5):
    indices = source.data['validation_index']
    if not len(indices):
        return None
    labels = source.data['validation_labels']
    available = source.data['available'][indices] if 'available' in source.data else np.ones(len(indices),bool)
    scores = np.full(labels.shape,np.nan,dtype=np.float32)
    if available.any():
        with torch.inference_mode():
            scores[available] = torch.cat([model(source.clips(group)).sigmoid() for group in np.array_split(
                indices[available],int(np.ceil(available.sum()/8)))]).numpy()
    frames = source.data['centers'][indices]
    predictions = scores>=threshold
    pos,neg = labels==1,labels==0
    # Keep reviewed spans on the source timeline, including uncertain boundary frames.
    expanded = np.full((frames[-1]-frames[0]+1,labels.shape[1]),np.nan)
    expanded[frames-frames[0]] = np.where(labels != UNREVIEWED,scores,np.nan)
    references = np.zeros_like(expanded)
    references[frames-frames[0]] = pos
    truth = list(jump_segments(references))
    found = list(jump_segments(expanded,threshold))
    pairs = sorted(((min(e,re)-max(s,rs),i,j) for i,(p,s,e) in enumerate(found)
                    for j,(rp,rs,re) in enumerate(truth) if p==rp and min(e,re)>max(s,rs)),reverse=True)
    used,matched = set(),set()
    for _,i,j in pairs:
        if i not in used and j not in matched:
            used.add(i)
            matched.add(j)
    frame_targets = np.full(expanded.shape,UNREVIEWED,dtype=np.int8)
    frame_targets[frames-frames[0]] = labels
    return {'source':source.name,'airborne_frames_detected':int((predictions&pos).sum()),
            'airborne_frames':int(pos.sum()),'false_positive_frames':int((predictions&neg).sum()),
            'negative_frames':int(neg.sum()),'matched_events':len(matched),'reference_events':len(truth),
            'predicted_events_in_reviewed_intervals':len(found),
            'benchmark':evaluate_jump_scores(frame_targets,expanded,fps=float(source.data.get('fps',30)),
                                             threshold=threshold)}


def _load_sources(directory):
    return [_Source(directory,name) for name in json.loads((directory/'sources.json').read_text())]


def evaluate(directory,checkpoint,threshold=.5):
    """Evaluate a checkpoint on a separate cache of reviewed footage."""
    torch.set_num_threads(3)
    sources = _load_sources(Path(directory))
    model = RGBJumpSegmenter(players=sources[0].data['validation_labels'].shape[1])
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.eval()
    validation = [report for source in sources if (report := _evaluate(model,source,threshold)) is not None]
    if not validation:
        raise ValueError('No validation labels in the evaluation cache.')
    return validation


def train(directory,output, *, pretrain_epochs=6,epochs=14,seed=7):
    directory,output = Path(directory),Path(output)
    report_path = output.with_suffix('.json')
    if output.suffix!='.pt' or output.exists() or report_path.exists():
        raise ValueError('Choose a new .pt output path.')
    if pretrain_epochs<0 or epochs<1:
        raise ValueError('Epochs must be positive, with zero allowed for pretraining.')
    torch.set_num_threads(3)
    torch.manual_seed(seed)
    sources = _load_sources(directory)
    players = sources[0].data['train_labels'].shape[1]
    model = RGBJumpSegmenter(players=players)
    pretrain_loss = _pretrain(model,sources,pretrain_epochs,seed) if pretrain_epochs else []
    trained = _train_classifier(model,sources,epochs,seed)
    model.eval()
    validation = [report for source in sources if (report := _evaluate(model,source)) is not None]
    height,width = sources[0].pixels.shape[1:3]
    report = {'input':'full-frame RGB' if players == 2 else 'per-player RGB crop',
              'window_frames':FRAMES,'resolution':[width,height],'threshold':.5,
              'seed':seed,'pretraining_losses':pretrain_loss,
              'pretraining_corpus':_pretraining_inventory(sources) if pretrain_epochs else None,
              'fine_tune_epochs':epochs,
              'training':trained,'validation':validation}
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(model.state_dict(),output)
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare',help='Decode labeled windows and unlabeled rally windows.')
    prep.add_argument('config',type=Path)
    prep.add_argument('cache',type=Path)
    fit = sub.add_parser('train',help='Train an RGB classifier from a prepared cache.')
    fit.add_argument('cache',type=Path)
    fit.add_argument('output',type=Path)
    fit.add_argument('--pretrain-epochs',type=int,default=6)
    fit.add_argument('--epochs',type=int,default=14)
    fit.add_argument('--seed',type=int,default=7)
    score = sub.add_parser('evaluate',help='Evaluate a checkpoint on reviewed footage.')
    score.add_argument('cache',type=Path)
    score.add_argument('checkpoint',type=Path)
    score.add_argument('--threshold',type=float,default=.5)
    args = parser.parse_args()
    if args.command=='prepare':
        prepare(args.config,args.cache)
    elif args.command=='train':
        train(args.cache,args.output,pretrain_epochs=args.pretrain_epochs,epochs=args.epochs,seed=args.seed)
    else:
        print(json.dumps(evaluate(args.cache,args.checkpoint,args.threshold),indent=2)+'\n')


if __name__ == '__main__':
    main()
