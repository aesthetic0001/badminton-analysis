"""Full-frame RGB model and optical-flow mask propagation for jump correction."""

import cv2
import numpy as np
import torch
from torch import nn


FRAMES = 13
CENTER = FRAMES // 2
WIDTH, HEIGHT, PATCH = 320, 192, 16
GRID = HEIGHT // PATCH, WIDTH // PATCH


def resize_frame(bgr):
    """Resize the entire source frame and letterbox without removing court context."""
    height, width = bgr.shape[:2]
    scale = min(WIDTH / width, HEIGHT / height)
    w, h = round(width * scale), round(height * scale)
    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    left, top = (WIDTH-w)//2, (HEIGHT-h)//2
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return cv2.copyMakeBorder(rgb, top, HEIGHT-h-top, left, WIDTH-w-left, cv2.BORDER_CONSTANT)


def flow_masks(clip, rng, flow, *, count=2, fraction=.55):
    """Seed patch visibility at the middle frame and backward-warp it onto neighbors."""
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in clip]
    map_x, map_y = np.meshgrid(np.arange(WIDTH,dtype=np.float32),np.arange(HEIGHT,dtype=np.float32))
    maps = {}
    for direction in (-1, 1):
        for t in range(CENTER+direction, -1 if direction < 0 else FRAMES, direction):
            # At a target pixel, sample the mask at the corresponding source pixel.
            vectors = flow.calc(gray[t], gray[t-direction], None)
            maps[t] = (map_x+vectors[...,0], map_y+vectors[...,1])
    motion = np.hypot(maps[CENTER-1][0]-map_x, maps[CENTER-1][1]-map_y)
    # Express motion on the actual patch grid while clipping extreme optical-flow noise.
    patch_motion = cv2.resize(np.minimum(motion, 8), GRID[::-1], interpolation=cv2.INTER_AREA)
    probability = (patch_motion + .2).ravel().astype(float)
    probability /= probability.sum()
    marks = []
    keep_hidden = round(fraction * np.prod(GRID))
    for _ in range(count):
        seed = rng.random(GRID, dtype=np.float32)
        hotspot = int(rng.choice(seed.size,p=probability))
        i,j = divmod(hotspot,GRID[1])
        seed[max(0,i-1):i+2,max(0,j-1):j+2] += 1
        field = cv2.resize(seed, (WIDTH,HEIGHT), interpolation=cv2.INTER_LINEAR)
        fields = {CENTER: field}
        for direction in (-1,1):
            for t in range(CENTER+direction, -1 if direction < 0 else FRAMES, direction):
                mx,my = maps[t]
                fields[t] = cv2.remap(fields[t-direction], mx, my, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((FRAMES,*GRID),dtype=bool)
        for t in range(FRAMES):
            patch = cv2.resize(fields[t], GRID[::-1], interpolation=cv2.INTER_AREA).ravel()
            hide = np.argpartition(patch, -keep_hidden)[-keep_hidden:]
            mask[t].flat[hide] = True
        marks.append(mask)
    return np.array(marks)


class _VideoBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.spatial = nn.Conv3d(channels, channels, (1,3,3), padding=(0,1,1),
                                 groups=channels)
        self.temporal = nn.Conv3d(channels, channels, (3,1,1), padding=(dilation,0,0),
                                  dilation=(dilation,1,1), groups=channels)
        self.mix = nn.Conv3d(channels, channels, 1)
        self.norm = nn.GroupNorm(4,channels)

    def forward(self, x):
        return x + self.mix(torch.nn.functional.gelu(self.norm(self.temporal(self.spatial(x)))))


class RGBJumpSegmenter(nn.Module):
    """Encode 13 full RGB frames and predict two centre-frame correction logits."""

    def __init__(self, channels=24):
        super().__init__()
        self.embedding = nn.Conv3d(3, channels, (1,PATCH,PATCH), stride=(1,PATCH,PATCH))
        self.mask_token = nn.Parameter(torch.zeros(1,channels,1,1,1))
        self.encoder = nn.Sequential(*[_VideoBlock(channels,d) for d in (1,2,3)])
        # Keep coarse image layout so near and far predictions can use different context.
        self.pool = nn.AdaptiveAvgPool2d((4,5))
        self.classifier = nn.Sequential(nn.Flatten(),nn.Linear(channels*2*20,64),
                                        nn.GELU(),nn.Dropout(.15),nn.Linear(64,2))

    def encode(self, video, mask=None):
        rgb = video.permute(0,4,1,2,3).float().div(255).sub(.5)
        features = self.embedding(rgb)
        if mask is not None:
            features = torch.where(mask[:,None],self.mask_token,features)
        return self.encoder(features)

    def forward(self, video):
        features = self.encode(video)
        context = torch.cat((features[:,:,CENTER],features.mean(dim=2)),dim=1)
        return self.classifier(self.pool(context))


def patch_rgb_targets(video):
    """Predict a 4×4 RGB grid per hidden patch, using a compact decoder on CPU."""
    batch = len(video)
    gy,gx = GRID
    blocks = video.reshape(batch,FRAMES,gy,4,4,gx,4,4,3)
    # Layout within each 16×16 patch: output bins and pixels within each bin.
    return blocks.float().mean(dim=(4,7)).permute(0,1,2,4,3,5,6).flatten(4).div(255).sub(.5)
