from argparse import ArgumentParser, Namespace

from collections import OrderedDict
import random
import math
import numpy as np
import torch
import torchvision.io as io
import av

import matplotlib.pyplot as plt
import transform as transform

from . import models


def convert_checkpoint(model_2d):
    state_dict_inflated = OrderedDict()
    for k, v2d in model_2d.items():
        if "head.projection.weight" in k:
            state_dict_inflated["head.weight"] = v2d.clone()
        elif "head.projection.bias" in k:
            state_dict_inflated["head.bias"] = v2d.clone()
        else:
            state_dict_inflated[k] = v2d.clone()
    return state_dict_inflated


def spatial_sampling(
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=None,
        motion_shift=False,
):
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames = transform.random_crop(frames, crop_size)
        else:
            transform_func = (
                transform.random_resized_crop_with_shift
                if motion_shift
                else transform.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames = transform.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale}) == 1
        frames = transform.random_short_side_scale_jitter(frames, min_scale, max_scale)
        frames = transform.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == tuple:
        mean = torch.tensor(mean)
    if type(std) == tuple:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    new_frames = torch.index_select(frames, 0, index)
    return new_frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips, use_offset=False):
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        if use_offset:
            if num_clips == 1:
                # Take the center clip if num_clips is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(delta / (num_clips - 1))
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def main(args: Namespace):
    mean = (0.45, 0.45, 0.45)
    std = (0.225, 0.225, 0.225)
    with open(args.video, 'rb') as f:
        video_container = f.read()

    video_tensor = torch.tensor(np.frombuffer(video_container, dtype=np.uint8))

    video_meta = {}
    meta = io._probe_video_from_memory(video_tensor)
    video_meta["video_timebase"] = meta.video_timebase
    video_meta["video_numerator"] = meta.video_timebase.numerator
    video_meta["video_denominator"] = meta.video_timebase.denominator
    video_meta["has_video"] = meta.has_video
    video_meta["video_duration"] = meta.video_duration
    video_meta["video_fps"] = meta.video_fps
    video_meta["audio_timebas"] = meta.audio_timebase
    video_meta["audio_numerator"] = meta.audio_timebase.numerator
    video_meta["audio_denominator"] = meta.audio_timebase.denominator
    video_meta["has_audio"] = meta.has_audio
    video_meta["audio_duration"] = meta.audio_duration
    video_meta["audio_sample_rate"] = meta.audio_sample_rate
    fps = video_meta["video_fps"]
    if video_meta["has_video"] and video_meta["video_denominator"] > 0 and video_meta["video_duration"] > 0:
        decode_all_video = False
        clip_size = args.sample_rate * args.num_frame / args.target_fps * fps
        start_idx, end_idx = get_start_end_idx(
            fps * video_meta["video_duration"],
            clip_size,
            -1,
            args.num_clips,
            use_offset=False,
        )
        # Convert frame index to pts.
        pts_per_frame = video_meta["video_denominator"] / fps
        video_start_pts = int(start_idx * pts_per_frame)
        video_end_pts = int(end_idx * pts_per_frame)

    v_frames, _ = io._read_video_from_memory(
        video_tensor,
        seek_frame_margin=1.0,
        read_video_stream=True,
        video_width=0,
        video_height=0,
        video_min_dimension=0,
        video_pts_range=(video_start_pts, video_end_pts),
        video_timebase_numerator=video_meta["video_numerator"],
        video_timebase_denominator=video_meta["video_denominator"],
    )

    frame_list = []
    for _ in range(args.num_aug):
        clip_size = args.sample_rate * args.num_frame / args.target_fps * fps
        start_idx, end_idx = get_start_end_idx(
            v_frames.shape[0],
            clip_size,
            0,
            1,
            use_offset=False,
        )

        new_frames = temporal_sampling(v_frames, start_idx, end_idx, args.num_frame)

        new_frames = tensor_normalize(new_frames, mean, std)
        new_frames = new_frames.permute(3, 0, 1, 2)

        min_scale, max_scale, crop_size = [args.crop_size] * 3
        new_frames = spatial_sampling(
            new_frames,
            spatial_idx=-1,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=False,
            inverse_uniform_sampling=False,
            aspect_ratio=None,
            scale=None,
        )
        frame_list.append(new_frames)

    frames = torch.stack(frame_list, dim=0)

    model = models.__dict__[args.model](num_classes=args.n_classes)
    checkpoints = torch.load(args.weight, map_location='cpu')
    if "model" in checkpoints.keys():
        checkpoint_model = checkpoints["model"]
    else:
        checkpoint_model = checkpoints["model_state"]

    model.load_state_dict(checkpoint_model, strict=False)

    preds = model(frames)

    print(preds.shape)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video', help='Input video path', type=str, default='Data/--6bJUbfpnQ_000017_000027.mp4')
    parser.add_argument('--weight', help='Network weight', type=str)
    parser.add_argument('--sample_rate', help='Sample rate', type=int, default=4)
    parser.add_argument('--num_frame', help='Number of frames', type=int, default=16)
    parser.add_argument('--target_fps', help='Target FPS', type=int, default=30)
    parser.add_argument('--num_clips', help='Number of Clip', type=int, default=10)
    parser.add_argument('--num_aug', help='Number of Augmentation', type=int, default=1)
    parser.add_argument('--num_spatial_crop', help="Number of spatial crops", type=int, default=3)
    parser.add_argument('--crop_size', help='Determine Crop size', type=int, default=256)
    parser.add_argument('--model', type=str, default='vit_large_patch16')
    parser.add_argument('--n_classes', type=int, default=400)
    main(args=parser.parse_args())
