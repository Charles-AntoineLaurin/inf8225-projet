import time

import av
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor

from implementations.prefix_timesformer import TimesformerModel


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    ...     Sample a given number of frame indices from the video.
    ...     Args:
    ...         clip_len (`int`): Total number of frames to sample.
    ...         frame_sample_rate (`int`): Sample every n-th frame.
    ...         seg_len (`int`): Maximum allowed index of sample's last frame.
    ...     Returns:
    ...         indices (`List[int]`): List of sampled frame indices
    """
    converted_len = int(clip_len * frame_sample_rate)

    while converted_len >= seg_len:
        # You could either adjust clip_len or frame_sample_rate, or both
        # For example, reduce clip_len to fit the available frames:
        frame_sample_rate = seg_len // clip_len
        # Recalculate converted_len based on the adjusted clip_len
        converted_len = clip_len * frame_sample_rate

        if converted_len == seg_len:
            frame_sample_rate -= 1
            converted_len = clip_len * frame_sample_rate

    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


def read_video_pyav(container, indices):
    """
    ...     Decode the video with PyAV decoder.
    ...     Args:
    ...         container (`av.container.input.InputContainer`): PyAV container.
    ...         indices (`List[int]`): List of frame indices to decode.
    ...     Returns:
    ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    ..."""
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    if len(frames) == 0:
        pass

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def format_video(video_path):
    container = av.open(video_path)
    seg_len = int(container.streams.video[0].frames)
    indices = sample_frame_indices(clip_len=8, frame_sample_rate=4, seg_len=seg_len)
    video = read_video_pyav(container, indices)
    return video


if __name__ == "__main__":
    file_path = hf_hub_download(
        repo_id="nielsr/video-demo",
        filename="eating_spaghetti.mp4",
        repo_type="dataset",
    )
    container = av.open(file_path)

    # sample 8 frames
    indices = sample_frame_indices(
        clip_len=8, frame_sample_rate=4, seg_len=container.streams.video[0].frames
    )
    video = read_video_pyav(container, indices)

    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")

    # prepare video for the model
    inputs = image_processor(list(video), return_tensors="pt")

    # forward pass
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(list(last_hidden_states.shape))
