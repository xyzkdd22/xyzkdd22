import os
import cv2
import random

import numpy as np
import pandas as pd
from PIL import Image
from natsort import natsorted

import torch
from torch.utils.data import Dataset

from pixellib.torchbackend.instance import instanceSegmentation


class VideoFilePathToTensor(object):
    """ load video at given file path to torch.Tensor (C x L x H x W, C = 3)
        It can be composed with torchvision.transforms.Compose().

    Args:
        max_len (int): Maximum output time depth (L <= max_len). Default is None.
            If it is set to None, it will output all frames.
        fps (int): sample frame per seconds. It must lower than or equal the origin video fps.
            Default is None.
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, root_dir, max_len=32):
        self.root_dir = root_dir
        self.max_len = max_len

    def __call__(self, path):
        """
        Args:
            path (str): path of video file.
        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """
        # open video file
        cap = cv2.VideoCapture(os.path.join(self.root_dir, path))
        assert (cap.isOpened())
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        for frame_idx in range(num_frames):
            # read frame
            ret, frame = cap.read()
            if ret:
                # successfully read frame
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # pil_frame = Image.fromarray(frame)
                # frames.append(pil_frame)
                frames.append(frame)
        cap.release()
        frames = frames[:self.max_len]
        return frames


class VideoToRandomFramesTensor(object):
    """ load video at given file path to torch.Tensor (C x L x H x W, C = 3)
        It can be composed with torchvision.transforms.Compose().

    Args:
        max_len (int): Maximum output time depth (L <= max_len). Default is None.
            If it is set to None, it will output all frames.
        fps (int): sample frame per seconds. It must lower than or equal the origin video fps.
            Default is None.
        padding_mode (str): Type of padding. Default to None. Only available when max_len is not None.
            - None: won't padding, video length is variable.
            - 'zero': padding the rest empty frames to zeros.
            - 'last': padding the rest empty frames to the last frame.
    """

    def __init__(self, root_dir, max_len=32):
        self.root_dir = root_dir
        self.max_len = max_len

    def __call__(self, path):
        """
        Args:
            path (str): path of video file.
        Returns:
            torch.Tensor: Video Tensor (C x L x H x W)
        """
        # open video file
        cap = cv2.VideoCapture(os.path.join(self.root_dir, path))
        assert (cap.isOpened())
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        for frame_idx in range(num_frames):
            # read frame
            ret, frame = cap.read()
            if ret:
                # successfully read frame
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame)
        cap.release()
        frame_indices = natsorted(random.sample(range(0, len(frames)-1), self.max_len))
        clip = [frames[i] for i in frame_indices]
        return clip


class VideoPairLoader(Dataset):

    def __init__(self, video_dir, video_list, trans_vid1, trans_vid2, num_frames=16, max_len=32):

        self.list_of_video_files = pd.read_csv(video_list)
        self.video_files = self.list_of_video_files['video_file']
        self.labels = self.list_of_video_files['label']
        self.targets = self.list_of_video_files['target']
        self.video_dir = video_dir

        self.max_len = max_len
        self.num_frames = num_frames

        self.transform_vid1 = trans_vid1
        self.transform_vid2 = trans_vid2

        unique_labels = np.unique(self.labels)
        self.class2idx = {unique_labels[i]: i for i in range(len(unique_labels))}
        self.idx2class = {k: v for k, v in self.class2idx.items()}

        self.classes = unique_labels

        self.ins = instanceSegmentation()
        self.ins.load_model("./checkpoints/pointrend_resnet50.pkl")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):

        video_file = self.video_files[idx]
        target = self.targets[idx]

        # List of ndarray images ==> size: mx_len
        video = VideoFilePathToTensor(self.video_dir, self.max_len)(video_file)

        # List of self.num_frames or 16 PIL.Image as clip_1 and clip_2
        clip1 = video[:self.num_frames]
        masked_clip = self.ins.segmentListOfFrames(clip1,
                                 mask_points_values=False,
                                 show_bboxes=True,
                                 save_extracted_objects=False,
                                 extract_from_box=False,
                                 extract_segmented_objects=True)
        pill_masked_clips = [Image.fromarray(frame) for frame in masked_clip]

        clip2 = video[self.num_frames:]
        pill_clip2 = [Image.fromarray(frame) for frame in clip2]

        # Transform each clip for positive and negative pairs
        clip_i = self.transform_vid1(pill_masked_clips)
        clip_j = self.transform_vid2(pill_clip2)

        return clip_i, clip_j, torch.tensor(target)
