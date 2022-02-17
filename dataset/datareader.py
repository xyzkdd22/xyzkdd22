import os
import cv2
import random

import numpy as np
from numpy.lib import math
from numpy.lib.type_check import imag
import pandas as pd
from PIL import Image
from natsort import natsorted
from natsort.ns_enum import ns

import torch
from torch.utils.data import Dataset


class VideoFilePathToTensorOD(object):

    def __init__(self, root_dir, num_frames=16):
        self.root_dir = root_dir
        self.num_frames = num_frames
        
    def __call__(self, path):
        # open video file
        path2frame_list = os.path.join(self.root_dir, path)
        
        masked_frames = []
        for frame in natsorted(os.listdir(path2frame_list), alg=ns.IC)[:self.num_frames]:
            frame_path = os.path.join(path2frame_list, frame)
            img = cv2.imread(frame_path, 1)
            masked_frames.append(Image.fromarray(img))
        
        return masked_frames

class VideoFilePathToTensorRGB(object):
    
    def __init__(self, root_dir, num_frames=16):
        self.root_dir = root_dir
        self.num_frames = num_frames

    def __call__(self, path):
        
        # open video file
        cap = cv2.VideoCapture(os.path.join(self.root_dir, path))
        assert (cap.isOpened())

        # init empty output frames (C x L x H x W)
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
        frames = frames[:self.num_frames]
        return frames


class SupervisedValidationVideoLoader(Dataset):

    def __init__(self, args, video_dir, video_list, trans_vid2, num_frames=16):
        self.list_of_video_files = pd.read_csv(video_list)
        
        self.video_dir = video_dir

        self.num_frames = num_frames

        self.transform_vid2 = trans_vid2

        labeled_indices, _ = load_n_percent(args, self.list_of_video_files['target'].to_numpy())

        self.labeled_video_files = np.array(self.list_of_video_files['video_file'])[labeled_indices]
        self.labeled_targets = np.array(self.list_of_video_files['target'])[labeled_indices]
        

    def __len__(self):
        return len(self.labeled_video_files)

    def __getitem__(self, idx):
        labeled_video_file = self.labeled_video_files[idx]
        labeled_target = self.labeled_targets[idx] # self.targets[idx]
        
        # List of PIL images ==> size: num_frames
        labeled_video = VideoFilePathToTensorRGB(self.video_dir, self.num_frames)(labeled_video_file)
        
        # Transform each clip for positive and negative pairs
        labeled_clip = self.transform_vid2(labeled_video)
        target = torch.tensor(labeled_target)

        return labeled_clip, target


def load_n_percent(args, total_labels): # list of all videos labels
    label_per_class = args.num_labeled // args.num_classes

    labels = np.array(total_labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0] 
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)

    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled
        )
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    return labeled_idx, unlabeled_idx


class LabeledVideoLoader(Dataset):

    def __init__(self, args, video_dir, video_list, trans_vid2, num_frames=16):
        self.list_of_video_files = pd.read_csv(video_list)
        
        self.video_dir = video_dir

        self.num_frames = num_frames

        self.transform_vid2 = trans_vid2

        labeled_indices, _ = load_n_percent(args, self.list_of_video_files['target'].to_numpy())

        self.labeled_video_files = np.array(self.list_of_video_files['video_file'])[labeled_indices]
        self.labeled_targets = np.array(self.list_of_video_files['target'])[labeled_indices]
        

    def __len__(self):
        return len(self.labeled_video_files)

    def __getitem__(self, idx):
       
        labeled_video_file = self.labeled_video_files[idx]
        labeled_target = self.labeled_targets[idx] # self.targets[idx]
       

        # List of PIL images ==> size: num_frames
        labeled_video = VideoFilePathToTensorRGB(self.video_dir, self.num_frames)(labeled_video_file)
        
        # Transform each clip for positive and negative pairs
        labeled_clip = self.transform_vid2(labeled_video)
        target = torch.tensor(labeled_target)

        return labeled_clip, target


class VideoPairLoader(Dataset):

    def __init__(self, args, video_dir, frame_dir, video_list, trans_vid1, trans_vid2, num_frames=16, use_label=True):

        self.list_of_video_files = pd.read_csv(video_list)

        self.video_dir = video_dir
        self.frame_dir = frame_dir

        self.num_frames = num_frames

        self.transform_vid1 = trans_vid1
        self.transform_vid2 = trans_vid2

        _, unlabeled_indices = load_n_percent(args, self.list_of_video_files['target'].to_numpy())
        
        self.unlabeled_video_files = self.list_of_video_files['video_file'].to_numpy()[unlabeled_indices]
        self.unlabeled_frame_folder = self.list_of_video_files['video_dir'].to_numpy()[unlabeled_indices]
        self.targets = self.list_of_video_files['target'].to_numpy()[unlabeled_indices]



    def __len__(self):
        return len(self.list_of_video_files['video_file'].to_numpy())

    def __getitem__(self, idx):

        unlabeled_video_file = self.unlabeled_video_files[idx] #self.video_files[idx]
        unlabeled_frame_folder = self.unlabeled_frame_folder[idx] # self.frame_paths[idx]
        target = self.targets[idx]

        # List of PIL images ==> size: num_frames
        masked_clip = VideoFilePathToTensorOD(self.frame_dir, self.num_frames)(unlabeled_frame_folder)
        rgb_clip = VideoFilePathToTensorRGB(self.video_dir, self.num_frames)(unlabeled_video_file)
       
        # Transform each clip for positive and negative pairs
        un_clip_i = self.transform_vid1(masked_clip)
        un_clip_j = self.transform_vid2(rgb_clip)

        return un_clip_i, un_clip_j, torch.tensor(target)
 
