import os
import numpy as np
from natsort import natsorted
from natsort.ns_enum import ns 

import torch
from torch.utils.data import Dataset, DataLoader


class ContrastiveNumpyDataset(Dataset):
    """
    This class loads a pair of video clips with their target from npy file stored in the following format:
    [clip_i, clip_j, target] ==> [[3, 16, 112, 112], [3, 16, 112, 112], 1]
    """

    def __init__(self, root_dir):
        self.root_path = root_dir
        self.npy_list = os.listdir(root_dir)

    def __getitem__(self, idx):
        qk_pair = np.load(os.path.join(self.root_path, self.npy_list[idx]), allow_pickle=True)
        clip_i, clip_j, target = qk_pair
        clip_i, clip_j = torch.from_numpy(clip_i).view(3, 16, 112, 112), torch.from_numpy(clip_j).view(3, 16, 112, 112)
        
        return clip_i, clip_j, torch.from_numpy(target)

    def __len__(self):
        return len(self.npy_list)


class SupervisedNumpyDataset(Dataset):
    """
    This class loads a video clip with their target from npy file stored in the following format:
    [clip, target] ==> [[3, 16, 112, 112], 1]
    """

    def __init__(self, root_dir):
        self.root_path = root_dir
        self.npy_list = os.listdir(root_dir)

    def __getitem__(self, idx):
        qk_pair = np.load(os.path.join(self.root_path, self.npy_list[idx]), allow_pickle=True)
        clip, target = qk_pair
        clip = torch.from_numpy(clip).view(3, 16, 112, 112)

        return clip, torch.from_numpy(target)

    def __len__(self):
        return len(self.npy_list)


class SupervisedValidationNumpyDataset(Dataset):
    """
    This class loads a video clip with their target from npy file stored in the following format:
    [clip, target] ==> [[3, 16, 112, 112], 1]
    """

    def __init__(self, root_dir):
        self.root_path = root_dir
        self.npy_list = os.listdir(root_dir)

    def __getitem__(self, idx):
        qk_pair = np.load(os.path.join(self.root_path, self.npy_list[idx]), allow_pickle=True)
        clip, target = qk_pair
        clip = torch.from_numpy(clip).view(3, 16, 112, 112)

        return clip, torch.from_numpy(target)

    def __len__(self):
        return len(self.npy_list)


def get_labeled_npy_loader(args):
    train_dataset = SupervisedNumpyDataset(args.semi_train_path)
    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=0,
                             drop_last=True)
    return trainloader

def get_unlabeled_npy_loader(args):
    train_dataset = ContrastiveNumpyDataset(args.npy_train_dir)
    trainloader = DataLoader(dataset=train_dataset,
                             batch_size=int(np.round(args.mu * args.batch_size)),
                             shuffle=True,
                             num_workers=0,
                             drop_last=False)
    return trainloader


def get_labeled_val_npy_loader(args):
    valid_dataset = SupervisedValidationNumpyDataset(args.npy_root_dir_valid)
    validloader = DataLoader(dataset=valid_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=0,
                             drop_last=False)
    return validloader