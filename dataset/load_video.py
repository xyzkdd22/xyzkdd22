from PIL import Image
import cv2
import numpy as np
from numpy.lib.npyio import load
import pandas as pd
from torch.utils import data
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler

from datareader import VideoPairLoader, LabeledVideoLoader

from torchvideotransforms.video_transforms import *
from torchvideotransforms.volume_transforms import ClipToTensor


import BP

# %% Use pytorch Dataloader to load from custom dataset

def get_mean_of_list(L):
    return sum(L) / len(L)


def get_total(loader):
    return int(len(loader.dataset) / loader.batch_size)


def get_train_loader(args):
    # Initialize video transformations
    trans_1 = Compose([
        RandomRotation(15),
        Resize((args.frame_size, args.frame_size)),
        ColorJitter(0.8, 0.8, 0.8, 0.3),
        ClipToTensor(),
        # Normalize(mean=[0.4286, 0.4046, 0.3764], std=[0.1989, 0.1930, 0.1889])
    ])

    trans_2 = Compose([
        Resize((args.frame_size, args.frame_size)),
        RandomGrayscale(p=0.2),
        ClipToTensor(),
        # Normalize(mean=[0.4286, 0.4046, 0.3764], std=[0.1989, 0.1930, 0.1889])
    ])

    # Load from local machine
    labeled_dataset = LabeledVideoLoader(args, args.train_path, args.csv_video_list, trans_2, args.num_frames)
    labeled_trainloader = DataLoader(labeled_dataset, batch_size=args.batch_size, drop_last=False, num_workers=args.workers) # sampler=SequentialSampler(labeled_dataset)

    unlabeled_dataset = VideoPairLoader(args, args.train_path, args.train_frames, args.csv_video_list, trans_1, trans_2, args.num_frames)
    unlabeled_trainloader = DataLoader(unlabeled_dataset, batch_size=int(np.round(args.mu * args.batch_size)), shuffle=True, drop_last=False, num_workers=args.workers) # sampler=SequentialSampler(unlabeled_dataset),


    return labeled_trainloader, unlabeled_trainloader


def save_batch_clips(batched_input, dest_dir):
     toPILImage = transforms.ToPILImage()
     B, C, T, H, W = batched_input.size()
     frames = batched_input.permute(0, 2, 1, 3, 4) # => torch.Size([B, T, H, W, C])
     for b_idx in range(B):
         print("Saving Batch... {}".format(str(b_idx+1)))
         dest_path = os.path.join(dest_dir, "batch_idx_{}".format(str(b_idx + 1)))
         os.makedirs(dest_path, exist_ok=True)
         for f_idx in range(T):
             frame = toPILImage(frames[b_idx][f_idx]).convert("RGB")
             frame.save(os.path.join(dest_path, "clip_{}_batch_idx_{}.jpg".format(str(f_idx+1), str(b_idx+1))))

if __name__ == '__main__':
    from arguments import getArgument
    import timeit
    import os
    from torchvision import transforms

    start = timeit.default_timer()
    args = getArgument(notebook=False)

    labeledloader, unlabeledloader = get_train_loader(args)

    for i, (labeled_sample, unlabeled_sample) in enumerate(tqdm(zip(labeledloader, unlabeledloader), desc='Loading batches: ', total=get_total(unlabeledloader))):
        clip, target = labeled_sample
        masked_clip, clip_j = unlabeled_sample
        
        mixed_labeled_clip, mixed_targets = BP.SpatialMixup(alpha=args.alpha, version=args.version).mixup_data(x=clip, y=target, size=args.crop_size)

        mixed_unlabeled, _ = BP.SpatialMixup(alpha=args.alpha, version=args.version).mixup_data(x=clip, y=target, size=args.crop_size)

        # mixed_unlabeled, _ = BP.SpatialMixup(alpha=args.alpha, version=args.version).mixup_data(x=clip, y=target,
        #                                                                                        size=args.crop_size)
        
        for b_idx in range(clip.size(0)):
            clip_1_frames = mixed_labeled_clip[b_idx, :, :, :, :]
            clip_2_frames = masked_clip[b_idx, :, :, :, :]
            clip_3_frames = mixed_unlabeled[b_idx, :, :, :]
            target_labels = mixed_targets[b_idx]
            
        
            npy_file = [clip_1_frames.view(-1).numpy(), target_labels.numpy(), clip_2_frames.view(-1).numpy(), clip_3_frames.view(-1).numpy()]
            np.save(args.npy_train_dir + '/ucf10_BP_ssl_batch_{}_step_{}_version_{}_alpha_{}_crop_size_{'
                                         '}_frames_per_clip_{}_frame_size_{}'.format(str(i),
                                                                                     str(b_idx),
                                                                                     str(args.version),
                                                                                     str(args.alpha),
                                                                                     args.crop_size,
                                                                                     args.num_frames,
                                                                                     args.frame_size), npy_file)

    print('Time taken to complete training is: [{}] minutes.'.format(int(timeit.default_timer() - start) / 60))
