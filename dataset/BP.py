import random
import numpy as np
from natsort import natsorted
from natsort.ns_enum import ns 
import torch

from torch.nn import functional as F


class SpatialMixup(object):
    def __init__(self, alpha=0.3, trace=True, version=5):
        self.alpha = alpha
        self.trace = trace
        self.version = version

    def mixup_data(self, x, y, size=56):
        """
        Return mixed inputs
        """
        # ====================================================================================================
        #                           Proposed Background Removal Approaches !!!                               #
        # ====================================================================================================
        elif self.version == 1:  
            B, C, T, H, W = x.size()

            loss_prob = self.alpha  # np.random.random() * self.alpha
            if self.trace:
                mixed_x = x
            else:
                mixed_x = torch.zeros_like(x)
            
            for i in range(B):

                batch_idx = np.random.randint(B)
                frame_idx = np.random.randint(T)
                rand_frame = x[batch_idx, :, frame_idx, :, :]

                for j in range(T):
                    mixed_x[i, :, j, :, :] = (1 - loss_prob) * x[i, :, j, :, :] + loss_prob * rand_frame

            return mixed_x

        elif self.version == 2:
            # randomly crop four patches from four different frames of the same video and fuse them with
            # all stable frames
            import random
            B, C, T, H, W = x.size()
            h, w = size, size
            loss_prob = self.alpha  # np.random.random() * self.alpha
            if self.trace:
                mixed_x = x
            else:
                mixed_x = torch.zeros_like(x)

            cropped_frames = {}
            patched_frames = []

            for i in range(B):
                four_frame_indices = random.sample(range(0, T), 4)  # list of four frames indices

                for k in sorted(four_frame_indices):
                    x1 = random.randint(0, H - h)
                    y1 = random.randint(0, W - w)
                    cropped_frames[k] = x[i, :, k, x1: x1 + h, y1:y1 + w]  # torch.Size([3, 56, 56])

                patch_01 = torch.cat((cropped_frames[four_frame_indices[0]],  
                                      cropped_frames[four_frame_indices[1]]), 1)  
                patch_23 = torch.cat((cropped_frames[four_frame_indices[2]],  
                                      cropped_frames[four_frame_indices[3]]), 1)  
                patched_frame = torch.cat((patch_01, patch_23), 2)  

                for j in range(T):
                    mixed_x[i, :, j, :, :] = (1 - loss_prob) * x[i, :, j, :, :] + loss_prob * patched_frame 

                patched_frames.append(patched_frame)

            return mixed_x, patched_frames

        elif self.version == 3:
            # randomly crop four patches from four different frames of four different videos and
            # patch them and mix the patched frame with all frames of the video.
            import random
            B, C, T, H, W = x.size()

            h, w = size, size
            loss_prob = self.alpha  # np.random.random() * self.alpha
            if self.trace:
                mixed_x = x
            else:
                mixed_x = torch.zeros_like(x)

            label_proportions = {}
            label_indices = {}
            cropped_frames = {}
            patched_frames = []
            mixed_targets = []
            for i in range(B):
                # make sure the drop_last=True, to prevent sample out-of range exception. 
                four_batch_indices = random.sample(range(0, B), 4)  
                frame_indices = random.sample(range(0, T), 4)
                
                for k, l in zip(natsorted(four_batch_indices, alg=ns.IC), natsorted(frame_indices, alg=ns.IC)):
                    x1 = random.randint(0, H - h)
                    y1 = random.randint(0, W - w)
                    cropped_frames[k] = x[k, :, l, x1: x1 + h, y1:y1 + w] 

                    label_indices[k] = y[k]  # keep target index
                    label_proportions[k] = (h * w) / T * (H * W) 

                patch_01 = torch.cat((cropped_frames[four_batch_indices[0]],  
                                      cropped_frames[four_batch_indices[1]]), 1)  
                patch_23 = torch.cat((cropped_frames[four_batch_indices[2]],  
                                      cropped_frames[four_batch_indices[3]]), 1)  
                patched_frame = torch.cat((patch_01, patch_23), 2)  
                
                for j in range(T):
                    # Mix frames here
                    mixed_x[i, :, j, :, :] = (1 - loss_prob) * x[i, :, j, :, :] + loss_prob * patched_frame

                label_prop_for_patched_frames = sum([label_proportions[k] * target[k] for k, _ in label_indices.items()])
                
                mixed_y = (1 - loss_prob) * y[i] + loss_prob * label_prop_for_patched_frames
                patched_frames.append(patched_frame)
                mixed_targets.append(mixed_y)


            return mixed_x, torch.tensor(mixed_targets)

        else:
            raise KeyError("Invalid version, specify (1-3) integer number, {} is invalid choice.".format(self.version))


class TemporalMixup(object):
    def __init__(self, alpha, version=1):
        self.alpha = alpha
        self.version = version

    def mixup_data(self, x, size=56):
        """
        Temporal mixing, meaning mixing or distorting randomly selected frames
        """
        if self.version == 1:
            # randomly select single frame from the same video and mix it with
            # some of randomly selected frames
            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
                # lam = np.random.uniform(self.alpha, self.alpha)
            else:
                lam = 1

            B, C, T, H, W = x.size()
            skip = np.random.randint(0, 16) # skip randomly
            # skip = 4
            mixed_x = x
            for i in range(B):
                for j in range(T):
                    mixed_x[i, :, j, :, :] = lam * x[i, :, j, :, :] + (1 - lam) * x[i, :, (j + skip) % T, :, :]
            return mixed_x

        elif self.version == 2:
            # temporally replace the background with randomly patched crops of different frames from the
            # same video
            import random

            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1

            B, C, T, H, W = x.size()
            h, w = size, size
            skip = np.random.randint(0, 16) # skip randomly
            # skip = 4
            mixed_x = x

            cropped_frames = {}

            for i in range(B):
                four_frame_indices = random.sample(range(0, T), 4)  # list of four frames indices

                for k in sorted(four_frame_indices):
                    x1 = random.randint(0, H - h)
                    y1 = random.randint(0, W - w)
                    cropped_frames[k] = x[i, :, k, x1: x1 + h, y1:y1 + w]  
                    
                patch_01 = torch.cat((cropped_frames[four_frame_indices[0]],  
                                      cropped_frames[four_frame_indices[1]]), 1)  
                patch_23 = torch.cat((cropped_frames[four_frame_indices[2]],  
                                      cropped_frames[four_frame_indices[3]]), 1)  
                patched_frame = torch.cat((patch_01, patch_23), 2)  
                
                for j in range(T):
                    mixed_x[i, :, j, :, :] = (1 - lam) * x[i, :, j, :, :] + \
                                             (patched_frame + x[i, :, (j + skip) % T, :, :])

            return mixed_x

        elif self.version == 3:
            # temporally replace the background with randomly patched crops of different frames from the
            # same video
            import random

            if self.alpha > 0:
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1

            B, C, T, H, W = x.size()
            h, w = size, size
            # skip = np.random.randint(0, 16) # skip randomly
            skip = 4
            mixed_x = x

            cropped_frames = {}

            for i in range(B):
                four_batch_indices = random.sample(range(0, B), 4)  # list of four frames indices
                frame_indices = random.sample(range(0, T), 4)

                for k, l in zip(four_batch_indices, frame_indices):
                    x1 = random.randint(0, H - h)
                    y1 = random.randint(0, W - w)
                    cropped_frames[k] = x[k, :, l, x1: x1 + h, y1:y1 + w] 

                patch_01 = torch.cat((cropped_frames[four_batch_indices[0]],  
                                      cropped_frames[four_batch_indices[1]]), 1) 
                patch_23 = torch.cat((cropped_frames[four_batch_indices[2]],  
                                      cropped_frames[four_batch_indices[3]]), 1)  
                patched_frame = torch.cat((patch_01, patch_23), 2)   

                for j in range(T):

                    mixed_x[i, :, j, :, :] = (1 - lam) * x[i, :, j, :, :] + \
                                             (patched_frame + x[i, :, (j + skip) % T, :, :])

            return mixed_x


class GeneratePatches(object):
    def __init__(self, size=56, gen_type='same'):
        self.size = size
        self.gen_type = gen_type

    def __call__(self, x, y):
        B, C, T, H, W = x.size()
        h, w = self.size, self.size

        cropped_frames = {}
        patched_list = []
        labels = {}
        label_indices = {}

        if self.gen_type == 'same':  # intra-video patch generation
            for i in range(B):
                four_frame_indices = random.sample(range(0, T), 4)  # list of four frames indices
                for k in sorted(four_frame_indices):
                    x1 = random.randint(0, H - h)
                    y1 = random.randint(0, W - w)
                    cropped_frames[k] = x[i, :, k, x1: x1 + h, y1:y1 + w]    

                patch_01 = torch.cat((cropped_frames[four_frame_indices[0]],  
                                      cropped_frames[four_frame_indices[1]]), 1)  
                patch_23 = torch.cat((cropped_frames[four_frame_indices[2]],  
                                      cropped_frames[four_frame_indices[3]]), 1)  
                patched_frame = torch.cat((patch_01, patch_23), 2)  
                
                patched_list.append(patched_frame)

            return patched_list

        elif self.gen_type == 'diff':  # inter-video patch generation
            for i in range(B):
                four_batch_indices = random.sample(range(0, B), 4)  # list of four frames indices
                frame_indices = random.sample(range(0, T), 4)

                for k, l in zip(four_batch_indices, frame_indices):
                    x1 = random.randint(0, H - h)
                    y1 = random.randint(0, W - w)
                    cropped_frames[k] = x[k, :, l, x1: x1 + h, y1:y1 + w]  # torch.Size([3, 56, 56])

                patch_01 = torch.cat((cropped_frames[four_batch_indices[0]],  # we patch them towards the height
                                      cropped_frames[four_batch_indices[1]]), 1)  # torch.Size([3, 112, 56])
                patch_23 = torch.cat((cropped_frames[four_batch_indices[2]],  # we patch them towards the height
                                      cropped_frames[four_batch_indices[3]]), 1)  # torch.Size([3, 112, 56])
                patched_frame = torch.cat((patch_01, patch_23),
                                          2)  # we patch them towards width as torch.Size([3, 112, 112])
                # patched_list.append(patched_frame.permute(1, 2, 0))
                patched_list.append(patched_frame)

            return patched_list

        elif self.gen_type == 'both':
            for i in range(B):
                four_batch_indices = random.sample(range(0, B), 4)  # list of four frames indices
                frame_indices = random.sample(range(0, T), 4)

                for k, l in zip(four_batch_indices, frame_indices):
                    x1 = random.randint(0, H - h)
                    y1 = random.randint(0, W - w)
                    cropped_frames[k] = x[k, :, l, x1: x1 + h, y1:y1 + w]  # torch.Size([3, 56, 56])
                    label_indices[k] = y[k]  # keep target index
                    labels[l] = (h * w) / (H * W)  # proportional labels
                patch_01 = torch.cat((cropped_frames[four_batch_indices[0]],  
                                      cropped_frames[four_batch_indices[1]]), 1)  
                patch_23 = torch.cat((cropped_frames[four_batch_indices[2]],  
                                      cropped_frames[four_batch_indices[3]]), 1)  
                patched_frame = torch.cat((patch_01, patch_23), 2)  

                patched_list.append(patched_frame)

            targets = (label_indices, labels)

            return patched_list
        else:
            print("Invalid generation type: {}, only 'same' or 'diff' allowed".format(self.gen_type))

def mix_labels(x, targets, label_indices, label_proportions, loss_prob):
    
    prop_mixed_targets = targets
    
    for k in range(4):
        prop_mixed_targets[label_indices[k]] = label_proportions[k] * targets[label_indices[k]]
    index = torch.randperm(x.size()[0])
    
    mixed_targets = (1 - loss_prob) * prop_mixed_targets[index] + loss_prob * targets

    return mixed_targets
