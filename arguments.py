import argparse

from torch.autograd.grad_mode import F

model_names = sorted(["mc3_18", "cnn_rnn34", "cnn_rnn", "r3d_18", "r3d_34", "p_r3d_18", "mc3_18", "r2plus1d_18", "p_r2plus1d_18", "r2plus1d_34"])
def getArgument(notebook=False):
    parser = argparse.ArgumentParser(description='PyTorch UCF101 Training')
    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='r2plus1d_18', choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: r2plus1d_18)')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--num_labeled', default=400, type=int, metavar='N',
                        help='number of labeled images')
    parser.add_argument('--num_classes', default=10, type=int, metavar='N',
                        help='number of classes')
    parser.add_argument("--expand_labels", action="store_true", default=False,
                        help="expand labels to fit eval steps")
    parser.add_argument('--total_steps', default=2**20, type=int, help='number of total steps to run')
    parser.add_argument('--eval_step', default=1024, type=int, # 1024=>1024-epochs, 2048=>512-epochs, 4096=>256-epochs
                        help='number of eval steps to run')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
                        
    parser.add_argument('--milestones', default='60,120,160', type=str)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--mu',default=7, type=float, help= 'coefficient for unlabeled data')
    parser.add_argument('--beta',default=1, type=float, help= 'coefficient for contrastive loss')

    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, metavar='LR', help='initial learning rate',
                        dest='lr')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='path to pretrained checkpoint (default: none) used for finetune')

    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_url', default='', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

    parser.add_argument('--device', default='cuda:0', type=str, help='GPU id to use.')
    parser.add_argument('--gpus', type=int, default=None, nargs='+', help='gpu indices for parallel training')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='where to save models')

    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # moco specific configs:
    parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension (default: 128)')
    parser.add_argument('--moco_k', default=4096, type=int, help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.07, type=float, help='softmax temperature (default: 0.07)')

    # knn monitor
    parser.add_argument('--knn_k', default=200, type=int, help='k in kNN monitor')
    parser.add_argument('--knn_t', default=0.1, type=float,
                        help='softmax temperature in kNN monitor; could be different with moco-t')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true', help='use mlp head')
    parser.add_argument('--aug_plus', action='store_true', help='use moco v2 data augmentation')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

    parser.add_argument('--npy_train_dir', type=str, default='', help='npy train file path')
    parser.add_argument('--npy_valid_dir', type=str, default='', help='path to train dir in npy format')

    parser.add_argument('--train_path', type=str,
                        default='',
                        help='path to train video dir')
    parser.add_argument('--semi_train_path', type=str,
                        default='',
                        help='path to train video dir')
    parser.add_argument('--train_frames', type=str,
                        default='',
                        help='path to train video dir')
    parser.add_argument('--csv_video_list', type=str,
                        default='',
                        help='csv file containing list of train videos with labels')

    parser.add_argument('--npy_root_dir_valid', type=str,
                        default="",
                        help='path to validation video dir')

    parser.add_argument('--results_dir', default='results', type=str, metavar='PATH',
                        help='path to cache (default: none)')
    parser.add_argument('--ssl_type', default='moco', type=str, choices=['simclr', 'moco', 'simsiam', 'byol'],
                        metavar='PATH', help='path to cache (default: moco)')

    parser.add_argument('--num_frames', type=int, default=16, help='number of frames for each clip')
    parser.add_argument('--max_len', type=int, default=32, help='number of frames for each video')
    parser.add_argument('--fps', type=int, default=10, help='number of frames per second')
    parser.add_argument('--padding', type=str, default='last', help='padding the video')
    parser.add_argument('--frame_size', type=int, default=112, help='video frame dimension')
    parser.add_argument('--alpha', type=float, default=0.35, help='alpha controller for the background removal')
    parser.add_argument('--crop_size', type=int, default=112, help='cropping dimension')
    parser.add_argument('--version', type=int, default=5, choices=[1, 2, 3, 4, 5, 6], help='version of mixup')
    parser.add_argument('--out_features', type=int, default=128, help='The output dimension of the pre-trained models')

    parser.add_argument('--T', default=1, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float, help='pseudo label threshold')
    parser.add_argument('--no_progress', action='store_true', default=False, help="don't use progress bar")

    if notebook:
        args = parser.parse_args("")
    else:
        args = parser.parse_args()

    return args
