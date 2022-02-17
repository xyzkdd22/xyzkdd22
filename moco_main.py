
from email.mime import base
import imp
import os
import time
import math
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch import device, nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from torchvision.models import video

from models.moco3d import MoCoV3
from arguments import getArgument
from dataset.utils import AverageMeter, accuracy
from dataset.npy_loader import get_labeled_npy_loader, get_unlabeled_npy_loader, get_labeled_val_npy_loader

def ctr(q, k, criterion, tau, args):
    logits = torch.mm(q, k.t())
    N = q.size(0)
    labels = range(N)
    labels = torch.LongTensor(labels).to(args.device)
    loss = criterion(logits/tau, labels)
    return 2*tau*loss


def save_checkpoint(state, is_best, checkpoint, filename='fixmatch_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
                                               
def get_models(name, pretrained=True):
    resnets = {
        "r3d_18": video.r3d_18(pretrained),
        "r2plus1d_18": video.r2plus1d_18(pretrained),
        "mc3_18": video.mc3_18(pretrained)
    }

    if name not in resnets.keys():
        raise KeyError("{} is not a valid 3D ResNet version".format(name))
    return resnets[name]

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def train_step(args, model, labeled_trainloader, unlabeled_trainloader, optimizer, criterion, scheduler, writer, epoch):
    # Track logs
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_prob = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    # Zeroing and Train mode
    model.zero_grad()
    model.train()

    total = int(len(unlabeled_trainloader.dataset) / unlabeled_trainloader.batch_size)
        
    for labeled_sample, unlabeled_sample in tqdm(zip(labeled_trainloader, unlabeled_trainloader), total=total):
        inputs_x, targets_x = labeled_sample
        inputs_u_mask, inputs_u_bp, _ = unlabeled_sample
        
        labeled_batch_size = inputs_x.size(0)
        unlabeled_batch_size = inputs_u_mask.size(0)

        labels = targets_x.long().to(args.device) # torch.tensor(targets_x, dtype=torch.long, device=args.device)
        
        inputs_x, inputs_u_mask, inputs_u_bp = inputs_x.to(args.device), inputs_u_mask.to(args.device), inputs_u_bp.to(args.device)
        _, _, _, _, logits_x, _ = model(inputs_x)
        q1, q2, k1, k2, logits_u_mask, logits_u_bp = model(inputs_u_mask, inputs_u_bp)
        # Compute the Contrastive loss
        L_con = ctr(q1, k2, criterion, args.moco_t, args.device) + ctr(q2, k1, criterion, args.moco_t, args.device)
        
        # Compute standard cross entropy loss for labeled samples
        labeled_loss = F.cross_entropy(logits_x, labels, reduction='mean')
        
        # Compute top1 and top5 accuracy scores
        prec1, prec5 = accuracy(logits_x, labels, topk=(1, 5)) # since we used label-smoothing, we cannot calculate the topk accuracy here as the targets are supposed to be integer tensors such as torch.Size([0, 3, 4, 1, ..., 5])
        top1.update(prec1.item(), labeled_batch_size)
        top5.update(prec5.item(), labeled_batch_size)

        # Compute the pseudo-labeles for the unlabeled sample based on model predictions on masked actors
        with torch.no_grad():
            pseudo_labels = F.softmax(logits_u_mask, dim=-1) # pseudo-labels:  torch.Size([8, 10])
            max_probs, targets_u = torch.max(pseudo_labels, dim=-1)
            mask = max_probs.ge(args.threshold).float()
        
        #  Compute unlabeled loss as cross entropy between strongly augmented (unlabeled) samples and previously computed pseudo-labels.
        unlabeled_loss = (F.cross_entropy(logits_u_bp, targets_u, reduction='none') * mask).mean()

        # Compute total loss 
        loss = labeled_loss.mean() + args.mu * unlabeled_loss + args.beta * L_con

        losses.update(loss.item())
        losses_x.update(labeled_loss.item())
        losses_u.update(unlabeled_loss.item())
        mask_prob.update(mask.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Test the model on-the-fly
        writer.add_scalar("train/1.train_loss", losses.avg, epoch)
        writer.add_scalar("train/2.train_loss_x", losses_x.avg, epoch)
        writer.add_scalar("train/3.train_loss_u", losses_u.avg, epoch)
        writer.add_scalar("train/4.mask", mask_prob.avg, epoch)
        writer.add_scalar("train/5.top1_acc", top1.avg, epoch)
        writer.add_scalar("train/6.top5_acc", top5.avg, epoch)
        
    log = OrderedDict([
        ('x_loss', losses_x.avg),
        ('u_loss', losses.avg),
        ('top1', top1.avg),
        ('top5', top5.avg),
    ])

    return log
    

def validate(args, model, valid_loader, writer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        total = total = int(len(valid_loader.dataset) / valid_loader.batch_size)

        for inputs, targets in tqdm(valid_loader, total=total):

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            
            _, _, _, _, outputs, _ = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            writer.add_scalar("test/1.test_acc", top1.avg, epoch)
            writer.add_scalar("test/2.test_loss", losses.avg, epoch)
            
    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', top1.avg)
    ])

    return log

def main():
    args = getArgument(notebook=False)

    args.device = 'cuda:{}'.format(args.gpus[0]) if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(args.device)
    print("Device used: ", args.device)

    
    # Set dataloaders
    labeledtrainloader = get_labeled_npy_loader(args)
    unlabeledtrainloader = get_unlabeled_npy_loader(args)
    validloader = get_labeled_val_npy_loader(args)

    # define the model
    encoder = get_models(args.arch, pretrained=True)
    model = MoCoV3(base_encoder=encoder, num_classes=args.num_classes)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=args.gpus, output_device=args.device)
    model = model.to(args.device)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    criterion = nn.CrossEntropyLoss().to(args.device)

    writer = SummaryWriter(log_dir=os.path.join(args.results_dir, 'logs'))

    logger = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'x_loss', 'u_loss', 'top1', 'top5', 'test_acc', 'test_loss'
    ])

    best_acc = 0

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_logs = train_step(args, model, labeledtrainloader, unlabeledtrainloader, optimizer, criterion, scheduler, writer, epoch)
        # evaluate on validation set
        valid_logs = validate(args, model, validloader, writer, epoch)
   
        print("Epoch [%d/%d] => X_loss: %.4f - U_loss: %.4f - Top-1_Acc: %.4f - Top-5_Acc: %.4f - Valid_Loss: %.4f - Val_Acc: %.4f"
         %(epoch + 1, args.epochs, train_logs['x_loss'], train_logs['u_loss'], train_logs['top1'], train_logs['top5'], valid_logs['loss'], valid_logs['acc']))

        # Track all logs
        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_logs['x_loss'],
            train_logs['u_loss'],
            train_logs['top1'],
            train_logs['top5'],
            valid_logs['acc'],
            valid_logs['loss']
        ], index=['epoch', 'lr', 'x_loss', 'u_loss', 'top1', 'top5', 'test_acc', 'test_loss'])

        log = logger.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(args.results_dir, '{}_logs.csv'.format(args.arch)), index=False)

        if valid_logs['acc'] > best_acc:
            torch.save({
            'epoch': args.epochs,
            'state_dict': model.state_dict(),
            }, os.path.join(args.results_dir, '{}_model.pth'.format(args.arch)))
            best_acc = valid_logs['acc']
            
    # close the writer upon finish
    writer.close()

if __name__ == '__main__':
    main()

