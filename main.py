"""
    author: cuiyunhao
"""

import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from hand25d import Hand
from RHP import RHP 
from utils.eval import getPreds2D, getPredsZkr, getPredsZroot, leastsq_s

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='hand')

parser.add_argument('--data', default = '/media/disk1/cuiyunhao/case/hand25D/hand_dataset/RHD_published_v2',
                    type=str, metavar='DIR',help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.0, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate',  
                    dest='evaluate', action='store_false', help='evaluate model on validation set')
parser.add_argument('--usehandmode', default=1, type=int, 
                    help='if use self hand model')
parser.add_argument('--pretrained', dest='pretrained', action='store_false', help='use pre-trained model')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=3, type=int,
                    help='GPU id to use.')

best_prec1 = 0

# optimizer hyper-parameter
alpha = 0.99
epsilon = 1e-8

def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.usehandmode:
        print("=> using self hand model ")
        model = Hand()
    elif args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.gpu is not None:
        model = model.cuda(args.gpu)
    elif args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion = torch.nn.MSELoss().cuda(args.gpu)

    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)
    optimizer = torch.optim.RMSprop(model.parameters(), args.lr, 
                                    alpha = alpha, 
                                    eps = epsilon, 
                                    weight_decay = args.weight_decay, 
                                    momentum = args.momentum)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'training')
    valdir = os.path.join(args.data, 'evaluation')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    '''
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    '''

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        RHP(args.data, 'training'), batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    '''
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''
    val_loader = torch.utils.data.DataLoader(
        RHP(args.data, 'evaluation'),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    #if args.evaluate:
        #validate(val_loader, model, criterion)
        #return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        #prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        '''
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        '''


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    Acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    t1 = time.time()
    for i, (input, target_p, target_z, K, xy, XYZ) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        t2 = time.time()

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target_p_var = target_p.cuda(args.gpu)
            target_z_var = target_z.cuda(args.gpu)

        tgpu = time.time()
        # compute output
        H2D, HZr = model(input)
        t3 = time.time()

        loss1 = criterion(H2D, target_p_var)
        loss2 = criterion(HZr, target_z_var)
        loss = loss1 + 5*loss2
        t4 = time.time()

        # measure accuracy and record loss
        #acc, _ = accuracy(H2D, HZr, K, XYZ)
        losses.update(loss.item(), input.size(0))
        t5 = time.time()
        #Acc.update(acc[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        t6 = time.time()
        loss.backward()
        t7 = time.time()
        optimizer.step()
        t8 = time.time()
        if i%5==1:
            print("********************************************")
            print("     dataloader:        {}".format(t2-t1))
            print("     input.cuda:        {}".format(tgpu-t2))
            print("   model(input):        {}".format(t3-tgpu))
            print(" criterion loss:        {}".format(t4-t3))
            print("    loss.update:        {}".format(t5-t4))
            print("optimizer.zero_grad:    {}".format(t6-t5))
            print("    loss.backward:      {}".format(t7-t6))
            print("     optimizer.step:    {}".format(t8-t7))
            print("     epoch total:       {}".format(t8-t1))
        t1 = time.time()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        '''
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        '''


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    Acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target_p, target_z, K, XYZ) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target_p_var = target_p.cuda(args.gpu)
                target_z_var = target_z.cuda(args.gpu)

            # compute output
            H2D, HZr = model(input)
            loss1 = criterion(H2D, target_p_var)
            loss2 = criterion(HZr, target_z_var)
            loss = loss1 + 5*loss2

            # measure accuracy and record loss
            acc, pred = accuracy(H2D, HZr, K, XYZ)
            losses.update(loss.item(), input.size(0))
            Acc.update(acc[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses, acc=Acc))

        #print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
        #      .format(top1=top1, top5=top5))

    return acc.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(H2D, HZr, K, xy, XYZ):
    """Computes the precision@k for the specified values of k"""
    xy2D = getPreds2D(H2D) # batch*21*2
    Zkr_s = getPredsZkr(HZr, xy2D) # batch*21
    Zroot_s = getPredsZroot(xy, XYZ) # batch
    Zroot_s = Zroot_s.unsqueeze(1).expand(Zroot_s.shape[0], 21)
    Zk_s = Zroot_s + Zkr_s # batch*21
    xy1 = torch.ones(xy2D.shape[0], xy2D.shape[1], 3) # batch*21*3
    xy1[:, :, :2] = xy2D


    Zk_s = Zk_s.unsqueeze(2)
    xy2DZk_s = Zk_s * xy1 # batch*21*3
    xy2DZk_s = xy2DZk_s.unsqueeze(3)
    XYZ_s = torch.zeros(xy2DZk_s.shape) # batch*21*3*1
    invK = torch.zeros(K.shape) # K: batch*3*3
    for i in range(K.shape[0]):
        invK[i,:,:] = torch.inverse(K[i,:,:])
    for i in range(xy2DZk_s.shape[0]):
        for j in range(xy2DZk_s.shape[1]):
            XYZ_s[i,j,:,:]=torch.mm(invK[i,:,:], xy2DZk_s[i,j,:,:])

    
    #XYZ_s = torch.inverse(K)*Zk_s*xy1 # batch*21*3

    s = leastsq_s(XYZ_s) # 1*1

    XYZ_pred = XYZ_s * s # batch*21*3

    scale = torch.sum(torch.max(XYZ, 1) + torch.min(XYZ, 1), 1)/3 # batch*1
    acc_batch = torch.sum(torch.norm(XYZ_pred - XYZ, 2, 2), 1)/21 # batch*1
    acc_scale = acc_batch/scale
    acc = torch.sum(acc_scale)/XYZ.shape[0]
    return acc, XYZ_pred
    '''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    '''


if __name__ == '__main__':
    main()
