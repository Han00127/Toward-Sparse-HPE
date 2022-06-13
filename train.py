from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from torchvision import transforms
# from Pytorch_Retinaface.data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from data import preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss_HPELoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from network import RetinaFace_HPE

from datasets import getDataset, detection_collate
from test import eval

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--dataset', dest='dataset', help='Dataset type. [BIWI, Pose_300W_LP, BIWI_masked, Pose_300W_LP_masked]',
                    default='BIWI', # BIWI,Pose_300W_LP,BIWI_masked,Pose_300W_LP_masked
                    type=str)
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--batch_size', default=24, type=int)

args = parser.parse_args()

cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

# rgb_mean = (104, 117, 123) # bgr order
rgb_mean = (123, 117, 104) # rgb order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = args.batch_size
max_epoch = args.epoch
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.dataset
save_folder = args.save_folder

net = RetinaFace_HPE(cfg=cfg)
# print("Printing net...")
# print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True


optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss_HPELoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    # dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))

    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225])
    # transformations = transforms.Compose([transforms.Resize(240),
    #                                       transforms.RandomCrop(224),
    #                                       transforms.ToTensor(),
    #                                       normalize])
    rgb_mean_normalize = tuple(m / 255. for m in rgb_mean)
    # dataset = getDataset(args.dataset, preproc(img_dim, rgb_mean_normalize))
    datasets = []
    for dataset_name in args.dataset.split(','):
        datasets.append(getDataset(dataset_name, preproc(img_dim, rgb_mean_normalize)))
        print(dataset_name, len(datasets[-1]))
        # cfg['name'] += '_' + dataset_name
    dataset = data.ConcatDataset(datasets)
    # dataloader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)

    test_dataset = getDataset('BIWI', train_mode=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers)
    test2_dataset = getDataset('AFLW2000', train_mode=False)
    test2_dataloader = torch.utils.data.DataLoader(test2_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=num_workers)

    save_folder = os.path.join(args.save_folder, args.dataset, args.dataset) + '__'
    if not os.path.exists(os.path.dirname(save_folder)):
        os.makedirs(os.path.dirname(save_folder))
    print('\nSave weights at ' + save_folder)

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    print('\nStart training')
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_epoch_' + str(epoch) + '.pth')
            epoch += 1


        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets, poses, names = next(batch_iterator)
        images = images.cuda()
        targets = targets.cuda()
        poses = poses.cuda()

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm, loss_pose = criterion(out, priors, targets, poses)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm + cfg['pose_weight'] * loss_pose
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        if (iteration + 1) % 100 == 0:
            print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} Pose: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
                  .format(epoch, max_epoch, (iteration % epoch_size) + 1,
                  epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), loss_pose.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))


        if iteration % epoch_size == 0 and iteration != 0:
            net.eval()
            with torch.no_grad():
                error_log = eval(net, test_dataloader, cfg)
                print('Epoch:{}/{} || BIWI Validation Error: [{}]'.format(epoch - 1, max_epoch, error_log))
                error_log = eval(net, test2_dataloader, cfg)
                print('Epoch:{}/{} || AFLW2000 Error: [{}]'.format(epoch - 1, max_epoch, error_log))
                print()
            net.train()

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()