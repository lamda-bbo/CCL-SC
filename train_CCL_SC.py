
from __future__ import print_function
import moco.CSC
import pandas as pd
import argparse
import os
import time
import random
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import models.cifar as models
import models.non_cifar as non_cifar_models
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, closefig
import dataset_utils, large_dataset_utils
from loss import SelfAdativeTraining, deep_gambler_loss, log_margin_loss

model_names = ("vgg16","vgg16_bn","resnet34", "EfficientNet",'resnet18', "resnext50_32x4d", "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5")

parser = argparse.ArgumentParser(description='Selective Classification for Self-Adaptive Training')
parser.add_argument('-d', '--dataset', default='cifar10', type=str, choices=['cifar10','celeba',  'imagenet', 'cifar100'])
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'tuning'],
                    help='mode: tuning refers to 80/20 split of the training data for hyperparameter tuning')
# Training
parser.add_argument('-t', '--train', dest='evaluate', action='store_true',
                    help='train the model. When evaluate is true, training is ignored and trained models are loaded.')
#  resume
parser.add_argument('-r', '--resume', dest='resume', action='store_true',
                    help='resume the model. When resume is true, training is ignored and trained models are loaded.')
# warmup
parser.add_argument('-w', '--warmup', dest='warmup', action='store_true',
                    help='warmup the reward.')
parser.add_argument('--ir', '--initialreward', default=1e-6, type=float,
                    metavar='IR', help='initial reward')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='WN',
                    help='warm-up iterations')
# 
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='resume epochs to run')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--save_model_step', default=25, type=int, metavar='N',
                    help='number of epochs to run before a model is saved')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=200, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--num_classes', default=150, type=int, metavar='N',
                    help='Number of Classes for ImageNetSubset ONLY')
parser.add_argument('--num_valid', default=1000, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500],
                        help='Multiply learning rate by gamma at the scheduled epochs (default: 25,50,75,100,125,150,175,200,225,250,275)')
parser.add_argument('--gamma', type=float, default=0.5, help='LR is multiplied by gamma on schedule (default: 0.5)') 
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--sat-momentum', default=0.9, type=float, help='momentum for sat')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-o', '--rewards', dest='rewards', type=float, nargs='+', default=[2.2],
                    metavar='o', help='The reward o for a correct prediction; Abstention has a reward of 1. Provided parameters would be stored as a list for multiple runs.')
parser.add_argument('--pretrain', type=int, default=0,
                    help='Number of pretraining epochs using the cross entropy loss, so that the learning can always start. Note that it defaults to 100 if dataset==cifar10 and reward<6.1, and the results in the paper are reproduced.')
parser.add_argument('--coverage', type=float, nargs='+',default=[100.,99.,98.,97.,95.,90.,85.,80.,75.,70.,60.,50.,40.,30.,20.,10.],
                    help='the expected coverages used to evaluated the accuracies after abstention')
# parser.add_argument('--margin_alpha', default=0.1, type=float, help='margin_alpha for log_margin loss')  edit as reward                  
# Save
parser.add_argument('-s', '--save', default='save', type=str, metavar='PATH',
                    help='path to save checkpoint (default: save)')
parser.add_argument('--loss', default='sat', type=str,
                    help='loss function (sat, ce, gambler, sat_entropy, csc)')
parser.add_argument('--entropy', type=float, default=0.0, help='Entropy Coefficient for the SAT Loss (default: 0.0)') 
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16_bn',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vgg16_bn) Please edit the code to train with other architectures')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate trained models on validation set, following the paths defined by "save", "arch" and "rewards"')
#moco
parser.add_argument(
    "--moco-k",
    default=300,
    type=int,
    help="queue size; number of negative keys (default: 65536)",
)
parser.add_argument(
    "--moco-m",
    default=0.999,
    type=float,
    help="moco momentum of updating key encoder (default: 0.999)",
)
parser.add_argument(
    "--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# set the abstention definitions
expected_coverage = args.coverage
reward_list = args.rewards

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
hidden_features = None
hidden_features_k = None
full = False
num_classes=10 # this is modified later in main() when defining the specific datasets
def hook_fn(module, input, output):
    global hidden_features
    hidden_features = output

def hook_fn_k(module, input, output):
    global hidden_features_k
    hidden_features_k = output
def main():
    print(args)

    # make path for the current archtecture & reward
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not resume_path and not os.path.isdir(save_path):
        mkdir_p(save_path)

    # Dataset
    print('==> Preparing dataset %s' % args.dataset)
    global num_classes
    if args.dataset == 'cifar10':
        dataset = dataset_utils.C10
        num_classes = 10
        input_size = 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        trainset = dataset(root='~/datasets/CIFAR10', train=True, download=True, transform=transform_train)
        testset = dataset(root='~/datasets/CIFAR10', train=False, download=True, transform=transform_test)
    elif args.dataset == 'cifar100':
        dataset = dataset_utils.C100
        num_classes = 100
        input_size = 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        trainset = dataset(root='~/datasets/CIFAR100', train=True, download=True, transform=transform_train)
        testset = dataset(root='~/datasets/CIFAR100', train=False, download=True, transform=transform_test)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        input_size = 224

        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        
        train_data_dir = '~/datasets/ImageNet/train'
        test_data_dir = '~/datasets/ImageNet/val'

        if args.mode == 'tuning':
            print("HYPERPARAMETER TUNING MODE")
            trainset = large_dataset_utils.ImageNet_Dataset(train_data_dir, transform=train_trsfm, split='train')
            testset = large_dataset_utils.ImageNet_Dataset(train_data_dir, transform=test_trsfm, split='test') # Different split of train data for hyperparameter tuning
        else:
            print("Normal Training Mode")
            trainset = large_dataset_utils.ImageNet_Dataset(train_data_dir, transform=train_trsfm)
            testset = large_dataset_utils.ImageNet_Dataset(test_data_dir, transform=test_trsfm)
    elif args.dataset == 'celeba':
        num_classes = 2
        input_size = 224
        normalize = transforms.Normalize(
                                    # mean and std on train_dataset ([0.5063486, 0.4258108, 0.38318512], [0.26577517, 0.24520662, 0.24129295])
                                    mean=[0.5063486, 0.4258108, 0.38318512],
                                    std=[0.26577517, 0.24520662, 0.24129295]
                                    )
    
        train_transform = transforms.Compose(
                                        [
                                        transforms.Resize((224, 224)),                                    
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize,
                                        ]
                                    )
        val_transform = transforms.Compose(
                                        [
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        normalize,
                                        ]
                                    )
        trainset = large_dataset_utils.Celeba(root='~/datasets/', split='train', target_type='attr', transform=train_transform, download=True)
        testset = large_dataset_utils.Celeba(root='~/datasets/', split='test', target_type='attr', transform=val_transform, download=True)    
        validset = large_dataset_utils.Celeba(root='~/datasets/', split='valid', target_type='attr', transform=val_transform, download=True)
    # DataLoaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
    if args.dataset == 'celeba':
        validloader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)
    
    # Model
    print("==> creating model '{}'".format(args.arch))
    
    if args.arch != 'vgg16_bn':
        model = non_cifar_models.__dict__[args.arch](num_classes=num_classes if args.loss == 'ce' or args.loss == 'log_ml' or args.loss == 'csc' or args.loss == 'csc_entropy' else num_classes+1)
    else:
        model = models.__dict__[args.arch](num_classes=num_classes if args.loss == 'ce' or args.loss == 'log_ml' or args.loss == 'csc' or args.loss == 'csc_entropy' else num_classes+1, input_size=input_size)

    
    model = model.cuda()
    # if use_cuda: model = torch.nn.DataParallel(model.cuda())

    # encoder_k
    model_k = copy.deepcopy(model)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion2 = None
    if args.pretrain: criterion = nn.CrossEntropyLoss()
    if args.loss == 'ce' or args.loss == 'csc' or args.loss == 'csc_entropy':
        criterion = nn.CrossEntropyLoss() 
        
    elif args.loss == 'gambler':
        criterion = deep_gambler_loss
    elif args.loss == 'sat' or args.loss == 'sat_entropy':
        criterion = SelfAdativeTraining(num_examples=len(trainset), num_classes=num_classes, mom=args.sat_momentum)
    elif args.loss == 'log_ml':
        criterion = log_margin_loss
    elif args.loss == 'csc_sat_entropy':
        criterion = nn.CrossEntropyLoss() 
        criterion2 = SelfAdativeTraining(num_examples=len(trainset), num_classes=num_classes, mom=args.sat_momentum)
    # the conventional loss is replaced by the gambler's loss in train() and test() explicitly except for pretraining
    if args.dataset == 'celeba':
        optimizer = optim.Adam(model.parameters(), lr=1e-5) 
    else:
        optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum, weight_decay=args.weight_decay)


    title = args.dataset + '-' + args.arch + ' o={:.2f}'.format(reward)
    logger = Logger(os.path.join(save_path, 'eval.txt' if args.evaluate else 'log.txt'), title=title)
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Train Loss2','Test Loss', 'Train Err.', 'Test Err.', 'MOCO Err.'])
    


    if args.arch == 'vgg16_bn':
        print(list(model.classifier._modules.items())[2][1].register_forward_hook(hook_fn))
        print(list(model_k.classifier._modules.items())[2][1].register_forward_hook(hook_fn_k))
        args.moco_dim = 512

    elif args.arch == 'resnet34' or args.arch == 'resnet18':
        print(model.avgpool.register_forward_hook(hook_fn))
        print(model_k.avgpool.register_forward_hook(hook_fn_k))
        args.moco_dim = 512
    
    if args.evaluate:
        # print(model)
        print('\nEvaluation only')
        assert os.path.isfile(resume_path), 'no model exists at "{}"'.format(resume_path)
        model = torch.load(resume_path)
        
        if args.arch == 'vgg16_bn':
            print(list(model.classifier._modules.items())[2][1].register_forward_hook(hook_fn))
            print(list(model.classifier._modules.items()))
        elif args.arch == 'resnet34' or args.arch == 'resnet18':
            print(model.avgpool)
            print(model.avgpool.register_forward_hook(hook_fn))
        if use_cuda: model = model.cuda()
        test(trainloader, testloader, model, criterion, args.epochs, use_cuda, evaluation=True)
        return

    if args.resume:
        # print(model)
        print('\nResume model')
        
        assert os.path.isfile(resume_path), 'no model exists at "{}"'.format(resume_path)
        model = torch.load(resume_path)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        optimizer = optim.SGD(model.parameters(), lr=state['lr'], momentum=args.momentum, weight_decay=args.weight_decay)
        if args.arch == 'vgg16_bn':
            print(list(model.classifier._modules.items())[2][1].register_forward_hook(hook_fn))
            print(list(model.classifier._modules.items()))
        elif args.arch == 'resnet34' or args.arch == 'resnet18':
            print(model.avgpool)
            print(model.avgpool.register_forward_hook(hook_fn))
        if use_cuda: model = model.cuda()
        if args.start_epochs in args.schedule:
            index = args.schedule.index(args.start_epochs)
            for _ in range(index):
                adjust_learning_rate(optimizer, args.start_epochs)
            print('resume lr:', state['lr'])

        
    # train
    archive = moco.CSC.MoCo(
        args.moco_dim,
        args.moco_k,
        args.moco_m,
        args.moco_t,
        num_class = num_classes
    )
    best_acc = 0
    for epoch in range(args.start_epochs, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        print('\n'+save_path)
        print('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_loss2, train_acc, moco_top1 = train(trainloader, archive, model, model_k, criterion, criterion2, optimizer, epoch, use_cuda) 
        if args.dataset == 'celeba':
            test_loss, test_acc = test(trainloader, validloader, model, criterion, epoch, use_cuda)
        else:
            test_loss, test_acc = test(trainloader, testloader, model, criterion, epoch, use_cuda)
        print(train_acc, train_loss2, test_acc, moco_top1 * 100)
        if best_acc < test_acc:
            filepath = os.path.join(save_path, "{:d}".format(619) + ".pth")
            torch.save(model, filepath)
            best_acc = test_acc
            print("best_acc: ", best_acc)


        if (epoch+1) % args.save_model_step == 0:
            # save the model
            filepath = os.path.join(save_path, "{:d}".format(epoch+1) + ".pth")
            torch.save(model, filepath)
        
        # append logger file
        logger.append([epoch+1, state['lr'], train_loss, train_loss2, test_loss, 100-train_acc, 100-test_acc, 100-moco_top1 * 100])

    # save the model
    filepath = os.path.join(save_path, "{:d}".format(args.epochs) + ".pth")
    torch.save(model, filepath)
    last_path = os.path.join(save_path, "{:d}".format(args.epochs-1) + ".pth")
    if os.path.isfile(last_path): os.remove(last_path)
    logger.plot(['Train Loss', 'Test Loss'])
    savefig(os.path.join(save_path, 'logLoss.eps'))
    closefig()
    logger.plot(['Train Err.', 'Test Err.'])
    savefig(os.path.join(save_path, 'logErr.eps'))
    closefig()
    logger.plot(['Train Loss', 'Train Loss2'])
    savefig(os.path.join(save_path, 'loss2.eps'))
    closefig()
    logger.close()

def linear_warmup(current_epoch, warmup_epochs, initial_reward, final_reward):
    if current_epoch < warmup_epochs:
        # Calculate the amount to increase at each epoch
        warmup_increment = (final_reward - initial_reward) / warmup_epochs
        # Calculate and return the current reward
        return initial_reward + warmup_increment * current_epoch
    else:
        # After warmup, return the final reward
        return final_reward

def train(trainloader, archive, model, model_k, criterion, criterion2, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    model_k.train()
    # model_k = copy.deepcopy(model)
    if args.arch =='vgg16_bn':
        feature_extractor = model.features

    if epoch == args.pretrain:
        for param_q, param_k in zip(
                model.parameters(), model_k.parameters()
            ):
                param_k.data = param_q.data
    if epoch > args.pretrain:
        for param_q, param_k in zip(
                model.parameters(), model_k.parameters()
            ):
                param_k.data = param_k.data * args.moco_m + param_q.data * (1.0 - args.moco_m)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses2 = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()
    moco_top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    print("TrainLoader Length:", len(trainloader))
    current_reward = reward
    if args.warmup and epoch >= args.pretrain:
        current_reward = linear_warmup(epoch - args.pretrain, args.warmup_epochs, args.ir, reward)
        print('current_reward: ', current_reward)
    for batch_idx,  batch_data in tqdm(enumerate(trainloader)):
        inputs, targets, indices = batch_data
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        with torch.no_grad():
            outputs_k = model_k(inputs)

        if args.arch == 'vgg16_bn':
            outputs_feature = feature_extractor(inputs)
            outputs_feature = outputs_feature.view(outputs_feature.size(0), -1)
            outputs_projection = nn.Sequential(*list(model.classifier.children())[:3])(outputs_feature)
            # print('new feature: ', outputs_projection, outputs_projection.shape)
            outputs = nn.Sequential(*list(model.classifier.children())[3:])(outputs_projection)
        else:
            outputs = model(inputs)
        global full_k1
        global full_k2
        if epoch >= args.pretrain and full_k1 and full_k2:
            # print('batch_idx')
            if args.loss == 'csc' or 'csc_entropy' or 'csc_sat_entropy':
                if args.arch != 'vgg16_bn':
                    # featuer_output, moco_target, moco_error, loss2 = archive(torch.flatten(hidden_features, 1), torch.flatten(hidden_features_k, 1), targets, outputs, outputs_k, epoch + 1, args.pretrain)
                    temp_full_k1, temp_full_k2, moco_error, loss2 = archive(torch.flatten(hidden_features, 1), torch.flatten(hidden_features_k, 1), targets, outputs, outputs_k, epoch + 1, args.pretrain, full_k1 and full_k2)

                else:
                    # featuer_output, moco_target, moco_error, loss2 = archive(hidden_features, hidden_features_k, targets, outputs, outputs_k, epoch + 1, args.pretrain)
                    temp_full_k1, temp_full_k2, moco_error, loss2 = archive(outputs_projection, hidden_features_k, targets, outputs, outputs_k, epoch + 1, args.pretrain, full_k1 and full_k2)
                full_k1 = full_k1 or temp_full_k1
                full_k2 = full_k2 or temp_full_k2
            # if full_k1 == True:
            #     print(full_k1, full_k2)
            if args.loss == 'gambler':
                loss = criterion(outputs, targets, reward)
            elif args.loss == 'sat':
                loss = criterion(outputs, targets, indices)
            elif args.loss == 'sat_entropy':
                softmax = nn.Softmax(-1)
                loss = criterion(outputs, targets, indices) + (args.entropy * (-softmax(outputs[:, :-1]) * outputs[:, :-1]).sum(-1)).mean()
            elif args.loss == 'log_ml':
                loss = F.cross_entropy(outputs, targets) + log_margin_loss(outputs, targets, reward)
            elif args.loss == 'csc':
                if full_k1 and full_k2:
                    losses2.update(loss2.item(), inputs.size(0))
                    moco_top1.update(moco_error, inputs.size(0))
                    # print(loss2)
                    loss = criterion(outputs, targets) + loss2 * current_reward
                else:
                    loss = F.cross_entropy(outputs, targets)
            elif args.loss == 'csc_entropy':
                if full_k1 and full_k2:
                    softmax = nn.Softmax(-1)
                    losses2.update(loss2.item(), inputs.size(0))
                    moco_top1.update(moco_error, inputs.size(0))
                    # print(loss2)
                    loss = criterion(outputs, targets) + loss2 * current_reward + (args.entropy * (-softmax(outputs) * outputs).sum(-1)).mean()
                else:
                    loss = F.cross_entropy(outputs, targets)
            elif args.loss == 'csc_sat_entropy':
                if full_k1 and full_k2:
                    softmax = nn.Softmax(-1)
                    losses2.update(loss2.item(), inputs.size(0))
                    moco_top1.update(moco_error, inputs.size(0))
                    # print(loss2)
                    loss = criterion(outputs, targets) + loss2 * current_reward + criterion2(outputs, targets, indices) + (args.entropy * (-softmax(outputs[:, :-1]) * outputs[:, :-1]).sum(-1)).mean()
                else:
                    loss = F.cross_entropy(outputs[:, :-1], targets)
                    outputs = outputs[:, :-1]
            else:
                loss = criterion(outputs, targets)
        else:
            moco_error = 1/ num_classes
            if args.arch != 'vgg16_bn' and epoch >= args.pretrain:
                temp_full_k1, temp_full_k2, moco_error = archive(torch.flatten(hidden_features, 1), torch.flatten(hidden_features_k, 1), targets, outputs, outputs_k, epoch + 1, args.pretrain, full_k1 and full_k2)
            else:
                if epoch >= args.pretrain:
                    temp_full_k1, temp_full_k2, moco_error = archive(outputs_projection, hidden_features_k, targets, outputs, outputs_k, epoch + 1, args.pretrain, full_k1 and full_k2)
            full_k1 = full_k1 or temp_full_k1
            full_k2 = full_k2 or temp_full_k2
            if args.loss == 'ce' or args.loss == 'log_ml' or args.loss == 'csc' or args.loss == 'csc_entropy':
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = F.cross_entropy(outputs[:, :-1], targets)
                outputs = outputs[:, :-1]
            moco_top1.update(moco_error, inputs.size(0))

        # measure accuracy and record loss
        if args.dataset != 'celeba':
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
        else:
            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss2=losses2.avg,
                    moco_top1=moco_top1.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, losses2.avg, top1.avg, moco_top1.avg)

def test(trainloader, testloader, model, criterion, epoch, use_cuda, evaluation = False):
    global best_acc

    # whether to evaluate uncertainty, or confidence
    if evaluation:
        evaluate(trainloader, testloader, model, use_cuda)
        return

    # switch to test mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    abstention_results = []
    sr_results = []
    abstention_results_nosoftmax = []
    for batch_idx, batch_data in enumerate(testloader):
        inputs, targets, indices = batch_data
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs = inputs.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            output_logits = model(inputs).cpu()
            outputs = output_logits
            values, predictions = outputs.data.max(1)
            # calculate loss
            if args.loss == 'gambler':
                loss = criterion(outputs, targets, reward)
            elif args.loss == 'sat' or args.loss == 'sat_entropy':
                loss = F.cross_entropy(outputs[:, :-1], targets)
            elif args.loss == 'log_ml':
                loss = F.cross_entropy(outputs, targets)
            else:
                loss = criterion(outputs, targets)
            outputs = F.softmax(outputs, dim=1)
            if args.loss == 'ce' or args.loss == 'csc' or args.loss == 'csc_entropy':
                outputs, reservation = outputs, (outputs * torch.log(outputs)).sum(-1) # Reservation is neg. entropy here.
            elif args.loss == 'log_ml':
                top_logits, top_indices = torch.topk(outputs, k=2, dim=-1)


                first_max_indices = top_indices[:, 0]
                second_max_indices = top_indices[:, 1]

                first_max_logits = torch.gather(outputs, 1, first_max_indices.unsqueeze(1))
                second_max_logits = torch.gather(outputs, 1, second_max_indices.unsqueeze(1))

                first_max_logits_nosoftmax = torch.gather(output_logits, 1, first_max_indices.unsqueeze(1))
                second_max_logits_nosoftmax = torch.gather(output_logits, 1, second_max_indices.unsqueeze(1))

                difference = first_max_logits - second_max_logits
                difference_nosoftmax = first_max_logits_nosoftmax - second_max_logits_nosoftmax
                outputs, reservation, reservation_nosoftmax = outputs, -difference.squeeze(), -difference_nosoftmax.squeeze()
                # print(reservation.shape, temp.shape)
            else:
                outputs, reservation = outputs[:,:-1], outputs[:,-1]
            # analyze the accuracy  different abstention level
            abstention_results.extend(zip(list( reservation.numpy() ),list( predictions.eq(targets.data).numpy() )))
            if args.loss == 'log_ml':
                abstention_results_nosoftmax.extend(zip(list( reservation_nosoftmax.numpy() ),list( predictions.eq(targets.data).numpy() )))
            if args.loss == 'ce' or args.loss == 'log_ml' or args.loss == 'csc' or args.loss == 'csc_entropy':
                pred_logits = nn.functional.softmax(output_logits, -1)
            else:
                pred_logits = nn.functional.softmax(output_logits[:,:-1], -1)
            sr_results.extend(zip(list(pred_logits.max(-1)[0].numpy()), list( predictions.eq(targets.data).numpy() )))

            # measure accuracy and record loss
            if args.dataset != 'celeba':
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            else:
                prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    if True:
    	# sort the abstention results according to their reservations, from high to low
        abstention_results.sort(key = lambda x: x[0])
        
        # get the "correct or not" list for the sorted results
        sorted_correct = list(map(lambda x: int(x[1]), abstention_results))
        size = len(sorted_correct)
        print('Abstention Logit: accuracy of coverage ',end='')
        for coverage in expected_coverage:
            covered_correct = sorted_correct[:round(size/100*coverage)]
            print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
        print('')

    	# sort the abstention results according to Softmax Response scores, from high to low
        sr_results.sort(key = lambda x: -x[0])
        # get the "correct or not" list for the sorted results
        sorted_correct = list(map(lambda x: int(x[1]), sr_results))
        size = len(sorted_correct)
        print('Softmax Response: accuracy of coverage ',end='')
        for coverage in expected_coverage:
            covered_correct = sorted_correct[:round(size/100*coverage)]
            print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
        print('')
        if args.loss == "log_ml":
            abstention_results_nosoftmax.sort(key = lambda x: x[0])
            sorted_correct = list(map(lambda x: int(x[1]), abstention_results_nosoftmax))
            size = len(sorted_correct)
            print('Abstention Logit_nonsoftmax: accuracy of coverage ',end='')
            for coverage in expected_coverage:
                covered_correct = sorted_correct[:round(size/100*coverage)]
                print('{:.0f}: {:.3f}, '.format(coverage, sum(covered_correct)/len(covered_correct)*100.), end='')
            print('')
    return (losses.avg, top1.avg)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule and args.dataset != 'celeba':
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']
            
# this function is used to evaluate the accuracy on test set per coverage
def evaluate(trainloader, testloader, model, use_cuda):
    model.eval()
    abortion_results = [[],[]]
    abortion_results_nosoftmax = [[],[]]
    abortion_results_valid = [[],[]]
    sr_results_valid = [[],[]]
    sr_results = [[],[]]

    feature = []
    sim_abortion = [[],[]]
    nce_abortion = [[],[]]
    sim_dot_product_abortion =[[],[]]
    prediction_list = []
    start_t = time.time()


    with torch.no_grad():
        for batch_idx, batch_data in enumerate(testloader):
            inputs, targets = batch_data[:2]
            if use_cuda:
                inputs, targets = inputs.cuda(), targets
            output_logits = model(inputs)
            if args.arch != 'vgg16_bn':
                feature.extend(list(torch.flatten(hidden_features, 1)))
            else:
                feature.extend(list( hidden_features))
            # print(hidden_features.shape)
            # input()
            output = F.softmax(output_logits,dim=1)
            if args.loss == 'ce':
                reservation = 1 - output.data.max(1)[0].cpu()
            elif args.loss == 'log_ml' or 'csc' or args.loss == 'csc_entropy':
                top_logits, top_indices = torch.topk(output, k=2, dim=-1)

                first_max_indices = top_indices[:, 0]
                second_max_indices = top_indices[:, 1]

                first_max_logits = torch.gather(output, 1, first_max_indices.unsqueeze(1))
                second_max_logits = torch.gather(output, 1, second_max_indices.unsqueeze(1))

                first_max_logits_nosoftmax = torch.gather(output_logits, 1, first_max_indices.unsqueeze(1))
                second_max_logits_nosoftmax = torch.gather(output_logits, 1, second_max_indices.unsqueeze(1))
                difference = first_max_logits - second_max_logits
                difference_nosoftmax = first_max_logits_nosoftmax - second_max_logits_nosoftmax
                output, reservation, reservation_nosoftmax = output, -difference.squeeze(), -difference_nosoftmax.squeeze()
                # print(reservation.shape, temp.shape)
            else:
                output, reservation = output[:,:-1], (output[:,-1]).cpu() #the first & the last(n+1 class)
            values, predictions = output.data.max(1)
            predictions = predictions.cpu()
            prediction_list.extend(list(predictions))
            abortion_results[0].extend(list( reservation ))
            abortion_results[1].extend(list( predictions.eq(targets.data) ))
            if args.loss == 'log_ml':
                abortion_results_nosoftmax[0].extend(list( reservation_nosoftmax ))
                abortion_results_nosoftmax[1].extend(list( predictions.eq(targets.data) ))
            if args.loss == 'ce' or args.loss == 'log_ml' or args.loss == 'csc' or args.loss == 'csc_entropy':
                pred_logits = nn.functional.softmax(output_logits, -1)
            else:
                pred_logits = nn.functional.softmax(output_logits[:,:-1], -1)
            sr_results[0].extend(list( -pred_logits.max(-1)[0]))
            sr_results[1].extend(list( predictions.eq(targets.data) ))

    end_t = time.time()
    print(start_t-end_t)
    # valid & test
    correct_list = sr_results[1]
    
    # average_features_train, class_counts_train = get_feature_avg(correct_list_train, prediction_list_train, feature_train)
    average_features, class_counts = get_feature_avg(correct_list[:args.num_valid], prediction_list[:args.num_valid], feature[:args.num_valid])
    sim_abortion[0], sim_abortion_class, nce_abortion[0], sim_dot_product_abortion[0] = cal_sim(average_features, feature[args.num_valid:], prediction_list[args.num_valid:], correct_list[args.num_valid:])
    sim_abortion[1] = sr_results[1][args.num_valid:]
    nce_abortion[1] = sr_results[1][args.num_valid:]
    sim_dot_product_abortion[1] = sr_results[1][args.num_valid:]

    # sim_abortion_by_train[0], sim_abortion_class_by_train, nce_abortion_by_train[0] = cal_sim(average_features, feature[args.num_valid:], prediction_list[args.num_valid:], correct_list[args.num_valid:])
    # sim_abortion_by_train[1] = sr_results[1][args.num_valid:]
    # print(average_features)
    # print(sim_abortion[0], sim_abortion[1] )

    abortion_results_valid[0] = abortion_results[0][:args.num_valid]
    abortion_results_valid[1] = abortion_results[1][:args.num_valid]
    sr_results_valid[0] = sr_results_valid[0][:args.num_valid]
    sr_results_valid[1] = sr_results_valid[1][:args.num_valid]

    abortion_results[0] = abortion_results[0][args.num_valid:]
    abortion_results[1] = abortion_results[1][args.num_valid:]
    sr_results[0] = sr_results[0][args.num_valid:]
    sr_results[1] = sr_results[1][args.num_valid:]

    data_sim = {'sim': sim_abortion[0].tolist(),'sr': [item.item() for item in sr_results[0]], 'true': [int(item) for item in sim_abortion[1]], 'predict': prediction_list[args.num_valid:]}
    df = pd.DataFrame(data_sim)

    # df.to_excel('output.xlsx', index=False)

    abortion_scores, abortion_correct = torch.stack(abortion_results[0]).cpu(), torch.stack(abortion_results[1]).cpu()
    print(abortion_scores.shape)
    # sr_scores, sr_correct = torch.stack(abortion_results[0]), torch.stack(abortion_results[1])
    # sr_scores, sr_correct = torch.stack(sr_results[0]), torch.stack(sr_results[1])
    sr_scores, sr_correct = torch.stack(sr_results[0]).cpu(), torch.stack(sr_results[1]).cpu()
    print(sr_scores.shape)
    # Abstention Logit Results
    abortion_results = []
    bisection_method(abortion_scores, abortion_correct, abortion_results)
    with open(os.path.join(save_path, 'selective risk.txt'), 'w') as file:
        file.write("\nAbstention\tLogit\tTest\tCoverage\tError")
        print("\nAbstention\tLogit\tTest\tCoverage\tError")
        for idx, _ in enumerate(abortion_results):
            print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0]*100., (1 - abortion_results[idx][1])*100))
            file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0]*100., (1 - abortion_results[idx][1])*100))
        # Softmax Response Results
        sr_results = []
        bisection_method(sr_scores, sr_correct, sr_results)
        file.write("\n\nSoftmax\tResponse\tTest\tCoverage\tError")
        print("\Softmax\tResponse\tTest\tCoverage\tError")
        for idx, _ in enumerate(sr_results):
            # print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results[idx][0]*100., (1 - abortion_results[idx][1])*100))
            print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sr_results[idx][0]*100., (1 - sr_results[idx][1])*100))
            file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sr_results[idx][0]*100., (1 - sr_results[idx][1])*100))
        if args.loss == "log_ml":
            abortion_scores_nosoftmax, abortion_correct_nosoftmax = torch.stack(abortion_results_nosoftmax[0]).cpu(), torch.stack(abortion_results_nosoftmax[1]).cpu()
            abortion_results_nosoftmax = []
            bisection_method(abortion_scores_nosoftmax, abortion_correct_nosoftmax, abortion_results_nosoftmax)
            file.write("\nAbstention\tLogit\tTest\tCoverage\tError")
            print("\nAbstention_nosoftmax\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(abortion_results_nosoftmax):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results_nosoftmax[idx][0]*100., (1 - abortion_results_nosoftmax[idx][1])*100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], abortion_results_nosoftmax[idx][0]*100., (1 - abortion_results_nosoftmax[idx][1])*100))
        if True:
            sim_scores, sim_correct = sim_abortion[0].cpu(), torch.stack(sim_abortion[1]).cpu()
            sim_results = []
            bisection_method(sim_scores, sim_correct, sim_results)
            file.write("\nsim\tLogit\tTest\tCoverage\tError")
            print("\nsim\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]*100., (1 - sim_results[idx][1])*100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]*100., (1 - sim_results[idx][1])*100))
        print(sim_scores.size(), sim_correct.size())
        if True:
            product_scores, product_correct = sim_dot_product_abortion[0].cpu(), torch.stack(sim_dot_product_abortion[1]).cpu()
            product_results = []
            print(product_scores)
            bisection_method(-product_scores, product_correct, product_results)
            file.write("\nproduct_results\tLogit\tTest\tCoverage\tError")
            print("\nproduct_results\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(product_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], product_results[idx][0]*100., (1 - product_results[idx][1])*100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], product_results[idx][0]*100., (1 - product_results[idx][1])*100))
        
        if True:
            nce_scores, nce_correct = nce_abortion[0].cpu(), torch.stack(nce_abortion[1]).cpu()
            nce_results = []
            bisection_method(nce_scores, nce_correct, nce_results)
            file.write("\nnce\tLogit\tTest\tCoverage\tError")
            print("\nnce\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(nce_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], nce_results[idx][0]*100., (1 - nce_results[idx][1])*100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], nce_results[idx][0]*100., (1 - nce_results[idx][1])*100))
        
        if True:
            sim_scores, sim_correct = sim_abortion[0].cpu(), torch.stack(sim_abortion[1]).cpu()
            sim_scores2 = (sr_scores + sim_scores)

            sim_results = []
            bisection_method(sim_scores2, sim_correct, sim_results)
            file.write("\nsimxsr\tLogit\tTest\tCoverage\tError")
            print("\nsimxsr\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]*100., (1 - sim_results[idx][1])*100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]*100., (1 - sim_results[idx][1])*100))
        if True:
            sim_scores, sim_correct = sim_abortion[0].cpu(), torch.stack(sim_abortion[1]).cpu()
            sim_scores_2 = torch.max(sim_scores, sr_scores)
            print(sim_scores_2[:10], sim_scores[:10], sr_scores[:10])
            sim_results = []
            bisection_method(sim_scores_2, sim_correct, sim_results)
            file.write("\nsimsr_max\tLogit\tTest\tCoverage\tError")
            print("\nsimsr_max\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]*100., (1 - sim_results[idx][1])*100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]*100., (1 - sim_results[idx][1])*100))
        
        if True:
            sim_results = []
            sim_scores = []
            sim_correct = []
            data_save = []
            for i in range(0, num_classes):
                sim_results_temp = []
                sim_scores_class, sim_correct_class = torch.Tensor(sim_abortion_class[i][0]), torch.stack(sim_abortion_class[i][1]).cpu()
                sim_scores.extend(sim_scores_class)
                sim_correct.extend(sim_correct_class)
            print(torch.Tensor(sim_scores).size(), torch.Tensor(sim_correct).size())



            bisection_method(torch.Tensor(sim_scores), torch.Tensor(sim_correct), sim_results)
            file.write("\nsim\tLogit\tTest\tCoverage\tError")
            print("\nsim\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]*100., (1 - sim_results[idx][1])*100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]*100., (1 - sim_results[idx][1])*100))
        if True:
            sim_results = [(0, 0)] * len(expected_coverage)
            
            for i in range(0, num_classes):
                sim_results_temp = []
                sim_scores_class, sim_correct_class = torch.Tensor(sim_abortion_class[i][0]), torch.stack(sim_abortion_class[i][1]).cpu()
                bisection_method(sim_scores_class, sim_correct_class, sim_results_temp)
                # print(sim_results_temp)
                for j, (a, b) in enumerate(sim_results_temp):
                    sim_results[j] = (sim_results[j][0] + a, sim_results[j][1] + b)
            # print(sim_results)
            file.write("\nsim\tLogit\tTest\tCoverage\tError")
            print("\nsim\tLogit\tTest\tCoverage\tError")
            for idx, _ in enumerate(sim_results):
                print('{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]/num_classes*100., (1 - sim_results[idx][1]/num_classes)*100))
                file.write('\n{:.0f},\t{:.2f},\t\t{:.3f}'.format(expected_coverage[idx], sim_results[idx][0]/num_classes*100., (1 - sim_results[idx][1]/num_classes)*100))
        
    return

def get_feature_avg(correct_list, prediction_list, features):
    data = [(feature, prediction, correct) for feature, prediction, correct in zip(features, prediction_list, correct_list)]

    average_features = [np.zeros_like(features[0].cpu().numpy()) for _ in range(num_classes)]
    class_counts = [0] * num_classes
    error_class_counts = [0] * num_classes

    for feature, prediction, correct in data:
        if correct:
            average_features[prediction.item()] += feature.cpu().numpy()
            class_counts[prediction.item()] += 1
        else:
            error_class_counts[prediction.item()] += 1
    for i in range(num_classes):
        if class_counts[i] > 0:
            average_features[i] /= class_counts[i]


    return average_features, class_counts



def cal_sim(average_features, features, prediction_list, correct_list):
    sim_abortion = []
    sim_abortion_class = [[[], []] for _ in range(num_classes)]
    sim_class = [0] * num_classes
    nce_abortion = []
    average_features_tensor = torch.tensor(average_features)
    sim_dot_product = []
    class_count = [0] * num_classes
    for i, prediction in enumerate(prediction_list):
        a = features[i].cpu().numpy().reshape(1, -1)
        b = average_features[prediction.item()].reshape(1, -1)
        cosine_similarity_matrix = cosine_similarity(a, b)
        
        dot_product = a[0, 0] * b[0, 0]
        sim_abortion.append(-cosine_similarity_matrix[0, 0])
        sim_dot_product.append(dot_product)
        sim_abortion_class[prediction.item()][0].append(-cosine_similarity_matrix[0, 0])
        sim_abortion_class[prediction.item()][1].append(correct_list[i])
        class_count[prediction.item()] += 1
        sim_class[prediction.item()] += sim_abortion[-1]

        # info_nce loss
        a = torch.tensor(a)
        labels = torch.tensor(prediction.item()).unsqueeze(0)
        logits = torch.mm(a, average_features_tensor.t())

        loss = F.cross_entropy(logits, labels)
        nce_abortion.append(loss.item())


    return torch.Tensor(sim_abortion), sim_abortion_class, torch.Tensor(nce_abortion), torch.Tensor(sim_dot_product)


def bisection_method(score, correct, results): 

    def calc_threshold(val_tensor,cov): # Coverage is a perentage in this input
        threshold=np.percentile(np.array(val_tensor), 100-cov*100)
        return threshold

    neg_score = -score
    for coverage in expected_coverage: # Coverage is a number from 0 to 100 here
        threshold = calc_threshold(neg_score, coverage/100)

        mask = (neg_score >= threshold)

        nData = len(correct)
        nSelected = mask.long().sum().item()
        isCorrect = correct[mask]
        nCorrectSelected = isCorrect.long().sum().item()
        passed_acc = nCorrectSelected/nSelected
        results.append((nSelected/nData, passed_acc))


if __name__ == '__main__':
    if args.loss == 'sat_entropy' or args.loss == 'csc_entropy' or args.loss == 'csc_sat_entropy':
        if args.mode == 'tuning':
            base_path = os.path.join(args.save, args.dataset, args.loss, args.mode, f'entropy_coeff-{str(args.entropy)}', args.arch)
        else:
            base_path = os.path.join(args.save, args.dataset, args.loss, f'entropy_coeff-{str(args.entropy)}', args.arch)
            base_path2 = os.path.join(args.save, args.dataset, args.loss)
    else:
        base_path = os.path.join(args.save, args.dataset, args.loss, args.arch)

        
    baseLR = state['lr']
    base_pretrain = args.pretrain
    resume_path = ""
    for i in range(len(reward_list)): 
        state['lr'] = baseLR
        reward = reward_list[i]
        if args.loss == 'csc':
            full_k1 = False
            full_k2 = False
        else:
            full_k1 = True
            full_k2 = True             
        if "imagenet_subset" == args.dataset:
            base_path = os.path.join(base_path, f"nClasses-{args.num_classes}")

        if args.warmup:
            save_path = os.path.join(base_path, 'o{:.2f}'.format(reward), 'k{:.0f}'.format(args.moco_k), 'm{:.4f}'.format(args.moco_m), 't{:.3f}'.format(args.moco_t), 'pretrain{:.0f}'.format(args.pretrain), 'warmup_epochs{:.0f}'.format(args.warmup_epochs), 'initialreward{:.0e}'.format(args.ir))
        else:
            save_path = os.path.join(base_path, 'o{:.2f}'.format(reward), 'k{:.0f}'.format(args.moco_k), 'm{:.4f}'.format(args.moco_m), 't{:.3f}'.format(args.moco_t), 'pretrain{:.0f}'.format(args.pretrain), f"seed-{args.manualSeed}")

        if args.evaluate:
            resume_path= os.path.join(save_path,'{:d}.pth'.format(args.epochs))
        if args.resume:
            if args.loss == 'csc_entropy' or args.loss == 'csc_sat_entropy':
                resume_path = os.path.join(base_path2, 'resume','{:d}.pth'.format(args.start_epochs))
            else:
                resume_path = os.path.join(base_path, 'resume','{:d}.pth'.format(args.start_epochs))
        args.pretrain = base_pretrain
        
        # default the pretraining epochs to 100 to reproduce the results in the paper
        if args.loss == 'gambler' and args.pretrain == 0:
            if  args.dataset == 'cifar10' and reward < 6.3:
                args.pretrain = 100
            elif args.dataset == 'svhn' and reward < 6.0:
                args.pretrain = 50
            elif args.dataset == 'catsdogs':
                args.pretrain = 50
        
        main()
