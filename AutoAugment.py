import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from AA.utils import *
from AA.wide_resnet import WideResNet
from AA.auto_augment import AutoAugment, Cutout
from kymatio import Scattering2D


from Dataset import Mydataset
from Config import opt
import torchvision as tv
import pickle
import copy
import torch.utils.data as Data
import time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='cifar10',
                        help='dataset name')
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--cutout', default=False, type=str2bool)
    parser.add_argument('--auto-augment', default=True, type=str2bool)
    parser.add_argument('--n_epochs', default=200, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float)
    parser.add_argument('--milestones', default='60,120,160', type=str)
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)
    parser.add_argument('--batch_size', type=int, default=128, help='size of the batches')
    parser.add_argument('--_lr_decay', type=float, default=0.2, help='adam: learning rate decay factor')

    args = parser.parse_args()

    return args







def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s_WideResNet%s-%s' %(args.dataset, args.depth, args.width)
        if args.cutout:
            args.name += '_wCutout'
        if args.auto_augment:
            args.name += '_wAutoAugment'

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # data loading code
    if args.dataset == 'cifar10':
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())
        if args.cutout:
            transform_train.append(Cutout())
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR10(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=8)

        test_set = datasets.CIFAR10(
            root='~/data',
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=8)

        num_classes = 10

    elif args.dataset == 'cifar100':
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())
        if args.cutout:
            transform_train.append(Cutout())
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_set = datasets.CIFAR100(
            root='~/data',
            train=True,
            download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            num_workers=8)

        test_set = datasets.CIFAR100(
            root='~/data',
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=8)

        num_classes = 100

    # create model
    model = WideResNet(args.depth, args.width, num_classes=num_classes)
    model = model.cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'
    ])

    best_acc = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch+1, args.epochs))

        scheduler.step()

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion)

        print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f'
            %(train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc']))

        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_log['loss'],
            train_log['acc'],
            val_log['loss'],
            val_log['acc'],
        ], index=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        if val_log['acc'] > best_acc:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_acc = val_log['acc']
            print("=> saved best model")





def adjust_learning_rate(optimizer, epoch, step, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    _lr_decay_epoch = [# 150,
                       int(4/8*opt.n_epochs),
                       # int(5/8*opt.n_epochs),
                       int(6/8*opt.n_epochs),
                       int(7/8*opt.n_epochs),
                       opt.n_epochs]
    if epoch in _lr_decay_epoch and step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * opt._lr_decay


if __name__ == '__main__':
    ds_list = ['NORB', 'MNIST', 'CIFAR10', 'TINYIMAGENET']
    args = parse_args()
    ResNet_block = 'ResNet'
    for dataset in ds_list:
        args.dataset = dataset
        training_nums = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]
        n_epochs = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
        if args.dataset.upper() == 'TINYIMAGENET'.upper():
            training_nums = [200, 100, 50, 25]
            n_epochs = [200, 200, 200, 200]
        for k in range(len(training_nums)):
            print(training_nums)
            training_num = training_nums[k]

            args.train_num = training_num
            if ResNet_block == 'DHN':
                result_file = os.path.join(os.path.join(os.path.abspath('.'), 'AA'),
                                                        '%s_%d_%s_ACC_DICT.pkl' %
                                                        (args.dataset.upper(), args.train_num, ResNet_block))
            if ResNet_block == 'ResNet':
                result_file = os.path.join(os.path.join(os.path.abspath('.'), 'AA'),
                                                        '%s_%d_ACC_DICT.pkl' %
                                                        (args.dataset.upper(), args.train_num))
            if os.path.exists(result_file) is True:
                if float(os.path.getsize(result_file)) != 0.0:
                    print('The experiment at current configuration has been done')
                    file = open(result_file, 'rb')
                    ACC_Dict = pickle.load(file)
                    # opt = ACC_Dict['opt']
                    # print(opt)
                    # ACC_Valid_List = ACC_Dict['ACC_Valid_List']
                    ACC_Test_List = ACC_Dict['ACC_Test_List']
                    print('The training set is %s with %f training samples' % (args.dataset.upper(), args.train_num))
                    print('The accuracy in test set is %.4f' % (ACC_Test_List[-1]))
                    file.close()
                    continue
            ACC_Test_List = []

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])
            if args.dataset.upper() in ['NORB'.upper(), 'MNIST'.upper()]:
                transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914],
                                    [0.2023]),
            ])
            if args.dataset.upper() == 'TINYIMAGENET'.upper():
                test_set = Mydataset(train_num=args.train_num, dataset=args.dataset, data_flag='valid',
                                        transform=[transform_test], embed=False)
            else:
                test_set = Mydataset(train_num=args.train_num, dataset=args.dataset, data_flag='test',
                                        transform=[transform_test], embed=False)
            transform_train = [
                transforms.RandomCrop(test_set.img_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            if args.auto_augment:
                transform_train.append(AutoAugment())
            if args.cutout:
                transform_train.append(Cutout())

            if args.dataset.upper() in ['NORB'.upper(), 'MNIST'.upper()]:
                transform_train.extend([
                transforms.ToTensor(),
                transforms.Normalize([0.4914],
                                    [0.2023]),])
            else:
                transform_train.extend([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),])
            transform_train = transforms.Compose(transform_train)
            train_set = Mydataset(train_num=args.train_num, dataset=args.dataset, data_flag='train',
                                    transform=[transform_train], embed=False)


            train_loader = Data.DataLoader(
                dataset=train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=1
            )
            test_loader = Data.DataLoader(
                dataset=test_set,
                batch_size=int(500),
                shuffle=False,
                num_workers=1
            )
            if ResNet_block == 'ResNet':
                if args.dataset.upper() in ['TINYIMAGENET'.upper()]:
                    from Resnet_imagenet import ResNet_blocks, ResNet_cf
                    z_x = ResNet_blocks(18, test_set.img_channel, test_set.img_size[0])
                    y_z = ResNet_cf(train_set.class_num)
                elif args.dataset.upper() in ['MNIST'.upper(), 'NORB'.upper(), 'CIFAR10'.upper()]:
                    from Resnet import ResNet_blocks, ResNet_cf
                    z_x = ResNet_blocks(5, test_set.img_channel, test_set.img_size[0])
                    y_z = ResNet_cf(train_set.class_num)
            if ResNet_block == 'DHN':
                if opt.dataset.upper() == 'TINYIMAGENET'.upper():
                    from Scatter_WRN import scatresnet6_2_blocks as Resnet12_8_scat
                else:
                    from Scatter_WRN import resnet12_8_scat_blocks as Resnet12_8_scat
                from DHN import opt as DHNopt
                from Scatter_WRN import ResNet_cf as ResNet_cf
                z_x = Resnet12_8_scat(N=train_set.img_size[0],J=DHNopt.scat, \
                    class_num=train_set.class_num, img_chns=train_set.img_channel)
                y_z = ResNet_cf(num_classes=train_set.class_num, dz=z_x.dz)
                from DHN import opt as DHNopt
                scat = Scattering2D(J=DHNopt.scat, shape=test_set.img_size).cuda()
            cuda = True if torch.cuda.is_available() else False
            # Loss function
            classifier_loss = torch.nn.CrossEntropyLoss()
            # Initialize generator and discriminator
            Test_flag = False
            if cuda:
                z_x.cuda()
                y_z.cuda()
                # full_connect.cuda()


            # Optimizers
            # optimizer_CF = torch.optim.Adam(z_x.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

            optimizer_CF = torch.optim.SGD([
                        {'params': z_x.parameters()},
                        {'params': y_z.parameters()},
                        # {'params': full_connect.parameters()},
            ],lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

            # ----------
            #  Training
            # ----------

            print('training_num = ', training_num)
            # opt.n_epochs = int(np.ceil(5000*0.7 / 64) * n_epochs / (np.ceil(training_num*0.7 / 64) ))
            args.n_epochs = n_epochs[k]
            # n_epochs[k]
            for epoch in range(args.n_epochs):
                for i, (x, y) in enumerate(train_loader):
                    start_time = time.time()
                    z_x.train()
                    y_z.train()
                    # full_connect.train()


                    if cuda is True:
                        x = torch.FloatTensor(x).cuda()
                        y = y.cuda()
                    else:
                        x = torch.FloatTensor(x)
                    ############################
                    # (1) Update classifier
                    ###########################
                    z_x.zero_grad()
                    y_z.zero_grad()
                    # Loss about real LVs distributions
                    if ResNet_block == 'DHN':
                        x = scat(x)
                    z = z_x(x)
                    y_x = y_z(z)
                    # y_x = full_connect(x)
                    errCF = classifier_loss(y_x, y)
                    errCF.backward()
                    optimizer_CF.step()
                    adjust_learning_rate(optimizer_CF, epoch, i, args)
                    if i % 10 == 0:
                        print('[%d/%d][%d/%d] '
                            '[Loss_CF: %.4f ] '
                            'at %.0f'
                    % (epoch, args.n_epochs, i, len(train_loader),
                        errCF.item(),
                        args.batch_size / (time.time()-start_time)))
                print('The accuracy in last batch is %6.4f' % np.mean(
                    torch.max(y_x, 1)[1].cpu().data.numpy().squeeze() ==
                    y.cpu().data.numpy().squeeze()))
                if errCF.item() < 0.1 or epoch > 0.90 * args.n_epochs:
                    Test_flag = True
                if Test_flag is True or epoch == 0:
                    acc = 0
                    data_loader = test_loader
                    start_time = time.time()
                    for i, (x, y) in enumerate(data_loader):
                        
                        z_x.eval()
                        y_z.eval()
                        # full_connect.eval()

                        if cuda is True:
                            x = torch.FloatTensor(x).cuda()
                            y = y.cuda()
                        else:
                            x = torch.FloatTensor(x)
                        if ResNet_block == 'DHN':
                            x = scat(x)
                        y_x = y_z(z_x(x))
                        # y_x = full_connect(x)
                        acc += np.sum(torch.max(y_x, 1)[1].cpu().data.numpy().squeeze() == y.cpu().data.numpy().squeeze())

                    acc = int(acc)/len(data_loader.dataset)
                    print('The accuracy in valid set is %6.4f percent at %d' % (acc * 100, len(data_loader.dataset)/(time.time()-start_time)))
                    ACC_Test_List.append(acc)
            ACC_Dict = dict()
            ACC_Dict['opt'] = opt
            # ACC_Dict['ACC_Valid_List'] = ACC_Valid_List
            ACC_Dict['ACC_Test_List'] = ACC_Test_List
            
            output = open(result_file, 'wb')
            pickle.dump(ACC_Dict, output)
            output.close()

            torch.save(z_x.state_dict(), os.path.join(os.path.join(os.path.abspath('.'),
                                                                'AA_Model'),
                                                    '%s_%d_z_x_DHN' %
                                    (args.dataset.upper(), args.train_num)))
            # torch.save(full_connect.state_dict(), os.path.join(os.path.join(os.path.abspath('.'),
            #                                                        'ResNetB_Model'),
            #                                           '%s_%4.0f_full_connect' %
            #                            (args.dataset.upper(), args.train_num)))
            torch.save(y_z.state_dict(), os.path.join(os.path.join(os.path.abspath('.'),
                                                                'AA_Model'),
                                                    '%s_%d_y_z_DHN' %
                                    (args.dataset.upper(), args.train_num)))
        print('\n\n')