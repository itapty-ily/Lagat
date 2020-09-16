"""
Config.py : the congfiguration of hyper-parameters and the setting of experiments,
            such as the number of training instances
ResNet.py : the definition of Resnet for MNIST and CIFAR10 data set, including the convolutional blocks, GAP
            and fully connectiong layers
ResNet_imagenet.py : the definition of Resnet for TinyImageNet data set, including the convolutional blocks, GAP
            and fully connectiong layers
Dataset.py: the definition of dataset for all data set, the callable one is Mydataset
main.py   : the main file in Lagat, including the influence of hyper-parameters,
            different number of training instances
Network.py: the definition of components in Lagat, including
                D_Z: the discriminator of GAN, fed with drawn latent variables from distributions
                    P(z_x|x) and P(z_y|y). It yields distributions D(z_x) and D(z_y)
                Y_Z: the classifier of Lagat, fed with drawn latent variables from P(z_x|x).
                    It yields the distribution f_x(z_x)
                Z_X: the generator of Lagat, fed with instances from data set, including a pair of ResNets
                    convolutional blocks (from ResNet.py or ResNet_imagenet.py) for \mu(x), \Sigma(x) respectively.
                    And it yeilds a set of LVs drawn from the distribution P(z_x|x)=N(\mu(x), \Sigma(x)),
                    which are denoted as z_x~P(z_x|x).
                    It also returns the \mu(x) for regularization ||\mu(x)-\alpha*e||_2^2.
                Z_Y: the generator of Lagat, fed with Prior LVs e.
                    It yields a set of LVs drawn from the distribution P(z_y|y)=N(\alpha*e, \beta*1)
                    where * denotes the element-wise product. This operation is denoted as z_y~P(z_y|y).
                    Besides the classifier for these f_y also included and it yields the distribution of
                    f_y(z_y) (denoted as y_z_y).
                    It also returns the \alpha*e for regularization ||\mu(x)-\alpha*e||_2^2.
                Y_Augmentation: It only repeats a mini-batch of labels for each sampled LVs.
            Notes:
                Z: LVs
                z_ys   : z_y~P(z_y|y),P(z_y|y)=N(\alpha*e, \beta*1)
                mu_z_y : \alpha*e,
                y_z_y  : f_y(z_y), z_y~P(z_y|y)
                d_z_ys : D(z_y), z_y~P(z_y|y)
                d_z_xs : D(z_x), z_x~P(z_x|x)
                z_xs   : z_x~P(z_x|x), P(z_x|x)=N(\mu(x), \Sigma(x))
                mu_z_x : \mu(x)
                y_zs   : f_x(z_x), z_x~P(z_x|x)
"""
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from Config import opt
import os
import numpy as np
import pickle
import time
import torch.utils.data as Data
import torch
import torchvision as tv
from Dataset import Mydataset
import copy
from Network import * 

from AA.utils import *
from AA.wide_resnet import WideResNet
from AA.auto_augment import AutoAugment, Cutout


def adjust_learning_rate(optimizer, epoch, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    _lr_decay_epoch = [
                       int(6/20*opt.n_epochs),
                       int(12/20*opt.n_epochs),
                       int(16/20*opt.n_epochs),
                       int(18/20*opt.n_epochs),
                       opt.n_epochs]
    if epoch in _lr_decay_epoch and step == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * opt._lr_decay






if __name__ == '__main__':
    result_file = os.path.join(os.path.join(os.path.abspath('.'), 'ICARCV_Result'),
                                '%s_'
                                '%s_'
                                '%d_%4.3f_%4.3f_ACC_DICT.pkl' % (opt.dataset.upper(),
                                                                    opt.ResNet_blocks,
                                                                    opt.train_num, opt.a, opt.b))
    if os.path.exists(result_file) is True:

        if float(os.path.getsize(result_file)) != 0.0:
            print('The experiment at current configuration has been done')

            file = open(result_file, 'rb')
            ACC_Dict = pickle.load(file)
            ACC_Valid_List = ACC_Dict['ACC_Valid_List']
            print('The training set is %s with %.0f training samples with a = %3.2f, b = %3.2f, accuracy = %4.2f' % (opt.dataset.upper(), opt.train_num, opt.a, opt.b, ACC_Valid_List[-1]*100))

            
            file.close()
            # continue
    else:
        print(result_file)
        ACC_Valid_List = []
        if 'AA' in opt.ResNet_blocks:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])
            if opt.dataset.upper() in ['NORB'.upper(), 'MNIST'.upper()]:
                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914],
                                        [0.2023]),
                ])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
            ])
        if opt.dataset.upper() == 'TINYIMAGENET'.upper():
            test_set = Mydataset(train_num=opt.train_num, dataset=opt.dataset, data_flag='valid',
                                    transform=[transform_test], embed=False)
        else:
            test_set = Mydataset(train_num=opt.train_num, dataset=opt.dataset, data_flag='test',
                                    transform=[transform_test], embed=False)
        if 'AA' in opt.ResNet_blocks:
            transform_train = [
                transforms.RandomCrop(test_set.img_size, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
            transform_train.append(AutoAugment())
            if opt.dataset.upper() in ['NORB'.upper(), 'MNIST'.upper()]:
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
        else:
            
            transform_train = transforms.Compose([
                tv.transforms.Pad(padding=2),
                tv.transforms.RandomCrop(size=test_set.img_size),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor()
            ])
        
        train_set = Mydataset(train_num=opt.train_num, dataset=opt.dataset, data_flag='train',
                                transform=[transform_train], embed=True)

        train_loader = Data.DataLoader(
            dataset=train_set,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=1
        )
        test_loader = Data.DataLoader(
            dataset=test_set,
            batch_size=int(500),
            shuffle=False,
            num_workers=1
        )
        z_x = Z_X(opt.gamma, img_chns=train_set.img_channel, img_res=train_set.img_size)
        z_y = Z_Y(opt.gamma, train_set.class_num, img_chns=train_set.img_channel,
                img_res=train_set.img_size, mode='one', dz=z_x.mu_resblock.dz)
        d_z = D_Z(train_set.class_num, dz=z_x.mu_resblock.dz)
        f_y = Y_Z(train_set.class_num, in_features=z_x.mu_resblock.dz)
        f_x = Y_Z(train_set.class_num, in_features=z_x.mu_resblock.dz)
        cuda = True if torch.cuda.is_available() else False
        # Loss function
        criterion = torch.nn.BCELoss()
        classifier_loss = torch.nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        Test_flag = False
        # Initialize generator and discriminator

        if cuda:
            d_z.cuda()
            f_x.cuda()
            z_x.cuda()
            z_y.cuda()
            f_y.cuda()

        # Optimizers
        optimizer_G_CF = torch.optim.Adam([
            {'params': z_y.params, 'lr': opt.lr, 'weight_decay': opt.weight_decay},
            {'params': z_x.params, 'lr': opt.lr},
            {'params': f_x.params, 'lr': opt.lr},
            {'params': f_y.params, 'lr': opt.lr},], lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

        optimizer_D = torch.optim.Adam([
            {'params': d_z.params}
        ], lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)

        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        print('training_num = ', opt.train_num)
        # opt.n_epochs = n_epochs[k]

        for epoch in range(opt.n_epochs):
            for i, (x, y, embeds) in enumerate(train_loader):
                start_time = time.time()
                d_z.train()
                f_x.train()
                z_x.train()
                z_y.train()
                f_y.train()


                if cuda is True:
                    x = torch.FloatTensor(x).cuda()
                    y = y.cuda()
                    embeds = embeds.cuda()
                else:
                    x = torch.FloatTensor(x)
                embeds.squeeze()
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                d_z.zero_grad()
                ys = Y_Augmentation(y, opt.gamma).detach()
                # Loss about real LVs distributions
                z_ys, mu_z_y = z_y(embeds=embeds)
                errCF_prior = torch.FloatTensor([0]).cuda()
                d_z_ys = d_z(z_ys.detach(), ys)
                label_real = torch.full(d_z_ys.size(), 1.0).cuda()
                errD_real = criterion(d_z_ys, label_real)
                D_z_ys = d_z_ys.mean().item()
                # Loss about fake LVs distributions
                z_xs, mu_z_x = z_x(x)
                label_fake = torch.full(d_z_ys.size(), 0.0).cuda()
                d_z_xs = d_z(z_xs.detach(), ys)
                errD_fake = criterion(d_z_xs, label_fake)
                D_z_xs1 = d_z_xs.mean().item()
                # Update D
                errD = errD_real + errD_fake
                errD.backward()
                optimizer_D.step()
                adjust_learning_rate(optimizer_D, epoch, i)

                # Zero_grad operation for components
                z_x.zero_grad()
                f_x.zero_grad()
                z_y.zero_grad()
                f_y.zero_grad()
                # Loss about  N(e*\alpha, 1*\beta) and f_y
                y_z_y = f_y(z_ys)
                errCF_real = classifier_loss(y_z_y, ys)
                hinge_alpha = torch.sum(F.relu(opt.a - z_y.alpha))
                hinge_beta = torch.sum(F.relu(z_y.beta - opt.b))
                # Loss about \mu, \Sigma and f_x jointly
                label = torch.full(d_z_ys.size(), 1.0).cuda()
                d_z_xs = d_z(z_xs, ys)
                errG = criterion(d_z_xs, label)
                y_zs = f_x(z_xs)
                errCF = classifier_loss(y_zs, ys)
                mse = 5.0 * mse_loss(mu_z_x, mu_z_y.squeeze().cuda().detach())
                # Update N(e*\alpha, 1*\beta), f_y, \mu, \Sigma and f_x

                errG_CF =  mse + errCF + errG + errCF_real + hinge_alpha + hinge_beta
                errG_CF.backward()
                # errCF.backward()
                D_z_xs2 = d_z_xs.mean().item()
                optimizer_G_CF.step()
                adjust_learning_rate(optimizer_G_CF, epoch, i)


                if i % 10 == 0:
                    print('[%d/%d][%d/%d] '
                        '[Loss_D: %.4f | Loss_G: %.4f] '
                        '[Loss_CF: %.4f | Loss_Prior_LV: %.4f] '
                        '[D(x): %.4f | D(G(z)): %.4f -> %.4f] at %.0f, lr=%f'
                % (epoch, opt.n_epochs, i, len(train_loader),
                    errD.item(), errG.item(),
                    errCF.item(), errCF_real.item(),
                    D_z_ys, D_z_xs1, D_z_xs2,
                    opt.batch_size / (time.time()-start_time), optimizer_D.param_groups[0]['lr']))
            if errCF.item() < 0.1 or epoch > 0.75 * opt.n_epochs:
                Test_flag = True
            if Test_flag is True or epoch == 0 or epoch == 1:
                acc = 0
                data_loader = test_loader
                start_time = time.time()
                for i, (x, y) in enumerate(data_loader):
                    f_x.eval()
                    z_x.eval()
                    x = torch.FloatTensor(x)
                    y = y.cuda()
                    y_zs = f_x(z_x(x.cuda()))
                    acc += np.sum(torch.max(y_zs, 1)[1].cpu().data.numpy().squeeze() == y.cpu().data.numpy().squeeze())
                acc = int(acc)/len(data_loader.dataset)
                print('The accuracy in valid set is %6.4f percent at %d' % (acc * 100, len(data_loader.dataset)/(time.time()-start_time)))
                ACC_Valid_List.append(acc)
            print('The mean of both distribution: %.4f VS %.4f VS %.4f\n' % (z_ys.mean().item(), z_xs.mean().item(), embeds.mean().item()))

        ACC_Dict = dict()
        ACC_Dict['opt'] = opt
        ACC_Dict['ACC_Valid_List'] = ACC_Valid_List
        output = open(
            result_file,
            'wb')
        pickle.dump(ACC_Dict, output)
        output.close()