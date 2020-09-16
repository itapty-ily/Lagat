import torch.nn as nn
import torch.nn.functional as F
from Resnet_imagenet import ResNet, ResNet_blocks, ResNet_cf
import numpy as np
from torch.nn.parameter import Parameter
import torch
import math
from Config import opt
from torch.distributions.normal import Normal
import time


class D_Z(nn.Module):
    def __init__(self, label_num, dz):
        super(D_Z, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dz, label_num),
            nn.Sigmoid()
        )
        self.params = list(filter(self.__filter_func__,
                                self.parameters()))

    def __filter_func__(self, item):

        if id(item) not in list(map(id, [])):
            return True
        else:
            return False

    def forward(self, z, y):
        if self.training is True:
            mask = torch.ByteTensor(self.model(z).size()).cuda()
            mask.zero_()
            mask.scatter_(1, torch.unsqueeze(y, -1), 1)
            validity = torch.masked_select(self.model(z), mask)
            return validity
        else:
            print('In test stage, this component is not callable.')

class Y_Z(nn.Module):
    def __init__(self, label_num, in_features=256):
        super(Y_Z, self).__init__()
        self.resnet_cf = nn.Linear(in_features=in_features, out_features=label_num)
        self.params = list(filter(self.__filter_func__,
                                self.parameters()))

    def __filter_func__(self, item):

        if id(item) not in list(map(id, [])):
            return True
        else:
            return False

    def forward(self, z):
        if self.training is True:
            logit = self.resnet_cf(z)
            # y = nn.Softmax(logit)
        else:
            logit = self.resnet_cf(z)
            # y = logit
        return logit

class Z_Y(nn.Module):
    def __init__(self, gamma, class_num, mode=None, img_chns=1, img_res=None, dz=64):
        super(Z_Y, self).__init__()
        self.mode = mode
        self.alpha = Parameter(torch.Tensor(dz))
        self.beta = Parameter(torch.Tensor(dz))
        self.gamma = gamma
        self.params = list(filter(self.__filter_func__,
                                self.parameters()))
        self.reset_parameters()
        self.dz = dz

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.alpha.size(0))
        self.alpha.data.uniform_(opt.a-stdv, opt.a+stdv)
        self.beta.data.uniform_(opt.b-stdv, opt.b+stdv)

    def __filter_func__(self, item):

        if id(item) not in list(map(id, [])):
            return True
        else:
            return False

    def forward(self, embeds=None, x=None):
        """
        :param input_instance: ->(batch_size, feature_dim)
        :return:
        """
        if embeds is not None and x is None:
            embeds.detach()
            if self.training is True:
                mu_z_y = torch.mul(embeds, self.alpha).squeeze()
                ones = torch.ones(embeds.size()).cuda()
                sigma_z_y = torch.mul(ones, self.beta).squeeze()
                gaussian_distribution = Normal(mu_z_y, sigma_z_y)
                z_ys = gaussian_distribution.sample([self.gamma]).reshape(-1, self.dz)
                if self.mode is not None:
                    return torch.squeeze(z_ys).cuda(), mu_z_y
                else:
                    y_z_y = self.full_connect(z_ys)
                    return torch.squeeze(z_ys).cuda(), mu_z_y, y_z_y
            else:
                print('In test stage, this component is not callable.')
        if x is not None and embeds is None:
            if self.training is True:
                y_x, embed  = self.mu_resnet(x)
                mu_z_y = torch.mul(embed.detach(), self.alpha).squeeze()
                ones = torch.ones(embed.size()).cuda()

                sigma_z_y = torch.mul(ones, self.beta).squeeze()
                z_ys = list()
                for i in range(self.gamma):
                    z_y_i = torch.normal(mu_z_y, sigma_z_y)
                    z_ys.append(z_y_i)
                z_ys = torch.cat(z_ys, 0)
                if self.mode is not None:
                    return torch.squeeze(z_ys).cuda(), mu_z_y
                else:
                    y_z_y = self.full_connect(z_ys)
                    return torch.squeeze(z_ys).cuda(), mu_z_y, y_z_y, y_x, embed
            else:
                print('In test stage, this component is not callable.')


    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.feature_dim, self.code_len
        )

class Z_X(nn.Module):
    def __init__(self, gamma, img_chns, img_res):
        super(Z_X, self).__init__()
        self.gamma = gamma
        if 'ResNet' in opt.ResNet_blocks:
            if opt.dataset.upper() == 'TINYIMAGENET'.upper():
                print("from Resnet_imagenet import ResNet_blocks")
                from Resnet_imagenet import ResNet_blocks
                self.mu_resblock = ResNet_blocks(18, img_chns, img_res)
                self.sigma_resblock = ResNet_blocks(18, img_chns, img_res)
            elif opt.dataset.upper() in ['MNIST'.upper(), 'NORB'.upper(), 'CIFAR10'.upper()]:
                print("from Resnet import ResNet_blocks")
                from Resnet import ResNet_blocks
                self.mu_resblock = ResNet_blocks(5, img_chns, img_res)
                self.sigma_resblock = ResNet_blocks(5, img_chns, img_res)
            

        self.mu_full_connect = nn.Sequential(
            nn.Linear(self.mu_resblock.dz, self.mu_resblock.dz),
        )
        self.sigma_full_connect = nn.Sequential(
            nn.Linear(self.mu_resblock.dz, self.mu_resblock.dz),
        )
        self.params = list(filter(self.__filter_func__,
                                self.parameters()))

    def __filter_func__(self, item):

        if id(item) not in list(map(id, [])):
            return True
        else:
            return False

    def forward(self, x):
        if self.training is True:
            mu_z_x = self.mu_resblock(x)
            sigma_z_x = self.sigma_resblock(x)
            gaussian_distribution = Normal(mu_z_x, sigma_z_x)
            sampled_LVs = gaussian_distribution.sample([self.gamma]).reshape(-1, self.mu_resblock.dz)
            return sampled_LVs, mu_z_x
        else:
            mu_z_x = self.mu_resblock(x)
            return mu_z_x

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.feature_dim, self.code_len
        )
def Y_Augmentation(y, gamma):
    ys = y.repeat(gamma, 1).reshape(-1)
    return ys
