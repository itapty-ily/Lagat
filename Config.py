import argparse

parser = argparse.ArgumentParser()
# -----------------
# Optimizer
# -----------------
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=384, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.01, help='adam: learning rate')
parser.add_argument('--_lr_decay', type=float, default=0.2, help='adam: learning rate decay factor')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--weight_decay', type=float, default=0.0002, help='weight decay')
# -----------------
# Network
# -----------------
parser.add_argument('--num_layer_stack_or_layers', type=int, default=18, help='number of layers in each block or the number of layers ')
parser.add_argument('--ResNet_block1', type=int, default=64, help='number of chns in block1, if num_layer_stack_or_layers=50/101/152, ResNet_block1=64*4')
parser.add_argument('--ResNet_block2', type=int, default=128, help='number of chns in block2, if num_layer_stack_or_layers=50/101/152, ResNet_block2=128*4')
parser.add_argument('--ResNet_block3', type=int, default=256, help='number of chns in block3, if num_layer_stack_or_layers=50/101/152, ResNet_block3=256*4')
parser.add_argument('--ResNet_blocks', type=str, default='ResNet_AA', help='The basis of Lagat, including ResNet')
# -----------------
# Algorithm
# -----------------
parser.add_argument('--a', type=float, default=1.50, help='factor for scaling means for all classes')
parser.add_argument('--b', type=float, default=1.00, help='factor for scaling variance for all classes')
parser.add_argument('--gamma', type=int, default=50, help='number of sampling new LVs')
parser.add_argument('--embed', type=str, default='LV', help="the mean of LV distribution, only 'LV' and 'WEV' are valid.")

# -----------------
# Data set
# -----------------
parser.add_argument('--train_num', type=int, default=1000, help='number of sample for training')
parser.add_argument('--dataset', type=str, default='TINYIMAGENET', help='data set: mnist, cifar10, norb, tinyimagenet')
# parser.add_argument('--gamma', type=int, default=50, help='number of sampling new LVs')
opt = parser.parse_args()
# print(opt)