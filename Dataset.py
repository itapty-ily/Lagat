"""Dataset.py

The conversion of date sets, including 'CIFAR10', 'CIFAR100', 'MNIST', 'STL10', 'TINYIMAGENET'
The conversion methods include generating the validation set and reducing the numbers of training samples.
The labels for each data set are dense.

The benchmark training set is split into two subsets, including the training set (70% of the training set) and validation set (30% of the training set).

itapty_ily@hotmail.com

09.15.20

"""

import os
import pickle
import time
import numpy as np
import struct
import torch.utils.data as data
from PIL import Image as Image
# import PIL
import torch
import cv2
import torch.nn as nn
from Config import opt
from sklearn.utils import shuffle
import scipy.misc as misc
import torchvision as tv
import torchvision.transforms as transforms
import random


class Conversion:
    """conversion operation class.

    Specify the date set and the number of training samples and save the converted date set in the new file.
    Besides, we must re-write the codes in torchvision.dataset.XXXX to point at the path to modified data set files.

    """
    def __init__(self, train_num=1000, dataset='mnist'):
        print(self.__class__.__name__, 'is initializing')
        dataset = dataset.upper()
        if dataset in ['CIFAR10', 'CIFAR100', 'MNIST', 'STL10', 'TINYIMAGENET']:
            self.dataset = dataset.upper()
        else:
            raise ValueError("Parameter dataset must be in 'CIFAR10', 'CIFAR100', 'MNIST', 'STL10'")
        self.train_num = train_num
        self._set_dataset_info()

    def _set_dataset_info(self):
        print(os.path.split(os.path.abspath('..')))
        self._cifar10_dir = os.path.abspath('.') + '/Dataset/Cifar/cifar-10-batches-py/'
        self._cifar100_dir = os.path.abspath('.')  + '/Dataset/Cifar/cifar-100-python/'
        self._mnist_dir = os.path.abspath('.')  + '/Dataset/Mnist/'
        self._stl10_dir = os.path.abspath('.')  + '/Dataset/STL-10/'
        self._tinyimagenet_dir = os.path.abspath('.')  + '/Dataset/TinyImagenet/'
        if self.dataset.upper() == 'CIFAR10':
            print('The data set is cifar10')
            self.img_size = [32, 32]
            self.class_num = 10
            self.img_channel = 3
            self.num_test_set = 10000
            if self.train_num > 50000:
                self.train_num = 50000
            self.num_train_set = int(self.train_num * 7 / 10)
            self.num_valid_set = int(self.train_num * 3 / 10)
            info = (self.dataset.upper(), self.num_train_set+self.num_valid_set, self.dataset.upper())
            self.validset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_valid.pkl' % info
            self.trainset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_tra.pkl' % info
            self.testset = os.path.abspath('.') + '/Data_Files/%s/%s_tes.pkl' % self.dataset.upper()
            if opt.embed == 'WEV':
                file = os.path.join(self._cifar10_dir, 'batches.meta')
                f = open(file, 'rb')
                self.label_list = pickle.load(f)['label_names']

        if self.dataset.upper() == 'CIFAR100':
            self.img_size = [32, 32]
            self.class_num = 100
            self.img_channel = 3
            self.num_test_set = 10000
            if self.train_num > 50000:
                self.train_num = 50000
            self.num_train_set = int(self.train_num * 7 / 10)
            self.num_valid_set = int(self.train_num * 3 / 10)
            print('The data set is cifar100')
            info = (self.dataset.upper(), self.num_train_set+self.num_valid_set, self.dataset.upper())
            self.validset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_valid.pkl' % info
            self.trainset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_tra.pkl' % info
            self.testset = os.path.abspath('.') + '/Data_Files/%s/%s_tes.pkl' % self.dataset.upper()
            if opt.embed == 'WEV':
                file = os.path.join(self._cifar100_dir, 'meta')
                f = open(file, 'rb')
                self.label_list = pickle.load(f)['fine_label_names']
        if self.dataset.upper() == 'MNIST':
            print('The data set is mnist')
            self.img_size = [28, 28]
            self.class_num = 10
            self.img_channel = 1
            self.num_test_set = 10000
            if self.train_num > 60000:
                self.train_num = 60000
            self.num_train_set = int(self.train_num * 7 / 10)
            self.num_valid_set = int(self.train_num * 3 / 10)
            info = (self.dataset.upper(), self.num_train_set+self.num_valid_set, self.dataset.upper(),)
            self.validset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_valid.pkl' % info
            self.trainset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_tra.pkl' % info
            self.testset = os.path.abspath('.') + '/Data_Files/%s/%s_tes.pkl' % self.dataset.upper()
            self.label_list = ['0', '1', '2', '3', '4',
                          '5', '6', '7', '8', '9']
        if self.dataset.upper() == 'STL10':
            print('The data set is STL10')
            self.img_size = [96, 96]
            self.class_num = 10
            self.img_channel = 3
            self. num_test_set = 8000
            if self.train_num > 5000:
                self.train_num = 5000
            self.num_train_set = int(self.train_num * 7 / 10)
            self.num_valid_set = int(self.train_num * 3 / 10)
            info = (self.dataset.upper(), self.num_train_set+self.num_valid_set, self.dataset.upper())
            self.validset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_valid.pkl' % info
            self.trainset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_tra.pkl' % info
            self.testset = os.path.abspath('.') + '/Data_Files/%s/%s_tes.pkl' % self.dataset.upper()
            self.label_list = ['airplane', 'bird', 'car', 'cat', 'deer',
                          'dog', 'horse', 'monkey', 'ship', 'truck']
        if self.dataset.upper() == 'TINYIMAGENET':
            print('The data set is TINYIMAGENET')
            self.img_size = [64, 64]
            self.class_num = 200
            self.train_num = self.train_num * self.class_num
            self.img_channel = 3
            self.num_test_set = 10000
            # self.test_files = os.listdir(os.path.join(self._tinyimagenet_dir + 'test', 'images'))
            if self.train_num > 100000:
                self.train_num = 100000
            self.num_train_set = self.train_num
            self.num_valid_set = 10000
            # print(self.train_num, self.num_train_set, self.num_valid_set)
            info = (self.dataset.upper(), self.train_num, self.dataset.upper())
            self.validset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_valid.pkl' % info
            self.trainset = os.path.abspath('.') + '/Data_Files/%s/%d_%s_tra.pkl' % info
            self.testset = os.path.abspath('.') + '/Data_Files/%s/%s_tes.pkl' % self.dataset.upper()
            if opt.embed == 'WEV':
                self.label_list = list()
                self.wnids = open(os.path.join(self._tinyimagenet_dir, 'wnids.txt')).read().split('\n')
                words = open(os.path.join(self._tinyimagenet_dir, 'words.txt')).read().split('\n')
                for wnid in self.wnids:
                    for word in words:
                        if wnid in word:
                            self.label_list.append(word.split('\t')[1])

    def _cv_cifar(self, dataset_path, files_list, ins_num=1000):
        """Convert a list of imgs in a py file to TFrecorder.

        Convert a batch of imgs in a py file to TFrecorder.

        Args:
            dataset_path: the path to save recorder file.
            files_list: the path to py files, save in a list.
        """
        img_counter = list()
        for i in range(self.class_num):
            img_counter.append(0)
        print('img_counter:', img_counter)
        total_counter = 0
        dataset_dict = dict()
        dataset_dict['labels'] = []
        dataset_dict['data'] = []
        for f in files_list:
            with open(f, 'rb')as file:
                file_dict = pickle.load(file, encoding='latin1')
                # print(list(file_dict.keys()))
                imgs = file_dict['data']
                labels = None
                if self.dataset.upper() == 'CIFAR10':
                    labels = file_dict['labels']
                elif self.dataset.upper() == 'CIFAR100':
                    labels = file_dict['fine_labels']
                print('type(imgs):', type(imgs))
                print('imgs.shape:', imgs.shape)
                print('type(labels):', type(labels))
                print('label.max VS label.min: ', np.max(labels), np.min(labels))
                imgs_num = imgs.shape[0]
                starttime = time.time()
                for i in range(imgs_num):
                    img = imgs[i]
                    label = labels[i]
                    if i % 1000 == 0:
                        endtime = time.time()
                        print('Convert 1000 images cost %4.3fs.' % (endtime - starttime))
                        print('\tLoading...', os.path.split(f)[-1])
                        starttime = endtime
                        print('\t%4.2f percent finished.' % (sum(img_counter)/ins_num*100))
                        print('\timg_counter:', img_counter)

                    if img is not None:
                        flag_train = ('tra' in dataset_path) and (0 <= img_counter[int(label)] < int(ins_num / self.class_num))
                        flag_eval = ('valid' in dataset_path) and (self.num_train_set < total_counter + i) and (img_counter[int(label)] < int(ins_num / self.class_num))
                        if flag_train or ('tes' in dataset_path) or flag_eval:
                            img_counter[int(label)] += 1
                            dataset_dict['labels'].append(label)
                            dataset_dict['data'].append(img)
                        else:
                            pass
                    else:
                        raise ValueError('\t', i, '-th images is None')

                total_counter += imgs_num
        temp = os.path.join(os.path.join(os.path.abspath('.'), 'Data_files'), self.dataset)
        if not os.path.exists(temp):
            os.makedirs(temp)
        dataset_dict['data'] = np.array(dataset_dict['data'])
        dataset_dict['data'] = dataset_dict['data'].reshape((-1, 3, 32, 32))
        dataset_dict['data'] = dataset_dict['data'].transpose((0, 2, 3, 1))
        output = open(dataset_path, 'wb')
        pickle.dump(dataset_dict, output)
        output.close()

    def _cv_mnist(self, dataset_path, files_list, ins_num=1000):
        """Convert a list of imgs in a py file to TFrecorder.

        Convert a batch of imgs in a py file to TFrecorder.

        Args:
            dataset_path: the path to save recorder file.
            files_list: the path to py files, save in a list.
        """
        image_path, label_path = files_list
        img_counter = list()
        for i in range(self.class_num):
            img_counter.append(0)
        print('img_counter:', img_counter)
        binfile = open(image_path, 'rb')
        buffers = binfile.read()
        head = struct.unpack_from('>IIII', buffers, 0)
        print("head,", head)
        offset = struct.calcsize('>IIII')
        img_num = head[1]
        width = head[2]
        height = head[3]
        bits = img_num * width * height
        bits_string = '>' + str(bits) + 'B'
        imgs = struct.unpack_from(bits_string, buffers, offset)
        binfile.close()
        imgs = np.reshape(imgs, [img_num, width, height])
        print(np.max(imgs), np.min(imgs))
        print('convert ', image_path, 'finished')
        print('converting ', label_path)
        binfile = open(label_path, 'rb')
        buffers = binfile.read()
        head = struct.unpack_from('>II', buffers, 0)
        print("head,", head)
        img_num = head[1]
        offset = struct.calcsize('>II')
        num_string = '>' + str(img_num) + "B"
        labels = struct.unpack_from(num_string, buffers, offset)
        binfile.close()
        labels = np.reshape(labels, [img_num])
        print('label.max VS label.min: ', np.max(labels), np.min(labels))
        print('converting ', label_path, 'finished')
        dataset_dict = dict()
        dataset_dict['labels'] = []
        dataset_dict['data'] = []
        imgs_num = imgs.shape[0]
        starttime = time.time()
        for i in range(imgs_num):
            img = imgs[i]
            # print(img.shape)
            label = labels[i]
            if img is not None:
                flag_train = ('tra' in dataset_path) and (0 <= img_counter[int(label)] < int(ins_num / self.class_num))
                flag_eval = ('valid' in dataset_path) and (self.num_train_set < i) and (img_counter[int(label)] < int(ins_num / self.class_num))
                if flag_train or ('tes' in dataset_path) or flag_eval:
                    img_counter[int(label)] += 1
                    dataset_dict['labels'].append(label)
                    dataset_dict['data'].append(img)
                    if i % 1000 == 0:
                        print('\n%0.2f-percent of images completed' % (sum(img_counter) / ins_num * 100))
                        endtime = time.time()
                        print('Convert 1000 images cost %4.3fs.' % (endtime - starttime))
                        starttime = endtime
                        print('img_counter:', img_counter)
                        # print('type(imgs):', type(imgs))
                        # print('imgs.shape:', imgs.shape)
                        # print('type(labels):', type(labels))
                        print('The label of the image is %f' % label)
                        show_img = np.reshape(np.array(img, dtype=np.uint8), (28, 28))
                        im = cv2.resize(show_img, (280, 280))
                        show_img = cv2.putText(im, str(label), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                        cv2.imshow('Resize', show_img)
                        k = cv2.waitKey(1) & 0xFF
                        if k == 27:
                            cv2.destroyAllWindows()
                        time.sleep(0.2)
                else:
                    pass
            else:
                raise ValueError('\t', i, '-th images is None')

        temp = os.path.join(os.path.join(os.path.abspath('.'), 'Data_files'), self.dataset)
        if not os.path.exists(temp):
            os.makedirs(temp)
        dataset_dict['data'] = np.array(dataset_dict['data'])
        dataset_dict['data'] = dataset_dict['data'].reshape((-1, 28, 28, 1))
        # dataset_dict['data'] = dataset_dict['data'].transpose((0, 2, 3, 1))
        output = open(dataset_path, 'wb')
        pickle.dump(dataset_dict, output)
        output.close()
        cv2.destroyAllWindows()

    def _cv_stl10(self, dataset_path, files_list, ins_num=1000):
        """Convert a list of imgs in a py file to TFrecorder.

        Convert a batch of imgs in a py file to TFrecorder.

        Args:
            dataset_path: the path to save recorder file.
            files_list: the path to py files, save in a list.
        """
        image_path, label_path = files_list
        img_counter = list()
        for i in range(self.class_num):
            img_counter.append(0)
        print('img_counter:', img_counter)
        images_file = open(image_path, 'rb')
        labels_file = open(label_path, 'rb')
        images_code = np.fromfile(images_file, dtype=np.uint8)
        images = np.reshape(images_code, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 1, 3, 2))
        labels = np.fromfile(labels_file, dtype=np.uint8)
        print(images.shape)
        print(labels.shape)
        dataset_dict = dict()
        dataset_dict['labels'] = []
        dataset_dict['data'] = []
        assert len(images) == len(labels)
        print('label.max VS label.min: ', np.max(labels), np.min(labels))
        starttime = time.time()
        for i in range(len(images)):
            img = images[i]
            label = int(labels[i]-1)
            if img is not None:
                flag_train = ('tra' in dataset_path) and (0 <= img_counter[int(label)] < int(ins_num / self.class_num))
                flag_eval = ('valid' in dataset_path) and (self.num_train_set < i) and (img_counter[int(label)] < int(ins_num / self.class_num))
                if flag_train or ('tes' in dataset_path) or flag_eval:
                    img_counter[int(label)] += 1
                    dataset_dict['labels'].append(label)
                    dataset_dict['data'].append(img)
                    if i % 1000 == 0:
                        endtime = time.time()
                        print('Convert 1000 images cost %4.3fs.' % (endtime - starttime))
                        starttime = endtime
                        print('\t%4.2f percent finished.' % (sum(img_counter)/ins_num*100))
                        print('\timg_counter:', img_counter)
                else:
                    pass
            else:
                raise ValueError('\t', i, '-th images is None')
        temp = os.path.join(os.path.join(os.path.abspath('.'), 'Data_files'), self.dataset)
        if not os.path.exists(temp):
            os.makedirs(temp)
        dataset_dict['data'] = np.array(dataset_dict['data'])
        dataset_dict['data'] = dataset_dict['data'].reshape((-1, 3, 96, 96))
        dataset_dict['data'] = dataset_dict['data'].transpose((0, 2, 3, 1))
        output = open(dataset_path, 'wb')
        pickle.dump(dataset_dict, output)
        output.close()

    def _cv_tinyimagenet(self, dataset_path, files_list, ins_num=1000):
        """Convert a list of imgs in a py file to TFrecorder.

        Convert a batch of imgs in a py file to TFrecorder.

        Args:
            dataset_path: the path to save recorder file.
            files_folder: the path to py files, save in a list.
        """
        files_folder = files_list[0]
        assert len(self.wnids) == self.class_num
        dataset_dict = dict()
        dataset_dict['labels'] = []
        dataset_dict['data'] = []
        if 'tra' in dataset_path:
            img_counter = list()
            for i in range(self.class_num):
                img_counter.append(0)
            print('img_counter:', img_counter)
            for item in os.listdir(files_folder):
                temp = os.path.join(os.path.join(files_folder, item), 'images')
                image_files = os.listdir(temp)
                random.seed(0)
                random.shuffle(image_files)
                if item in self.wnids:
                    label = self.wnids.index(item)
                else:
                    raise ValueError('The label of %s is not in wnids.txt' % item)
                for image_file in image_files:
                    image = cv2.imread(os.path.join(temp, image_file))
                    img_counter[label] += 1
                    dataset_dict['labels'].append(label)
                    dataset_dict['data'].append(image)
                    if img_counter[label] >= int(self.num_train_set / self.class_num):
                        break
                print('%4.2f percent of images is converted.' % (sum(img_counter)/self.num_train_set*100))
                print('img_counter:', img_counter)
            dataset_dict['data'] = np.array(dataset_dict['data'])
        elif 'valid' in dataset_path:
            val_annotations = open(os.path.join(files_folder, 'val_annotations.txt')).read().split('\n')
            i = 0
            assert len(val_annotations) == self.num_valid_set
            for val_annotation in val_annotations:
                # print(val_annotation)
                image_file, label_name = val_annotation.split('\t')[0:2]
                temp = os.path.join(files_folder, 'images')
                image = cv2.imread(os.path.join(temp, image_file))
                label = self.wnids.index(label_name)
                dataset_dict['labels'].append(label)
                dataset_dict['data'].append(image)
                i += 1
                if i % 100 == 0:
                     print('%4.2f percent of images is converted.' % (i/self.num_valid_set*100))
        # else:
        #     i = 0
        #     temp = os.path.join(files_folder, 'images')
        #     assert len(self.test_files) == self.num_test_set
        #     for image_file in self.test_files:
        #         image = cv2.imread(os.path.join(temp, image_file))
        #         dataset_dict['data'].append(image)
        #         dataset_dict['labels'].append(0)
        #         i += 1
        #         if i % 100 == 0:
        #              print('%4.2f percent of images is converted.' % (i/self.num_test_set*100))
        temp = os.path.join(os.path.join(os.path.abspath('.'), 'Data_files'), self.dataset)
        if not os.path.exists(temp):
            os.makedirs(temp)
        output = open(dataset_path, 'wb')
        pickle.dump(dataset_dict, output)
        output.close()

    def dataset_to_pkl(self):
        """convert the cifar dataset bin files to TFRecoder with labels.

        convert the cifar dataset bin files to TFRecoder with labels.

        Return:
            None
        """
        if self.dataset.upper() == 'cifar10'.upper():
            if not os.path.exists(self.trainset):
                print('Converting training data')
                self._cv_cifar(self.trainset,
                               files_list=[self._cifar10_dir+'data_batch_%d' % (i+1) for i in range(5)],
                               ins_num=self.num_train_set)
            if not os.path.exists(self.validset):
                print('Converting validation data')
                self._cv_cifar(self.validset,
                               files_list=[self._cifar10_dir+'data_batch_%d' % (i+1) for i in range(5)],
                               ins_num=self.num_valid_set)
            if not os.path.exists(self.testset):
                print('Converting testing data')
                self._cv_cifar(self.testset, files_list=[self._cifar10_dir + 'test_batch'])
        elif self.dataset.upper() == 'cifar100'.upper():
            if not os.path.exists(self.trainset):
                print('Converting training data')
                self._cv_cifar(self.trainset, files_list=[self._cifar100_dir + 'train'],
                               ins_num=self.num_train_set)
            if not os.path.exists(self.validset):
                print('Converting validation data')
                self._cv_cifar(self.validset, files_list=[self._cifar100_dir + 'train'],
                               ins_num=self.num_valid_set)
            if not os.path.exists(self.testset):
                print('Converting testing data')
                self._cv_cifar(self.testset, files_list=[self._cifar100_dir + 'test'])
        elif self.dataset.upper() == 'mnist'.upper():
            if not os.path.exists(self.trainset):
                self._cv_mnist(self.trainset, [self._mnist_dir + 'train-images.idx3-ubyte',
                                               self._mnist_dir + 'train-labels.idx1-ubyte'],
                               ins_num=self.num_train_set)
            if not os.path.exists(self.validset):
                self._cv_mnist(self.validset, [self._mnist_dir + 'train-images.idx3-ubyte',
                                               self._mnist_dir + 'train-labels.idx1-ubyte'],
                               ins_num=self.num_valid_set)
            if not os.path.exists(self.testset):
                self._cv_mnist(self.testset, [self._mnist_dir + 't10k-images.idx3-ubyte',
                                              self._mnist_dir + 't10k-labels.idx1-ubyte'])
        elif self.dataset.upper() == 'stl10'.upper():
            if not os.path.exists(self.trainset):
                self._cv_stl10(self.trainset, [self._stl10_dir + 'train_X.bin',
                                               self._stl10_dir + 'train_y.bin'],
                               ins_num=self.num_train_set)
            if not os.path.exists(self.validset):
                self._cv_stl10(self.validset, [self._stl10_dir + 'train_X.bin',
                                               self._stl10_dir + 'train_y.bin'],
                               ins_num=self.num_valid_set)
            if not os.path.exists(self.testset):
                self._cv_stl10(self.testset, [self._stl10_dir + 'test_X.bin',
                                              self._stl10_dir + 'test_y.bin'])
        elif self.dataset.upper() == 'tinyimagenet'.upper():
            if not os.path.exists(self.trainset):
                self._cv_tinyimagenet(self.trainset, [
                    self._tinyimagenet_dir + 'train'],
                               ins_num=self.num_train_set)
            if not os.path.exists(self.validset):
                self._cv_tinyimagenet(self.validset, [
                    self._tinyimagenet_dir + 'val'],
                               ins_num=self.num_valid_set)
            if not os.path.exists(self.testset):
                self._cv_tinyimagenet(self.testset, [
                    self._tinyimagenet_dir + 'test'])
        else:
            raise ValueError('The input argues must be cifar10, cifar100, stl10, tinyimagenet or mnist')

    def get_embeds(self, label_list=None):
        dataset_folder = os.path.split(self.trainset)[0]
        word_embed_file = os.path.join(dataset_folder, '%s_%d.pkl' % (opt.ResNet_blocks, self.train_num))
            
        if not os.path.exists(word_embed_file):
            file = open(self.trainset, 'rb')
            temp = pickle.load(file)
            file.close()
            self.images = temp['data']
            if 'ResNet' == opt.ResNet_blocks:
                print("The prior LVs are sampled from ResNet")
                if os.path.exists(os.path.join(os.path.abspath('.'), 'ResNetB_Model')) is False:
                    raise ValueError('The ResNetB model dose not exits. Please run ResNetB.py first.')
                if opt.dataset.upper() == 'TINYIMAGENET'.upper():
                    from Resnet_imagenet import ResNet_blocks
                    z_x = ResNet_blocks(34, self.img_channel, self.img_size[0])
                else:
                    from Resnet import ResNet_blocks
                    z_x = ResNet_blocks(5, self.img_channel, self.img_size[0])
                z_x.load_state_dict(torch.load(os.path.join(os.path.join(os.path.abspath('.'),
                                                               'ResNetB_Model'),
                                                  '%s_%d_z_x' %
                                   (opt.dataset.upper(), opt.train_num))))
                z_x.eval()
            if opt.ResNet_blocks == 'ResNet_AA':
                print("The prior LVs are sampled from ResNet_AA")
                if os.path.exists(os.path.join(os.path.abspath('.'), 'AA_Model')) is False:
                    raise ValueError('The AA model dose not exits. Please run AutoAugment.py first.')
                if self.dataset.upper() in ['TINYIMAGENET'.upper()]:
                    from Resnet_imagenet import ResNet_blocks, ResNet_cf
                    z_x = ResNet_blocks(18, self.img_channel, self.img_size[0])
                elif self.dataset.upper() in ['MNIST'.upper(), 'CIFAR10'.upper()]:
                    from Resnet import ResNet_blocks, ResNet_cf
                    z_x = ResNet_blocks(5, self.img_channel, self.img_size[0])
                z_x.load_state_dict(torch.load(
                    os.path.join(os.path.join(os.path.abspath('.'), 'AA_Model'),\
                        '%s_%d_z_x' % (opt.dataset.upper(), opt.train_num))), strict=False)
                z_x.eval()


            self.word_embeds = list()
            i = 0
            z_x.cuda()
            if 'AA' in opt.ResNet_blocks:
                img_transformer = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                        (0.2023, 0.1994, 0.2010)),
                ])
                if self.dataset.upper() in ['MNIST'.upper()]:
                    img_transformer = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914],
                                            [0.2023]),
                    ])
            else:
                img_transformer = tv.transforms.ToTensor()
            for image in self.images:
                # print(image.dtype)
                image = Image.fromarray(image.squeeze())
                image = img_transformer(image).unsqueeze(0).cuda()
                LV = z_x(image)
                if type(LV) is tuple:
                    LV = LV[-1]
                self.word_embeds.append(LV.detach().cpu())
                i += 1
                if i % 100 == 0:
                    print('%d LVs has been transformed' % i)
            print('Totally %d LVs has been transformed' % i)
            temp = os.path.join(os.path.join(os.path.abspath('.'), 'Data_files'), self.dataset)
            if not os.path.exists(temp):
                os.makedirs(temp)
            output = open(word_embed_file, 'wb')
            pickle.dump(self.word_embeds, output)
            output.close()

        else:
            word_embed_file = open(word_embed_file, 'rb')
            self.word_embeds = pickle.load(word_embed_file)
            word_embed_file.close()



class Mydataset(data.Dataset, Conversion):
    """Definition of my modified dataset including the training, validation and test set.

    Notes that the dataset is downloaded by user manually, and the dataset is saved in the following absolute path:
        cifar10_dir = os.path.split(os.path.abspath('..'))[0] + '/Dataset/Cifar/cifar-10-batches-py/'
        cifar100_dir = os.path.split(os.path.abspath('..'))[0] + '/Dataset/Cifar/cifar-100-python/'
        mnist_dir = os.path.split(os.path.abspath('..'))[0] + '/Dataset/Mnist/'
        stl10_dir = os.path.split(os.path.abspath('..'))[0] + '/Dataset/STL-10/'

    Args:
        train_num (int): the number of samples from training set, including training set and validation set:
                  training set = train_num % 70%
                  validation set = train_num % 30%
        dataset (string): specify the data set to be used for our algorithm.
        data_flag (string): specify which data set to be loaded,
            including 'train', 'test' or 'valid'.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.


    """
    def __init__(self, train_num=1000, dataset='stl10', data_flag='train',
                 transform=None, target_transform=None, embed=False):
        print(self.__class__.__name__, 'is initializing')
        print(Conversion.__name__, 'is initializing')
        Conversion.__init__(self, train_num=train_num, dataset=dataset)
        self.transform = transform
        self.target_transform = target_transform
        self.train = data_flag.upper()
        self.data_flag = data_flag
        self.embed = embed
        if self.data_flag == 'train':
            self.data_file = self.trainset
        elif self.data_flag == 'valid':
            self.data_file = self.validset
        elif self.data_flag == 'test':
            self.data_file = self.testset
        if not os.path.exists(self.data_file):
            print('The data set files are not exist, we call dataset_to_pkl() to convert them.')
            self.dataset_to_pkl()
        if self.embed is True:
            if opt.embed == 'WEV':
                self.get_embeds(self.label_list)
            else:
                self.get_embeds()
        file = open(self.data_file, 'rb')
        temp = pickle.load(file)
        self.images = temp['data']
        self.labels = np.array(temp['labels'])
        file.close()
        if self.data_flag == 'train':
            self.train_data = self.images
            self.train_labels = self.labels

        elif self.data_flag == 'valid':
            self.valid_data = self.images
            self.valid_labels = self.labels

        elif self.data_flag == 'test':
            self.test_data = self.images
            self.test_labels = self.labels
        """
        The images in the pkl files is correct, we can use the following codes to confirm it:
        img = self.images[1000]
        label = self.labels[1000]
        show_img = np.reshape(np.array(img, dtype=np.uint8), (28, 28))
        im = cv2.resize(show_img, (280, 280))
        show_img = cv2.putText(im, str(label), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow('Resize', show_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
        time.sleep(5)
        """

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        """
        print(type(label)):
             for MNIST: <class 'numpy.int32'> incorrect
        """
        if self.dataset == 'MNIST':
            # print(image.shape): (28, 28)
            # print(np.max(image)): 255
            image = np.array(image, dtype=np.byte).squeeze()
            image = Image.fromarray(image, 'L')
            label = int(label)
        elif self.dataset == 'STL10':
            # print(image.shape): (96, 96, 3)
            # print(np.max(image)): 255
            image = Image.fromarray(image)
            label = int(label)
        elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            # print(image.shape): (32, 32, 3)
            # print(np.max(image)): 255
            image = Image.fromarray(image)
            label = int(label)
        elif self.dataset == 'TINYIMAGENET':
            # print(image.shape): (64, 64, 3)
            # print(np.max(image)): 256
            image = Image.fromarray(image)
            label = int(label)

        if self.transform is not None:
            for item in self.transform:
                image = item(image)

        if self.target_transform is not None:
            label = self.target_transform(label)
        """
        print(image.shape):
            for MNIST: torch.Size([1, 28, 28]) correct
        """
        if self.embed is True:
            if opt.embed is not 'LV':
                word_embed = self.word_embeds[label]
            else:
                word_embed = self.word_embeds[index]
            return image, label, word_embed

        else:
            return image, label


    def __len__(self):
        if self.data_flag == 'train':
            if len(self.images) == len(self.labels) == self.num_train_set:
                return self.num_train_set
            else:
                print('len(self.images): ', len(self.images))
                print('len(self.labels): ', len(self.labels))
                print('self.num_train_set:: ', self.num_train_set)
                raise ValueError('The length of data and label is incorrect.')
        elif self.data_flag == 'valid':
            if len(self.images) == len(self.labels) == self.num_valid_set:
                return self.num_valid_set
            else:
                raise ValueError('The length of data and label is incorrect.')
        elif self.data_flag == 'test':
            if len(self.images) == len(self.labels) == self.num_test_set:
                return self.num_test_set
            else:
                raise ValueError('The length of data and label is incorrect.')


if __name__ == '__main__':
    pass











