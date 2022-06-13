import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from PIL import Image, ImageFilter
import utils.utils as utils
import json
from tqdm import tqdm
import scipy.io as sio

Datasets = {
    'Pose_300W_LP': {
        'data_dir': './datasets/300W_LP',
        'filename_list': [
            './datasets/300W_LP/files.txt'
        ]
    },
    'BIWI': {
        'data_dir': './datasets/BIWI',
        'filename_list': [
            './datasets/BIWI/BIWI_train.txt',
            './datasets/BIWI/BIWI_test.txt'
        ]
    },
    'Pose_300W_LP_masked': {
        'data_dir': './datasets/300W_LP_masked',
        'filename_list': [
            './datasets/300W_LP_masked/files.txt'
        ]
    },
    'BIWI_masked': {
        'data_dir': './datasets/BIWI_masked',
        'filename_list': [
            './datasets/BIWI_masked/train.txt',
            './datasets/BIWI_masked/test.txt'
        ]
    },
    'AFLW2000': {
        'data_dir': './datasets/AFLW2000',
        'filename_list': [
            './datasets/AFLW2000/files.txt'
        ]
    },
    'AFLW2000_masked': {
        'data_dir': './datasets/AFLW2000_masked',
        'filename_list': [
            './datasets/AFLW2000_masked/files.txt'
        ]
    }
}


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    # print(file_path)
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines

    
class AFLW2000(Dataset):
    def __init__(self, dataset_name, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = Datasets[dataset_name]['data_dir']
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_path = Datasets[dataset_name]['filename_list'][0]
        filename_list = get_list_from_filenames(filename_path)
        # filename_list = ['image00777', 'image01292', 'image01991', 'image02530']

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        img = np.array(img)
        img = img / 255.

        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0,:])
        y_min = min(pt2d[1,:])
        x_max = max(pt2d[0,:])
        y_max = max(pt2d[1,:])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        # img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0]# * 180 / np.pi
        yaw = pose[1] #* 180 / np.pi
        roll = pose[2]# * 180 / np.pi
     
        R = utils.get_R(yaw, pitch, roll)

        labels = torch.FloatTensor([yaw, pitch, roll])


        # if self.transform is not None:
        #     img = self.transform(img)

        return img, torch.ones((1, 15)), torch.FloatTensor(R), labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length


class AFLW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in radians
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        pose = [float(line[1]), float(line[2]), float(line[3])]
        # And convert to degrees.
        yaw = pose[0] * 180 / np.pi
        pitch = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Fix the roll in AFLW
        roll *= -1
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # train: 18,863
        # test: 1,966
        return self.length

class AFW(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.txt', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)
        img_name = self.X_train[index].split('_')[0]

        img = Image.open(os.path.join(self.data_dir, img_name + self.img_ext))
        img = img.convert(self.image_mode)
        txt_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # We get the pose in degrees
        annot = open(txt_path, 'r')
        line = annot.readline().split(' ')
        yaw, pitch, roll = [float(line[1]), float(line[2]), float(line[3])]

        # Crop the face loosely
        k = 0.32
        x1 = float(line[4])
        y1 = float(line[5])
        x2 = float(line[6])
        y2 = float(line[7])
        x1 -= 0.8 * k * abs(x2 - x1)
        y1 -= 2 * k * abs(y2 - y1)
        x2 += 0.8 * k * abs(x2 - x1)
        y2 += 1 * k * abs(y2 - y1)

        img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # Around 200
        return self.length

class BIWI(Dataset):
    def __init__(self, preproc=None, image_mode='RGB', train_mode=True):
        self.data_dir = Datasets['BIWI']['data_dir']
        self.preproc = preproc

        filename_path = Datasets['BIWI']['filename_list'][0 if train_mode else 1]
        self.img_list = np.loadtxt(filename_path, dtype=str)

        self.img_dir = './faces_0'
        self.det_anno_dir = './datasets/BIWI/BIWI_det_annotations'

        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(self.img_list)

    def get_label(self, img_name):
        json_file = {}
        # print(self.img_list[index])
        with open(os.path.join(self.det_anno_dir, img_name.replace('.png', '.json')), "r") as st_json:
            json_file = json.load(st_json)

        pose = json_file['pose']
        roll = pose[2] / 180 * np.pi
        yaw = pose[0] / 180 * np.pi
        pitch = pose[1] / 180 * np.pi
        pose_labels = np.array([yaw, pitch, roll])
        # R = utils.get_R(pitch, yaw, roll)

        labels = []
        for key in list(json_file.keys())[:-1]:
            labels.append(json_file[key])

        annotations = np.zeros((0, 15))
        max_inter = 0

        mask_path = os.path.join(self.data_dir, 'head_pose_masks', img_name.replace('rgb', 'depth_mask'))
        if not os.path.exists(mask_path):
            return annotations, pose_labels

        mask = Image.open(mask_path)
        mask = np.array(mask) / 255.
        h, w = mask.shape
        y, x = np.where(mask == 1)
        x1, x2 = max(x.min() - 1, 0), min(x.max() + 1, w)
        y1, y2 = max(y.min() - 1, 0), min(y.max() + 1, h)
        bbox = np.array([x1, y1, x2, y2])


        for idx, label in enumerate(labels):
            if label['score'] < 0.95:
                continue

            h = min(y2, label['facial_area'][3]) - max(y1, label['facial_area'][1])
            w = min(x2, label['facial_area'][2]) - max(x1, label['facial_area'][0])
            inter = 0
            if h > 0 and w > 0:
                inter = h * w
            if max_inter > inter:
                continue
            max_inter = inter

            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label['facial_area'][0]  # x1
            annotation[0, 1] = label['facial_area'][1]  # y1
            annotation[0, 2] = label['facial_area'][2]  # x2
            annotation[0, 3] = label['facial_area'][3]  # y2
            # landmarks
            # left_eye, right_eye, nose, mouth_left, mouth_right
            annotation[0, 4] = label['landmarks']['left_eye'][0]  # l0_x
            annotation[0, 5] = label['landmarks']['left_eye'][1]  # l0_y
            annotation[0, 6] = label['landmarks']['right_eye'][0]  # l1_x
            annotation[0, 7] = label['landmarks']['right_eye'][1]  # l1_y
            annotation[0, 8] = label['landmarks']['nose'][0]  # l2_x
            annotation[0, 9] = label['landmarks']['nose'][1]  # l2_y
            annotation[0, 10] = label['landmarks']['mouth_left'][0]  # l3_x
            annotation[0, 11] = label['landmarks']['mouth_left'][1]  # l3_y
            annotation[0, 12] = label['landmarks']['mouth_right'][0]  # l4_x
            annotation[0, 13] = label['landmarks']['mouth_right'][1]  # l4_y

            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
                continue
            else:
                annotation[0, 14] = 1

            annotations = annotation
        target = np.array(annotations)

        return target, pose_labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.img_dir, self.img_list[index])).convert('RGB')
        img = np.array(img) / 255.

        img_name = self.img_list[index]
        target, pose_label = self.get_label(img_name)
        R = utils.get_R(*pose_label)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        # if self.train_mode:
        #     # Flip?
        #     rnd = np.random.random_sample()
        #     if rnd < 0.5:
        #         yaw = -yaw
        #         roll = -roll
        #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        #
        #     # Blur?
        #     rnd = np.random.random_sample()
        #     if rnd < 0.05:
        #         img = img.filter(ImageFilter.BLUR)
        #
        # R = utils.get_R(pitch, yaw, roll)
        #
        # labels = torch.FloatTensor([yaw, pitch, roll])
        #
        # if self.transform is not None:
        #     img = self.transform(img)
        #
        #     # mask = self.transform.transforms[0](mask)
        #
        # # Get target tensors
        # cont_labels = torch.FloatTensor([yaw, pitch, roll])
        #
        # h, w = img.shape[1:]
        # mask = mask.resize((h, w))
        # mask = np.array(mask) / 255.
        # y, x = np.where(mask == 1)
        # x1, x2 = max(x.min() - 1, 0), min(x.max() + 1, w)
        # y1, y2 = max(y.min() - 1, 0), min(y.max() + 1, h)
        # bbox = torch.FloatTensor([x1, y1, x2 - x1, y2 - y1])
        #
        # # import matplotlib.patches as patches
        # # fig, ax = plt.subplots(1, 2)
        # # rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        # # ax[0].imshow(img.permute(1, 2, 0).numpy())
        # # ax[0].add_patch(rect)
        # # ax[1].imshow(mask)
        # # ax[1].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none'))
        # # plt.show()


        # from test import get_det_landm_image, get_hpe_image
        # show_img = (img * 255).astype(np.uint8).copy()
        # # show_img = get_det_landm_image(show_img, target[:, :5], target[:, 4:-1])
        # show_img = get_hpe_image(show_img, target[:, :5], pose_label.reshape(1, 3) * 180 / np.pi)
        # plt.imshow(show_img)
        # plt.show()


        # img = torch.from_numpy(img).float()
        # target = torch.from_numpy(target).float()
        # R = torch.from_numpy(R).unsqueeze(0).float()
        img = img.astype(np.float32)
        target = target.astype(np.float32)
        R = np.expand_dims(R, 0).astype(np.float32)

        if self.train_mode:
            return img, target, R, self.img_list[index]
        else:
            pose_label = pose_label.astype(np.float32)
            return img, target, R, pose_label, self.img_list[index]

    def __len__(self):
        # 15,667
        return self.length

class BIWI_masked(Dataset):
    def __init__(self, preproc=None, image_mode='RGB', train_mode=True):
        self.data_dir = Datasets['BIWI_masked']['data_dir']
        self.preproc = preproc

        filename_path = Datasets['BIWI_masked']['filename_list'][0 if train_mode else 1]
        self.img_list = np.loadtxt(filename_path, dtype=str, delimiter='\n')

        self.img_dir = './faces_0'

        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(self.img_list)

    def get_label(self, label_path, mask_path):
        json_file = {}
        with open(label_path, "r") as st_json:
            json_file = json.load(st_json)

        pose = json_file['pose']
        roll = pose[2] / 180 * np.pi
        yaw = pose[0] / 180 * np.pi
        pitch = pose[1] / 180 * np.pi
        pose_labels = np.array([yaw, pitch, roll])
        # R = utils.get_R(pitch, yaw, roll)

        labels = []
        for key in list(json_file.keys())[:-1]:
            labels.append(json_file[key])

        annotations = np.zeros((0, 15))
        max_inter = 0

        mask = Image.open(mask_path)
        mask = np.array(mask) / 255.
        h, w = mask.shape
        y, x = np.where(mask == 1)
        x1, x2 = max(x.min() - 1, 0), min(x.max() + 1, w)
        y1, y2 = max(y.min() - 1, 0), min(y.max() + 1, h)
        bbox = np.array([x1, y1, x2, y2])


        for idx, label in enumerate(labels):
            if label['score'] < 0.95:
                continue

            h = min(y2, label['facial_area'][3]) - max(y1, label['facial_area'][1])
            w = min(x2, label['facial_area'][2]) - max(x1, label['facial_area'][0])
            inter = 0
            if h > 0 and w > 0:
                inter = h * w
            if max_inter > inter:
                continue
            max_inter = inter

            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label['facial_area'][0]  # x1
            annotation[0, 1] = label['facial_area'][1]  # y1
            annotation[0, 2] = label['facial_area'][2]  # x2
            annotation[0, 3] = label['facial_area'][3]  # y2
            # landmarks
            # right_eye, left_eye, nose, mouth_right, mouth_left
            annotation[0, 4] = label['landmarks']['left_eye'][0]  # l0_x
            annotation[0, 5] = label['landmarks']['left_eye'][1]  # l0_y
            annotation[0, 6] = label['landmarks']['right_eye'][0]  # l1_x
            annotation[0, 7] = label['landmarks']['right_eye'][1]  # l1_y
            annotation[0, 8] = label['landmarks']['nose'][0]  # l2_x
            annotation[0, 9] = label['landmarks']['nose'][1]  # l2_y
            annotation[0, 10] = label['landmarks']['mouth_left'][0]  # l3_x
            annotation[0, 11] = label['landmarks']['mouth_left'][1]  # l3_y
            annotation[0, 12] = label['landmarks']['mouth_right'][0]  # l4_x
            annotation[0, 13] = label['landmarks']['mouth_right'][1]  # l4_y

            if (annotation[0, 4] < 0):
                annotation[0, 14] = -1
                continue
            else:
                annotation[0, 14] = 1

            annotations = annotation
        target = np.array(annotations)

        return target, pose_labels

    def __getitem__(self, index):
        img_path, label_path, mask_path = self.img_list[index].split(',')
        img = Image.open(img_path).convert('RGB')
        img = np.array(img) / 255.

        target, pose_label = self.get_label(label_path, mask_path)
        R = utils.get_R(*pose_label)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        img = img.astype(np.float32)
        target = target.astype(np.float32)
        R = np.expand_dims(R, 0).astype(np.float32)
        if self.train_mode:
            return img, target, R, self.img_list[index]
        else:
            pose_label = pose_label.astype(np.float32)
            return img, target, R, pose_label, self.img_list[index]

    def __len__(self):
        # 15,667
        return self.length


class Pose_300W_LP(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB', train_mode=True):
        self.data_dir = Datasets['Pose_300W_LP']['data_dir']
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(Datasets['Pose_300W_LP']['filename_list'][0])

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(
            self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        img = np.array(img) / 255.

        mat_path = os.path.join(
            self.data_dir, self.y_train[index] + self.annot_ext)

        mat = sio.loadmat(mat_path)

        # Crop the face loosely
        # pt2d = utils.get_pt2d_from_mat(mat_path)
        pt2d = mat['pt2d']
        # x_min = min(pt2d[0, :]) - 1
        # y_min = min(pt2d[1, :]) - 1
        # x_max = max(pt2d[0, :]) + 1
        # y_max = max(pt2d[1, :]) + 1


        # # k = 0.2 to 0.40
        # k = np.random.random_sample() * 0.2 + 0.2
        # x_min -= 0.6 * k * abs(x_max - x_min)
        # y_min -= 2 * k * abs(y_max - y_min)
        # x_max += 0.6 * k * abs(x_max - x_min)
        # y_max += 0.6 * k * abs(y_max - y_min)
        # img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        # pose = utils.get_ypr_from_mat(mat_path)
        pose = mat['Pose_Para'][0][:3]
        # And convert to degrees.
        pitch = pose[0]  # * 180 / np.pi
        yaw = pose[1]  # * 180 / np.pi
        roll = pose[2]  # * 180 / np.pi

        pose_label = np.array([yaw, pitch, roll])
        # pose_label = pose_label.reshape(1, -1)

        data_dir = self.y_train[index].split('/')[1].split('_')[0]
        mat_name = self.y_train[index].split('/')[-1] + '_pts.mat'
        landmark_path = os.path.join(self.data_dir, 'landmarks', data_dir, mat_name)
        ldmks = sio.loadmat(landmark_path)

        pt2d = ldmks['pts_2d']
        # pt2d = pt2d.T

        height, width, _ = img.shape
        if 'Flip' in self.y_train[index]:
            pt2d[:, 0] = width - pt2d[:, 0]

        x_min = min(min(pt2d[:, 0]) - 1, width)
        y_min = min(min(pt2d[:, 1]) - 1, height)
        x_max = max(max(pt2d[:, 0]) + 1, 0)
        y_max = max(max(pt2d[:, 1]) + 1, 0)

        right_eye = (pt2d[42] + pt2d[45]) / 2
        left_eye = (pt2d[36] + pt2d[39]) / 2
        nose = pt2d[30]
        mouth_right, mouth_left = pt2d[54], pt2d[48]

        target = np.array([x_min, y_min, x_max, y_max,
                           right_eye[0], right_eye[1], left_eye[0], left_eye[1], nose[0], nose[1],
                           mouth_right[0], mouth_right[1], mouth_left[0], mouth_left[1], 1], dtype=np.float32)
        target = target.reshape(1, -1)

        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(np.array(img))
        # ax[0].add_patch(patches.Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min),
        #                                   linewidth=2, edgecolor='blue', fill=False))
        # for coord in pt2d:
        #     ax[0].add_patch(patches.Circle(coord, radius=2, color='red'))
        # for coord in [right_eye, left_eye, nose, mouth_right, mouth_left]:
        #     ax[0].add_patch(patches.Circle(coord, radius=2, color='yellow'))
        # ax[1].imshow(utils.plot_pose_cube(np.array(img), yaw* 180 / np.pi, pitch* 180 / np.pi, roll* 180 / np.pi,
        #                                   (x_min + x_max) / 2, (y_min + y_max) / 2,
        #                                   size=(x_max - x_min)))
        # ax[0].set_title(self.y_train[index])
        # plt.show()


        # Gray images

        # # Flip?
        # rnd = np.random.random_sample()
        # if rnd < 0.5:
        #     yaw = -yaw
        #     roll = -roll
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #
        # # Blur?
        # rnd = np.random.random_sample()
        # if rnd < 0.05:
        #     img = img.filter(ImageFilter.BLUR)

        # Add gaussian noise to label
        #mu, sigma = 0, 0.01 
        #noise = np.random.normal(mu, sigma, [3,3])
        #print(noise) 

        # Get target tensors
        R = utils.get_R(pitch, yaw, roll)#+ noise
        R = R.reshape(1, 3, 3)
        
        #labels = torch.FloatTensor([temp_l_vec, temp_b_vec, temp_f_vec])

        if self.transform is not None:
            img, target = self.transform(img, target)

        # return img,  torch.FloatTensor(R),[], self.X_train[index]
        if self.train_mode:
            return img, target, R, self.X_train[index] + self.img_ext
        else:
            pose_label = pose_label.astype(np.float32)
            return img, target, R, pose_label, self.X_train[index] + self.img_ext

    def __len__(self):
        # 122,450
        return self.length


class Pose_300W_LP_masked(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB', train_mode=True):
        self.data_dir = Datasets['Pose_300W_LP_masked']['data_dir']
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext
        filename_list = get_list_from_filenames(Datasets['Pose_300W_LP_masked']['filename_list'][0])

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.train_mode = train_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img_path, label_path, landm_path = self.X_train[index].split(',')
        img = Image.open(os.path.join(self.data_dir, img_path))
        img = img.convert(self.image_mode)
        img = np.array(img) / 255.

        mat_path = os.path.join(self.data_dir, label_path)

        mat = sio.loadmat(mat_path)

        # Crop the face loosely
        # pt2d = mat['pt2d']
        # x_min = min(pt2d[0, :]) - 1
        # y_min = min(pt2d[1, :]) - 1
        # x_max = max(pt2d[0, :]) + 1
        # y_max = max(pt2d[1, :]) + 1

        # # k = 0.2 to 0.40
        # k = np.random.random_sample() * 0.2 + 0.2
        # x_min -= 0.6 * k * abs(x_max - x_min)
        # y_min -= 2 * k * abs(y_max - y_min)
        # x_max += 0.6 * k * abs(x_max - x_min)
        # y_max += 0.6 * k * abs(y_max - y_min)
        # img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        # pose = utils.get_ypr_from_mat(mat_path)
        pose = mat['Pose_Para'][0][:3]
        # And convert to degrees.
        pitch = pose[0]  # * 180 / np.pi
        yaw = pose[1]  # * 180 / np.pi
        roll = pose[2]  # * 180 / np.pi

        pose_label = np.array([yaw, pitch, roll])
        # pose_label = pose_label.reshape(1, -1)

        # data_dir = self.y_train[index].split('/')[1].split('_')[0]
        # mat_name = self.y_train[index].split('/')[-1] + '_pts.mat'
        landmark_path = os.path.join(self.data_dir, landm_path)
        ldmks = sio.loadmat(landmark_path)

        pt2d = ldmks['pts_2d']
        # pt2d = pt2d.T

        height, width, _ = img.shape
        if 'Flip' in self.y_train[index]:
            pt2d[:, 0] = width - pt2d[:, 0]

        x_min = min(min(pt2d[:, 0]) - 1, width)
        y_min = min(min(pt2d[:, 1]) - 1, height)
        x_max = max(max(pt2d[:, 0]) + 1, 0)
        y_max = max(max(pt2d[:, 1]) + 1, 0)

        right_eye = (pt2d[42] + pt2d[45]) / 2
        left_eye = (pt2d[36] + pt2d[39]) / 2
        nose = pt2d[30]
        mouth_right, mouth_left = pt2d[54], pt2d[48]

        target = np.array([x_min, y_min, x_max, y_max,
                           right_eye[0], right_eye[1], left_eye[0], left_eye[1], nose[0], nose[1],
                           mouth_right[0], mouth_right[1], mouth_left[0], mouth_left[1], 1], dtype=np.float32)
        target = target.reshape(1, -1)

        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(np.array(img))
        # ax[0].add_patch(patches.Rectangle((x_min, y_min), (x_max - x_min), (y_max - y_min),
        #                                   linewidth=2, edgecolor='blue', fill=False))
        # for coord in pt2d:
        #     ax[0].add_patch(patches.Circle(coord, radius=2, color='red'))
        # for coord in [right_eye, left_eye, nose, mouth_right, mouth_left]:
        #     ax[0].add_patch(patches.Circle(coord, radius=2, color='yellow'))
        # ax[1].imshow(utils.plot_pose_cube(np.array(img), yaw* 180 / np.pi, pitch* 180 / np.pi, roll* 180 / np.pi,
        #                                   (x_min + x_max) / 2, (y_min + y_max) / 2,
        #                                   size=(x_max - x_min)))
        # ax[0].set_title(self.y_train[index])
        # plt.show()

        # Gray images

        # # Flip?
        # rnd = np.random.random_sample()
        # if rnd < 0.5:
        #     yaw = -yaw
        #     roll = -roll
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #
        # # Blur?
        # rnd = np.random.random_sample()
        # if rnd < 0.05:
        #     img = img.filter(ImageFilter.BLUR)

        # Add gaussian noise to label
        # mu, sigma = 0, 0.01
        # noise = np.random.normal(mu, sigma, [3,3])
        # print(noise)

        # Get target tensors
        R = utils.get_R(pitch, yaw, roll)  # + noise
        R = R.reshape(1, 3, 3)

        # labels = torch.FloatTensor([temp_l_vec, temp_b_vec, temp_f_vec])

        if self.transform is not None:
            img, target = self.transform(img, target)

        # return img,  torch.FloatTensor(R),[], self.X_train[index]
        if self.train_mode:
            return img, target, R, self.X_train[index] + self.img_ext
        else:
            pose_label = pose_label.astype(np.float32)
            return img, target, R, pose_label, self.X_train[index] + self.img_ext

    def __len__(self):
        # 122,450
        return self.length

def getDataset(dataset, transformations=None, train_mode = True):
    if dataset == 'Pose_300W_LP':
        pose_dataset = Pose_300W_LP(transformations, train_mode=train_mode)
    elif dataset == 'AFLW2000':
        pose_dataset = AFLW2000('AFLW2000', transformations)
    elif dataset == 'BIWI':
        pose_dataset = BIWI(transformations, train_mode=train_mode)
    elif dataset == 'AFLW':
        pose_dataset = AFLW(transformations)
    elif dataset == 'AFW':
        pose_dataset = AFW(transformations)
    elif dataset == 'BIWI_masked':
        pose_dataset = BIWI_masked(transformations, train_mode=train_mode)
    elif dataset == 'Pose_300W_LP_masked':
        pose_dataset = Pose_300W_LP_masked(transformations, train_mode=train_mode)
    elif dataset == 'AFLW2000_masked':
        pose_dataset = AFLW2000('AFLW2000_masked', transformations)
    else:
        raise NameError('Error: not a valid dataset name')

    return pose_dataset

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    poses = []
    for _, sample in enumerate(batch):
        imgs.append(torch.from_numpy(sample[0]).float())
        targets.append(torch.from_numpy(sample[1]).float())
        poses.append(torch.from_numpy(sample[2]).float())
        # for _, tup in enumerate(sample):
        #     if torch.is_tensor(tup):
        #         imgs.append(tup)
        #     elif isinstance(tup, type(np.empty(0))):
        #         annos = torch.from_numpy(tup).float()
        #         targets.append(annos)

    return (torch.stack(imgs, 0), torch.stack(targets, 0), torch.stack(poses, 0))
