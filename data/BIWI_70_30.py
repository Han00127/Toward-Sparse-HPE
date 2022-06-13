import scipy.io as sio
import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import sys
import cv2
from moviepy.editor import *
import numpy as np
import argparse
# from mtcnn.mtcnn import MTCNN
from retinaface import RetinaFace
import json
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--db", type=str, default='./datasets/BIWI/faces_0',
                        help="path to database")
    parser.add_argument("--output", type=str, default='./datasets/BIWI/BIWI',
                        help="path to output database mat file")
    parser.add_argument("--img_size", type=int, default=256,
                        help="output image size")
    parser.add_argument("--ad", type=float, default=0.4,
                        help="enlarge margin")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    mypath = args.db
    output_path = args.output
    img_size = args.img_size
    ad = args.ad

    isPlot = False
    # detector = MTCNN()
    # detector = Retinaface

    randFlag = np.zeros(24)
    randFlag[0:16] = 1
    randFlag = np.random.permutation(randFlag)

    print(randFlag)
    # output_train_path = output_path + '_train.npz'
    # output_test_path = output_path + '_test.npz'
    output_train_path = output_path + '_train.txt'
    output_test_path = output_path + '_test.txt'
    output_det_path = output_path + '_det_annotations'

    onlyfiles_png = []
    onlyfiles_txt = []
    for num in range(0, 24):
        if num < 9:
            mypath_obj = mypath + '/0' + str(num + 1)
        else:
            mypath_obj = mypath + '/' + str(num + 1)
        print(mypath_obj)
        onlyfiles_txt_temp = [f for f in listdir(mypath_obj) if
                              isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.txt')]
        onlyfiles_png_temp = [f for f in listdir(mypath_obj) if
                              isfile(join(mypath_obj, f)) and join(mypath_obj, f).endswith('.png')]

        onlyfiles_txt_temp.sort()
        onlyfiles_png_temp.sort()

        onlyfiles_txt.append(onlyfiles_txt_temp)
        onlyfiles_png.append(onlyfiles_png_temp)
    print(len(onlyfiles_txt))
    print(len(onlyfiles_png))

    out_imgs_train = []
    out_poses_train = []
    out_names_train = []
    out_detect_train = []

    out_imgs_test = []
    out_poses_test = []
    out_names_test = []
    out_detect_test = []


    for i in range(len(onlyfiles_png)):
        print('object %d' % i)

        mypath_obj = ''
        if i < 9:
            mypath_obj = mypath + '/0' + str(i + 1)
        else:
            mypath_obj = mypath + '/' + str(i + 1)

        for j in tqdm(range(len(onlyfiles_png[i]))):

            img_name = onlyfiles_png[i][j]
            txt_name = onlyfiles_txt[i][j]

            det_anno_path = os.path.join('../datasets/BIWI/BIWI_det_annotations', mypath_obj[-2:], img_name.replace('.png', '.json'))
            if not os.path.exists(det_anno_path):
                continue
            target, pose = get_label(det_anno_path)
            if len(target) == 0:
                continue

            name = '/'.join((mypath_obj + '/' + img_name).split('/')[-2:])
            if randFlag[i] == 1:
                out_names_train.append(name)
            elif randFlag[i] == 0:
                out_names_test.append(name)

    np.savetxt(output_train_path, out_names_train, fmt='%s', delimiter='/n')
    np.savetxt(output_test_path, out_names_test, fmt='%s', delimiter='/n')
    # np.savez(output_train_path, image=np.array(out_imgs_train), pose=np.array(out_poses_train),
    #          detect=np.array(out_detect_train), img_size=img_size, name=np.array(out_names_train))
    # np.savez(output_test_path, image=np.array(out_imgs_test), pose=np.array(out_poses_test),
    #          detect=np.array(out_detect_test), img_size=img_size, name=np.array(out_names_test))

def get_label(det_anno_path):
    json_file = {}
    # print(self.img_list[index])
    with open(os.path.join(det_anno_path), "r") as st_json:
        json_file = json.load(st_json)

    img_name = '/'.join(det_anno_path.split('/')[-2:]).replace('.json', '.png')

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

    mask_path = os.path.join('../datasets/BIWI/head_pose_masks', img_name.replace('rgb', 'depth_mask'))
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

if __name__ == '__main__':
    main()