import os
import numpy as np

train_list_file = './datasets/BIWI/BIWI_train.txt'
test_list_file = './datasets/BIWI/BIWI_test.txt'

root_dir = './datasets/BIWI_masked/faces_0'
label_dir = './datasets/BIWI/BIWI_det_annotations'
mask_dir = './datasets/BIWI/head_pose_masks'

file_list = []
train_list = np.loadtxt(train_list_file, dtype=str)
for img_path in train_list:
    label_path = os.path.join(label_dir, img_path[:-4] + '.json')
    mask_path = os.path.join(mask_dir, img_path.replace('rgb', 'depth_mask'))
    if os.path.exists(os.path.join(root_dir, img_path)) and os.path.exists(label_path) and os.path.exists(mask_path):
        file_list.append(os.path.join(root_dir, img_path) + ',' + label_path + ',' + mask_path)
np.savetxt('./datasets/BIWI_masked/train.txt', file_list, fmt='%s', delimiter='/n')
print('Train:', str(len(file_list)))

file_list = []
test_list = np.loadtxt(test_list_file, dtype=str)
for img_path in test_list:
    label_path = os.path.join(label_dir, img_path[:-4] + '.json')
    mask_path = os.path.join(mask_dir, img_path.replace('rgb', 'depth_mask'))
    if os.path.exists(os.path.join(root_dir, img_path)) and os.path.exists(label_path) and os.path.exists(mask_path):
        file_list.append(os.path.join(root_dir, img_path) + ',' + label_path + ',' + mask_path)
np.savetxt('./datasets/BIWI_masked/test.txt', file_list, fmt='%s', delimiter='/n')
print('Test:', str(len(file_list)))

