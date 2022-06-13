## Amir Shahroudy
# https://github.com/shahroudy

import os, sys, argparse
import numpy as np
from utils import utils

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Create filenames list txt file from datasets root dir.'
        ' For head pose analysis.')
    parser.add_argument('--root_dir = ',
        dest='root_dir',
        help='root directory of the datasets files',
        default='./datasets/300W_LP_masked',
        type=str)
    parser.add_argument('--filename',
        dest='filename',
        help='Output filename.',
        default='files.txt',
        type=str)
    parser.add_argument('--label_dir',
                        default='../300W_LP',
                        type=str)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    os.chdir(args.root_dir)

    file_counter = 0
    rej_counter = 0
    outfile = open(args.filename, 'w')

    mask_list = ['empty', 'KN95', 'gas', 'inpaint', 'surgical', 'cloth', 'N95']

    for root, dirs, files in os.walk('.'):
        for f in files:
            mask = 'inpaint'
            for m in mask_list:
                if m in f:
                    mask = m
                    break
            if mask == 'inpaint':
                print(f)
                continue

            img_name = f.split('_' + mask)[0]
            dir_name = root.split('_')[0][2:]

            mat_path = os.path.join(args.label_dir, dir_name, img_name + '.mat')
            # We get the pose in radians
            pose = utils.get_ypr_from_mat(mat_path)
            # And convert to degrees.
            pitch = pose[0] * 180 / np.pi
            yaw = pose[1] * 180 / np.pi
            roll = pose[2] * 180 / np.pi

            landm_path = os.path.join(args.label_dir, 'landmarks', dir_name, img_name + '_pts.mat')

            if abs(pitch) <= 99 and abs(yaw) <= 99 and abs(roll) <= 99:
                if file_counter > 0:
                    outfile.write('\n')
                file_txt = os.path.join(root, f) + ',' + mat_path + ',' + landm_path
                outfile.write(file_txt)
                file_counter += 1
            else:
                rej_counter += 1
    outfile.close()
    print(f'{file_counter} files listed! {rej_counter} files had out-of-range'
        f' values and kept out of the list!')