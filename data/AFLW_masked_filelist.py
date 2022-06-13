import os
import numpy as np

root_dir = './datasets/AFLW2000_masked'
file_path = './datasets/AFLW2000_masked/files.txt'

file_list = os.listdir(root_dir)
file_list = [file[:-4] for file in file_list if '.jpg' in file]

np.savetxt(file_path, file_list, fmt='%s', delimiter='/n')