import os
import numpy as np
from skimage import io
from face_detection import RetinaFace
# from retinaface import RetinaFace
import json
from tqdm import tqdm

img_dir = './datasets/BIWI/faces_0'
save_dir = './datasets/BIWI/BIWI_det_annotations'

# os.chdir(img_dir)
img_list = []
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file[-4:] == '.png':
            img_path = os.path.join(root.split('/')[-1], file)

            img_list.append(img_path)


detector = RetinaFace(gpu_id=0)

for img_path in tqdm(img_list, total=len(img_list)):

    img = io.imread(os.path.join(img_dir, img_path))
    det = detector(img)

    resp = {}
    for i, (box, landm, score) in enumerate(det):
        if score > 0.95:
            resp.update({"face_" + str(i + 1): {
                "score": float(score),
                "facial_area": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "landmarks": {
                    "left_eye": [float(landm[0][0]), float(landm[0][1])],
                    "right_eye": [float(landm[1][0]), float(landm[1][1])],
                    "nose": [float(landm[2][0]), float(landm[2][1])],
                    "mouth_left": [float(landm[3][0]), float(landm[3][1])],
                    "mouth_right": [float(landm[4][0]), float(landm[4][1])]
                }
            }})


    # resp = RetinaFace.detect_faces(os.path.join(img_dir, img_path))

    # for faces_n in resp.keys():
    #     resp[faces_n]['score'] = float(resp[faces_n]['score'])
    #     resp[faces_n]['facial_area'] = [int(x) for x in resp[faces_n]['facial_area']]
    #     for landm_key in resp[faces_n]['landmarks'].keys():
    #         resp[faces_n]['landmarks'][landm_key] = [float(x) for x in resp[faces_n]['landmarks'][landm_key]]


    R_path = os.path.join(img_dir, img_path.replace('rgb.png', 'pose.txt'))
    R = np.loadtxt(R_path, dtype=np.float32)
    R = R[:3, :].T

    yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
    pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
    roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
    pose = [float(yaw), float(pitch), float(roll)]

    resp.update({"pose": pose})

    json_path = os.path.join(save_dir, img_path[:-3] + 'json')

    with open(json_path, 'w') as json_file:
        json.dump(resp, json_file, indent=4)

print('Done')