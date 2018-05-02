import cv2
import sys
import os
import numpy as np

def detect_and_crop(filename, cascade_file = "./lbpcascade_animeface.xml", size=96):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (size, size))
    face_crops = []
    for (x, y, w, h) in faces:
        x_c = int(x+0.5*w)
        y_c = int(y+0.5*h)
        scale = min(w,h) # minimum scale
        x0, y0 = max(0,x_c-scale//2), max(0,y_c-scale//2)
        crop = image[y0:y0+scale, x0:x0+scale, :3]
        p0 = max(0,scale-crop.shape[0])
        p1 = max(0,scale-crop.shape[1])
        crop = np.pad(crop, ( (p0//2, p0-p0//2) , (p1//2, p1-p1//2) , (0,0) ), 'constant', constant_values=0)
        if scale != size:
            cv2.resize(crop, (size, size), interpolation = cv2.INTER_AREA)
        face_crops.append(crop)

    return face_crops

import argparse
parser = argparse.ArgumentParser(description='Make dataset')
parser.add_argument('--input_dir', type=str, default='./pixiv_all', required=False,
                    help='')
parser.add_argument('--output_dir', type=str, default='./cropped_faces', required=False,
                    help='')
parser.add_argument('--size', type=int, default=96, required=False,
                    help='')
parser.add_argument('--cascade_file', type=str, default='./lbpcascade_animeface.xml', required=False,
                    help='')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

import glob
img_p_list = glob.glob(args.input_dir+'/*.jpg')
img_p_list.extend(glob.glob(args.input_dir+'/*.png'))
img_p_list.extend(glob.glob(args.input_dir+'/**/*.png'))
img_p_list.extend(glob.glob(args.input_dir+'/**/*.jpg'))

face_cnt = 0
for img_p in img_p_list:
    faces = detect_and_crop(img_p, cascade_file=args.cascade_file, size=args.size)
    if len(faces)==0: 
        continue
    for face in faces:
        cv2.imwrite(os.path.join(args.output_dir, 'face_{:08d}.png'.format(face_cnt)), face)
        face_cnt += 1
