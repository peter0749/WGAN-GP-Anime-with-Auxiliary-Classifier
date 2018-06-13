import sys
import os
import tqdm
import glob
import face_recognition
from skimage.io import imsave

if not os.path.exists(sys.argv[2]):
    os.makedirs(sys.argv[2])

l = glob.glob(sys.argv[1] + "/*.jpg")
for img_name in tqdm.tqdm(l, total=len(l)):
    image = face_recognition.load_image_file(img_name)
    try:
        miny, maxx, maxy, minx = face_recognition.face_locations(image)[0]
        imsave(os.path.join(sys.argv[2], os.path.split(img_name)[-1]), image[miny:maxy, minx:maxx])
    except:
        continue
