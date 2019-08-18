from PIL import Image
import glob
import os
import cv2
import numpy as np

dirpath = os.getcwd()
dirpath += "/data"
olddir = dirpath + "/old"
newdir = dirpath +'/imgs'
if(os.path.exists(newdir) == False):
    os.mkdir(newdir)

folders= ["bottle", "container", "packaging"]
for folder in folders:
    path = olddir +'/' + folder
    num = 0
    for filename in glob.glob(path+'/*.jpg'):
        num += 1;
        img = np.array(cv2.imread(filename))
        cv2.imwrite(newdir+"/" + folder + "_"+str(num)+'.jpg', img)




