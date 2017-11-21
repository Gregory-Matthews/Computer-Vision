import sys

import numpy as np
from scipy.misc import *

from p1_skeleton import mosaicing

img1 = imread(sys.argv[1]).astype(np.float32) / 255.
img2 = imread(sys.argv[2]).astype(np.float32) / 255.
img3 = imread(sys.argv[3]).astype(np.float32) / 255.
pts1_3 = np.loadtxt(sys.argv[4])
pts3_1 = np.loadtxt(sys.argv[5])
pts2_3 = np.loadtxt(sys.argv[6])
pts3_2 = np.loadtxt(sys.argv[7])

mosaic = mosaicing(img1, img2, img3, pts1_3, pts3_1, pts2_3, pts3_2)
   
imsave("mosaic.png", mosaic)
