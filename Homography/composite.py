import sys

import numpy as np
from scipy.misc import *

from p1_skeleton import composite

try:
    from util import pick_corrs
    HAVE_CORR_PICKER = True
except:
    HAVE_CORR_PICKER = False

def print_usage():
    print("""USAGE: python composite.py SOURCE TARGET MASK

(Requires Python OpenCV bindings, lets you pick correspondences with your mouse.)

OR

    python composite.py SOURCE TARGET SOURCE_POINTS TARGET_POINTS MASK

(You provide two text files with points, one per line, in the form
    X1 Y1
    X2 Y2
    X3 Y3
    X4 Y4
)""")

if __name__ == "__main__":
    if len(sys.argv) not in [4,6]:
        print_usage()
        exit(1)

    source = imread(sys.argv[1]).astype(np.float32) / 255.
    target = imread(sys.argv[2]).astype(np.float32) / 255.
    mask = imread(sys.argv[-1]).astype(np.float32) / 255.

    if mask.ndim > 2:
        mask = mask[:,:,0]
    mask = mask[:, :, np.newaxis]
    
    # Use the OpenCV corodinate picker if available, otherwise load from a file.
    if HAVE_CORR_PICKER and len(sys.argv) == 4:
        corrs = pick_corrs([source[:,:,::-1], target[:,:,::-1]])
        if corrs is None:
            print("You must pick all 4 correspondences in each image.")
            exit(1)

        source_pts, target_pts = corrs

        source_pts = np.array(source_pts)
        target_pts = np.array(target_pts)

        print("Source Points:")
        for pt in source_pts:
            print(pt[0], pt[1])
        print()
        print("Target Points:")
        for pt in target_pts:
            print(pt[0], pt[1])
        print()
    elif not HAVE_CORR_PICKER and len(sys.argv) == 4:
        print("Correspondence picker window could not be created. Are python OpenCV bindings installed/available?")
        exit(1)
    else:
        if len(sys.argv) != 6:
            print_usage()
            exit(1)

        source_pts = np.loadtxt(sys.argv[3])
        target_pts = np.loadtxt(sys.argv[4])

    composited = composite(source, target, source_pts, target_pts, mask)
    imsave("composited.png", composited)
