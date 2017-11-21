import os
import sys

import numpy as np
from scipy.misc import *

from p2_skeleton import mosaic

if __name__ == "__main__":
    filenames, xinit, yinit = zip(*[l.strip().split() for l in open(sys.argv[1]).readlines()])

    cyl_img_fns = []
    for fn in filenames:
        ext = "." + fn.split(".")[-1]
        cyl_img_fns.append(fn.replace(ext, "_cylindrical" + ext))
        if not os.path.exists(cyl_img_fns[-1]):
            print("Could not find cylindrical image '%s' for input image '%s'. Did you run make_cylindrical.py yet?" % (cyl_img_fns[-1], fn))
            exit(1)

    xinit = np.array([float(x) for x in xinit])[:, np.newaxis]
    yinit = np.array([float(y) for y in yinit])[:, np.newaxis]
    disps = np.hstack([xinit, yinit])

    images = [imread(fn)[:,:,:3].astype(np.float32)/255. for fn in cyl_img_fns]

    panorama = mosaic(images, disps)
    imsave("panorama.png", panorama)
