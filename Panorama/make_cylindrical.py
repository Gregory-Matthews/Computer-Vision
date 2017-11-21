import sys

import numpy as np
from scipy.misc import *

from p2_skeleton import to_cylindrical

if __name__ == "__main__":
    filenames = [l.strip().split()[0] for l in open(sys.argv[1]).readlines()]
    camera_params = np.loadtxt(sys.argv[2])

    for fn in filenames:
        ext = "." + fn.split(".")[-1]
        cyl_fn = fn.replace(ext, "_cylindrical" + ext)

        print("Processing '%s' -> '%s' ..." % (fn, cyl_fn))
        img = imread(fn).astype(np.float32) / 255.
        cyl = to_cylindrical(img, camera_params)
        imsave(cyl_fn, cyl)
