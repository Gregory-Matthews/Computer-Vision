import sys
import os

import cv2
import numpy as np

try:
    LINE_AA = cv2.LINE_AA
except:
    LINE_AA = cv2.CV_AA

from p3_skeleton import sfm, get_texture

if __name__ == "__main__":
    desc_file = sys.argv[1]
    quad_file = sys.argv[2]
    desc_base = desc_file.split(os.path.sep)[:-1]

    # Load image filenames and correspondences from the description file.
    cols = list(zip(*[ln.strip().split() for ln in open(desc_file).readlines()]))
    image_fns = [os.path.sep.join(desc_base + [imgfn]) for imgfn in cols[0]]
    images = [cv2.imread(fn) for fn in image_fns]
    images = [cv2.resize(img, (1024*img.shape[1] // max(img.shape), 1024*img.shape[0] // max(img.shape))) for img in images]
    image_points = np.column_stack(cols[1:]).astype(np.float32).reshape(len(image_fns), -1, 2)
    F, N = image_points.shape[:2]

    # Call the supplied SFM implementation.
    Rmats, P = sfm(image_points)

    # Reproject all of the reconstructed 3D points into each image and save the
    # result for debugging.
    if not os.path.exists("reprojections"):
        os.mkdir("reprojections")

    for R, img, fn, ipts in zip(Rmats, images, image_fns, image_points):
        Rpts = np.dot(P, R.T)
        center = ipts.mean(0)
        pproj = Rpts + center

        img_reproj = img.copy()
        for i, pt in enumerate(pproj):
            ipt = (int(pt[0]), int(pt[1]))
            cv2.circle(img_reproj, ipt, 5, (0,0,255), -1)
            cv2.putText(img_reproj, str(i+1), ipt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, LINE_AA)

        img_corrs = img.copy()
        for i, pt in enumerate(ipts):
            ipt = (int(pt[0]), int(pt[1]))
            cv2.circle(img_corrs, ipt, 5, (0,0,255), -1)
            cv2.putText(img_corrs, str(i+1), ipt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, LINE_AA)

        fn = fn.split(os.path.sep)[-1]
        ext = fn.split(".")[-1]
        cv2.imwrite("reprojections" + os.path.sep + fn.replace("."+ext, "_reprojected.png"), img_reproj)
        cv2.imwrite("reprojections" + os.path.sep + fn.replace("."+ext, "_correspondences.png"), img_corrs)

    # Save an OBJ file with the reconstructed 3D points and texture-mapped
    # polygons.
    if not os.path.exists("textures"):
        os.mkdir("textures")

    quad_inds = np.loadtxt(quad_file).astype(np.int)
    if quad_inds.ndim == 1:
        quad_inds = quad_inds[np.newaxis]

    # Make sure the quad normals have positive Z value (facing the camera).
    for q in quad_inds:
        qpts = P[q-1]
        v1 = qpts[0] - qpts[1]
        v2 = qpts[2] - qpts[1]
        n = np.cross(v2, v1)
        if n[2] < 0:
            q[:] = q[::-1]

    objfile = open("mesh.obj", "w")
    mtlfile = open("textures/textures.mtl", "w")
    objfile.write("mtllib textures/textures.mtl\n\n")

    # Write the 3D points, swapping Y/Z axes to match the space of most 3D
    # programs.
    P = P[:, [0,2,1]]
    P[:, 2] *= -1
    objfile.writelines(["v " + " ".join(map(str, v)) + "\n" for v in P])

    # Vertex texture coordinates. OpenGL uses bottom-left as the origin, so the
    # image origin of top-left/0,0 maps to 0,1 etc...
    objfile.write("\n")
    objfile.write("vt 0 1\n")
    objfile.write("vt 1 1\n")
    objfile.write("vt 1 0\n")
    objfile.write("vt 0 0\n")
    


    for i, q in enumerate(quad_inds):
        # Extract a texture for each quad, taking the median across all views.
        tex = get_texture(images, image_points[:, q-1])
        cv2.imwrite("textures/texture%d.png" % i, tex)

        # Write the material information for each quad.
        objfile.write("\n")

        mtlfile.write("newmtl texture%d\n" % i)
        mtlfile.write("Ka 1 1 1\n")
        mtlfile.write("Kd 1 1 1\n")
        mtlfile.write("Ks 0 0 0\n")
        mtlfile.write("Tr 1\n")
        mtlfile.write("illum 1\n")
        mtlfile.write("Ns 0\n")
        mtlfile.write("map_Kd textures/texture%d.png\n" % i)
        mtlfile.write("map_Ka textures/texture%d.png\n" % i)
        mtlfile.write("\n")

        # Write the texture coordinates for each quad vertex.
        objfile.write("usemtl texture%d\n" % i)
        objfile.write("f " + " ".join(["%d/%d" % (r, (j+1)) for j, r in enumerate(q)]) + "\n")
    objfile.close()
