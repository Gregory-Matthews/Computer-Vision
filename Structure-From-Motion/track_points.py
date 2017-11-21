import sys

import cv2
import numpy as np

try:
    LINE_AA = cv2.LINE_AA
except:
    LINE_AA = cv2.CV_AA

def point_picker_callback(event, x, y, ignored, data):
    image = data[0]
    point = data[1]
    window_name = data[2]

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        point[:] = (x, y)

def pick_point(image, prev_pts=None):
    data = [image, -np.ones((2,), np.float32), "Click a point."]
    if prev_pts is None:
        prev_pts = []

    cv2.namedWindow(data[-1])
    cv2.setMouseCallback(data[-1], point_picker_callback, data)

    for i, pt in enumerate(prev_pts):
        cv2.circle(image, tuple(pt), 3, (0,0,255), -1)
        cv2.putText(image, str(i+1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, LINE_AA)

    cv2.imshow(data[-1], data[0])

    while data[1][0] == -1:
        key = cv2.waitKey(33)

    cv2.destroyAllWindows()

    return data[1]

if __name__ == "__main__":
    npts = int(sys.argv[1])
    filenames = sys.argv[2:]

    imgs = [cv2.imread(fn) for fn in filenames]
    imgs = [cv2.resize(img, (1024*img.shape[1] // max(img.shape), 1024*img.shape[0] // max(img.shape))) for img in imgs]

    points = [[] for _ in range(len(imgs))]
    for _ in range(npts):
        for pts, img in zip(points, imgs):
            pt = pick_point(img, pts)
            pts.append(pt)

    points = np.array(points, dtype=np.int)

    out = open("tracked_points.txt", "w")
    for i, fn in enumerate(filenames):
        pts = points[i].ravel()
        pts_str = " ".join(map(str, pts))
        out.write(fn + " " + pts_str + "\n")

        img_corrs = imgs[i].copy("C")
        for j, pt in enumerate(points[i]):
            ipt = (int(pt[0]), int(pt[1]))
            cv2.circle(img_corrs, ipt, 5, (0,0,255), -1)
            cv2.putText(img_corrs, str(j+1), ipt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, LINE_AA)
        ext = filenames[i].split(".")[-1]
        cv2.imwrite(filenames[i].replace("."+ext, "_correspondences.png"), img_corrs)
    out.close()
