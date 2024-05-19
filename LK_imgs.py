import numpy as np
import cv2
import matplotlib.pyplot as plt

#   Read Input
img1 = cv2.imread("./img/cat/cat1.jpg")
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("./img/cat/cat2.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# find corners in img1
p1 = cv2.goodFeaturesToTrack(img1_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(img1)


# calculate optical flow
p2, st, err = cv2.calcOpticalFlowPyrLK(img1_gray, img2_gray, p1, None, **lk_params)

# Select good points
if p2 is not None:
    good_new = p2[st==1]
    good_old = p1[st==1]

# draw the tracks
for i, (new, old) in enumerate(zip(good_new, good_old)):
    a, b = new.ravel()
    c, d = old.ravel()
    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
    img2 = cv2.circle(img2, (int(a), int(b)), 5, color[i].tolist(), -1)
img = cv2.add(img2, mask)


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imsave()
plt.show()
