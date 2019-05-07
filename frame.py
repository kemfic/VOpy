import cv2
import numpy as np
from params import param
from utils import *

class Frame(object):
  def __init__(self, img):
    self.img = img
    self.coords = getCorners(img)
    self.coords, self.des = get_features_orb(self.img, self.coords)

  def match_frames(self, prev):
    MIN_DISPLACE = 10
    kp1 = self.coords
    kp2 = prev.coords
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(self.des, prev.des, k=2)

    # Lowe's Ratio Test
    good = []
    des_idxs = []
    for m, n in matches:
      if m.distance < 0.75*n.distance and MIN_DISPLACE < np.linalg.norm(np.subtract(kp2[m.trainIdx], kp1[m.queryIdx])) < 300: #m.distance < 32
        good.append(m.trainIdx)
        des_idxs.append((m.queryIdx, m.trainIdx))
    print(len(des_idxs))
    self.des_idxs = np.array(des_idxs)
    self.kp1 = kp1
