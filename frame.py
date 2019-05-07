import cv2
import numpy as np
from params import param
from utils import *

class Frame():
  def __init__(self, img):
    self.img = img
    self.coords = getCorners(img)
    self.coords, self.des = get_features_orb(self.img, self.coords)

  def match_frames(self, prev):
    kp1 = coords_to_kp(self.coords)
    kp2 = coords_to_kp(prev.coords)
    bf = bf.knnMatch(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's Ratio Test
    good = []
    des_idxs = []
    for m, n in matches:
      if m.distance < 0.75*n.distance and MIN_DISPLACE < np.linalg.norm(np.subtract(kp2[m.trainIdx].pt, kp1[m.queryIdx].pt)) < 300: #m.distance < 32
        good.append(m.trainIdx)
        des_idxs.append((m.queryIdx, m.trainIdx))
