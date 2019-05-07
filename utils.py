# Utility functions
import cv2
import numpy as np
from params import param

def bgr2gray(img):
  return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def getCorners(img, params=param.get("gftt")):
  """
   - Example from: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
 
   - API Ref: https://docs.opencv.org/master/dd/d1a/group__imgproc__feature.html#ga1d6bb77486c8f92d79c8793ad995d541
  """
  gray = bgr2gray(img)
  corners =  cv2.goodFeaturesToTrack(gray, **params)
  corners = np.int0(corners)
  return corners

def get_features_orb(img, corners):
  '''
  Computes ORB descriptors and coords for an image
  '''

  kp = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=15) for f in corners]

  orb = cv2.ORB_create()
  kps, des = orb.compute(img, kp)

  # Convert Keypoint to coord
  coords = np.array([kp.pt for kp in kps])
  
  print(kps[1].pt)
  return coords, des

def match_frames(des1, des2, pt1, pt2):
  '''
  Matches features using K-Nearest Neighbors, and returns the indexes of the matches
  '''

  kp1 = coords_to_kp(pt1)
  kp2 = coords_to_kp(pt2)
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(des1, des2, k=2)

  # Lowe's Ratio Test
  good = []
  des_idxs = []
  for m, n in matches:
    if m.distance < 32 and (m.distance < 0.75*n.distance and 5 < np.linalg.norm(np.subtract(kp2[m.trainIdx].pt, kp1[m.queryIdx].pt)) < 300): #m.distance < 32
      good.append(m.trainIdx)
      des_idxs.append((m.queryIdx, m.trainIdx))

  des_idxs = np.array(des_idxs)

  return des_idxs
