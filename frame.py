import cv2
import numpy as np
from params import param
from utils import *

class Frame(object):
  def __init__(self, img):
    self.Rt = np.eye(4)
    self.img = img
    self.coords = getCorners(img)
    self.focal = 718.8560
    self.K = np.array([
        [self.focal, 0, img.shape[1]//2],
        [0, self.focal, img.shape[0]//2],
        [0, 0, 1]])
    self.coords, self.des = get_features_orb(self.img, self.coords)

  def match_frames(self, prev):
    MIN_DISPLACE = 0
    kp1 = self.coords
    kp2 = prev.coords
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(self.des, prev.des, k=2)

    # Lowe's Ratio Test
    good = []
    des_idxs = []
    for m, n in matches:
      if (m.distance < 0.75*n.distance and MIN_DISPLACE < np.linalg.norm(np.subtract(kp2[m.trainIdx], kp1[m.queryIdx])) < 200): #m.distance < 32
        good.append(m.trainIdx)
        des_idxs.append((m.queryIdx, m.trainIdx))
    #print(len(des_idxs))
    self.des_idxs = np.array(des_idxs)
    self.kp1 = kp1

  def get_essential_matrix(self, prev):
    idxs = self.des_idxs
    
    coord1 = np.array(self.coords)
    coord2 = np.array(prev.coords)
    #print(idxs.shape)
    #print(coord1.shape)
    #print(coord2.shape)
  
    pt1 = coord1[idxs[:,0]]
    pt2 = coord2[idxs[:,1]]
  
    #return idxs[inliers], model.params
    E, mask = cv2.findEssentialMat(pt1, pt2, cameraMatrix=self.K, method=cv2.RANSAC, prob=0.9, threshold=1.0)
    #E, mask = cv2.findFundamentalMat(coord1[idxs[:,0]], coord2[idxs[:,1]], method=cv2.FM_RANSAC, ransacReprojThreshold=3.0, confidence=0.99)
  
    mask = mask.flatten()
    idxs = idxs[(mask==1), :]

    self.des_idxs = idxs

    self.E = E
    #print(self.E)

  def get_Rt(self, prev):
    prev_pts = prev.coords[self.des_idxs[:,1]]
    cur_pts = self.coords[self.des_idxs[:,0]]
    ret, R, t, mask, pts = cv2.recoverPose(self.E, prev_pts, cur_pts, cameraMatrix=self.K, distanceThresh=1000)
    t = t/t[-1]
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = np.squeeze(t)
    #Rt = Rt/t[-1]
    self.Rt = prev.Rt.dot(Rt)
    #self.Rt = Rt.dot(prev.Rt)
    #print(self.Rt)
