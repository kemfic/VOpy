import numpy as np

K_KITTI = np.array([
        [self.focal, 0, img.shape[1]//2],
        [0, self.focal, img.shape[0]//2],
        [0, 0, 1]])

FOCAL_KITTI = 718.8560

class Params(object):
  def __init__(self, 
              focal=FOCAL_KITTI
              K=K_KITTI, 

              datasetPath = None, 
              gtPath = None, 
              maxCorners = 10000, 
              cornerQuality = 0.01, 
              minCornerDist=10):

  self.K = K
  self.datsetPath = 
    

param = dict(
  gftt = dict(maxCorners = 100000,
              qualityLevel = 0.01,
              minDistance =10)
              
              )
