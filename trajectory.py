import numpy as np

class Trajectory(object):
  def __init__(self):
    self.poses = [np.eye(4),]
    self.tforms = []


  def add(self, Rt=np.eye(4), scale=1.0):
    t = Rt[:,-1]
    
    Rt[:3, 3] = scale*Rt[:3, -1]/Rt[-1, -1]
    self.tforms.append(Rt)
    self.poses.append(self.poses[-1].dot(
