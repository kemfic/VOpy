import cv2
import numpy as np
from frame import Frame
from viewer import Viewer3D
from utils import getError
class SimpleVO(object):
  def __init__(self, img, K=None):

    self.poses = []
    self.gt = []
    self.errors = []
    self.gt.append(np.eye(4))
    self.focal = 7.070912
    if K is not None:
      self.K = np.array(K)
    else:
      self.K = np.array([
        [self.focal, 0, img.shape[1]//2],
        [0, self.focal, img.shape[0]//2],
        [0, 0, 1]])

    self.prevFrame = Frame(img)
    self.curFrame = Frame(img)
    self.poses.append(self.curFrame.Rt)

  def update(self, img, gt):
    self.curFrame = Frame(img)
    self.curFrame.match_frames(self.prevFrame)
    self.curFrame.get_essential_matrix(self.prevFrame)

    self.gt.append(gt)

    #TODO: set scale to 1.0 if there is no gt
    self.scale = np.sqrt(np.sum((self.gt[-1]-self.gt[-2])**2) )    
    #self.scale = 1.0
    
    self.curFrame.get_Rt(self.prevFrame, self.scale)
    self.poses.append(self.curFrame.Rt)

    error_r, error_t = getError(vo.poses[-1],vo.poses[-2],vo.gt[-1], vo.gt[-2])

    self.errors.append((error_r, error_t))

  def annotate_frames(self):
    """
    a is current frame
    b is prev frame
    """
    a = self.curFrame
    b = self.prevFrame
    out = np.copy(a.img)
    old_coords = b.coords
    
    [cv2.line(out, tuple(np.int0(a.coords[i_a])), tuple(np.int0(b.coords[i_b])), (255, 0, 255), 2) for i_a, i_b in a.des_idxs]

    [cv2.circle(out, tuple(np.int0(a.coords[i_a])), 4,(0,255,0), 2) for i_a, i_b in a.des_idxs]
    return out

if __name__ == "__main__":
  cap = cv2.VideoCapture('vid/06.mp4')
  #cap = cv2.VideoCapture('/home/kemfic/projects/ficicislam/dataset/vids/15.mp4')

  ret, frame = cap.read()
  vo = SimpleVO(frame)#, np.eye(4))
  viewer = Viewer3D()
  
  txt = np.loadtxt("vid/06.txt")
  gt_prev = np.eye(4)
  error = []
  while cap.isOpened():
    ret, frame = cap.read()
    #cv2.imshow("frame", vo.annotate_frames())
    
    framenum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    gt = np.eye(4)
    gt[:3, :] = txt[framenum].reshape(3,4)
    gt_tform = gt * np.linalg.inv(gt_prev)


    gt_prev = gt
    vo.update(frame, gt)
    if framenum > 2:
      viewer.update(vo)
      p_tform = vo.poses[-1] * np.linalg.inv(vo.poses[-2])
      error.append((p_tform * np.linalg.inv(gt_tform))[:3, -1])
      #error.append(abs(np.linalg.norm((vo.poses[-1][:3, -1] - vo.poses[-2][:3,-1]) - (gt_prev[:3,-1] - gt[:3,-1]))))
      #print(np.mean(error), error[-1])

    vo.prevFrame = vo.curFrame
    if cv2.waitKey(1) & 0xFF == ord('q'):
      print("exiting...")
      break

  cap.release()
  cv2.destroyAllWindows()
