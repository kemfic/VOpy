import cv2
import numpy as np
from frame import Frame
from viewer import Viewer3D
class SimpleVO(object):
  def __init__(self, img, K=None):

    self.poses = []

    self.focal = 718.8560
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

  def update(self, img):
    self.curFrame = Frame(img)
    self.curFrame.match_frames(self.prevFrame)
    self.curFrame.get_essential_matrix(self.prevFrame)
    self.curFrame.get_Rt(self.prevFrame)
    self.poses.append(self.curFrame.Rt)
    #self.prevFrame = self.curFrame

  def annotate_frames(self):
    """
    a is current frame
    b is prev frame
    """
    a = self.curFrame
    b = self.prevFrame
    out = np.copy(a.img)
    old_coords = b.coords
    
    [cv2.line(out, tuple(np.int0(a.coords[i_a])), tuple(np.int0(b.coords[i_b])), (255, 0, 255), 1) for i_a, i_b in a.des_idxs]

    [cv2.circle(out, tuple(np.int0(a.coords[i_a])), 2,(0,255,0)) for i_a, i_b in a.des_idxs]
    return out

if __name__ == "__main__":
  cap = cv2.VideoCapture('vid/06.mp4')
  ret, frame = cap.read()
  vo = SimpleVO(frame)
  viewer = Viewer3D()

  while cap.isOpened():
    ret, frame = cap.read()
    vo.update(frame)
    cv2.imshow("frame", vo.annotate_frames())
    
    if cap.get(cv2.CAP_PROP_POS_FRAMES) > 2:
      viewer.update(vo)
    vo.prevFrame = vo.curFrame
    if cv2.waitKey(1) & 0xFF == ord('q'):
      print("exiting...")
      break

  cap.release()
  cv2.destroyAllWindows()
