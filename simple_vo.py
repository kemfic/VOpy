import cv2
import numpy as np
from frame import Frame

def annotate_frames(a, b):
  """
  a is current frame
  b is prev frame
  """

  out = np.copy(a.img)
  old_coords = b.coords

  [cv2.line(out, tuple(np.int0(a.coords[i_a])), tuple(np.int0(b.coords[i_b])), (255, 0, 255), 1) for i_a, i_b in a.des_idxs]

  [cv2.circle(out, tuple(np.int0(a.coords[i_a])), 2,(0,255,0)) for i_a, i_b in a.des_idxs]
  return out

if __name__ == "__main__":
  cap = cv2.VideoCapture('vid/06.mp4')
  ret, frame = cap.read()
  prevFrame = Frame(frame)
  while cap.isOpened():
    ret, frame = cap.read()
    
    # Visual Odom Stuff Here
    curFrame = Frame(frame)
    curFrame.match_frames(prevFrame)

    
    
    
    cv2.imshow("frame", annotate_frames(curFrame, prevFrame))
    prevFrame = curFrame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      print("exiting...")
      break

  cap.release()
  cv2.destroyAllWindows()
