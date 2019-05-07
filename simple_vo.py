import cv2
import numpy as np
from params import param

if __name__ == "__main__":
  cap = cv2.VideoCapture('vid/06.mp4')
  ret, frame = cap.read()
  while cap.isOpened():
    ret, frame = cap.read()
    corners = getCorners(frame)
    coords, des = get_features_orb(frame, corners)
    
    #print(corners[0])
    
    [cv2.circle(frame, tuple(i.ravel()), 2, (0,255,0)) for i in corners]
    [cv2.circle(frame, tuple(np.int0(i)), 2, (255,255,0)) for i in coords]
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      print("exiting...")
      break

  cap.release()
  cv2.destroyAllWindows()
