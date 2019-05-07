import cv2
import numpy as np
from frame import Frame

if __name__ == "__main__":
  cap = cv2.VideoCapture('vid/06.mp4')
  ret, frame = cap.read()
  prevFrame = Frame(frame)
  while cap.isOpened():
    ret, frame = cap.read()
    
    # Visual Odom Stuff Here
    curFrame = Frame(frame)
    curFrame.match(prevFrame)

    
    
    
    cv2.imshow("frame", frame)
    prevFrame = curFrame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      print("exiting...")
      break

  cap.release()
  cv2.destroyAllWindows()
