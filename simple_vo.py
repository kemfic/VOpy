import cv2
import numpy as np

if __name__ == "__main__":
  cap = cv2.VideoCapture('vid/06.mp4')
  ret, frame = cap.read()
  while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      print("exiting...")
      break

  cap.release()
  cv2.destroyAllWindows()
