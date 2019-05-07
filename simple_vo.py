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

def getMatches(im1, im2, corner1, corner2):
  return NotImplemented

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
