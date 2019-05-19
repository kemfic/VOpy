import lib.pangolin as pango
import cv2
import numpy as np
import OpenGL.GL as gl
from multiprocessing import Process, Queue
#from pyquaternion import Quaternion
import quaternion

class Viewer3D(object):
  '''
  3d viewer for g2o maps
    - based off ficiciSLAM's viewer
       - github.com/kemfic/ficiciSLAM
  '''
  w_i, h_i = (600, 200)
  def __init__(self):
    self.state = None
    self.state_gt = None
    self.q_poses = Queue()
    self.q_gt = Queue()
    self.q_img = Queue()
    self.q_errors = Queue()
    self.vt = Process(target=self.viewer_thread, args=(self.q_poses,self.q_gt,self.q_img,self.q_errors,))
    self.vt.daemon = True
    self.vt.start()


    self.poses = []
    self.gt = []
    self.poses.append(np.eye(4))
    self.gt.append(np.eye(4))

  def viewer_thread(self, q_poses, q_gt, q_img, q_errors):
    self.viewer_init()

    while not pango.ShouldQuit():#True:
      #print('refresh')
      self.viewer_refresh(q_poses,q_gt, q_img, q_errors)
    
    self.stop()
  def viewer_init(self):
    w, h = (1024,768)
    f = 2000 #420

    pango.CreateWindowAndBind("Visual Odometry Trajectory Viewer", w, h)
    gl.glEnable(gl.GL_DEPTH_TEST) #prevents point overlapping issue, check out fake-stereo's issues for more info

    # Projection and ModelView Matrices
    self.scam = pango.OpenGlRenderState(
        pango.ProjectionMatrix(w, h, f, f, w //2, h//2, 0.1, 100000),
        pango.ModelViewLookAt(0, -50.0, -10.0,
                              0.0, 0.0, 0.0,
                              0.0, -1.0, 0.0))#pango.AxisDirection.AxisY))
    self.handler = pango.Handler3D(self.scam)

    # Interactive View in Window
    self.dcam = pango.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
    self.dcam.SetHandler(self.handler)
    self.dcam.Activate()


    #Image viewport
    
    self.dimg = pango.Display('image')
    self.dimg.SetBounds(0, self.h_i/h, 1-self.w_i/w, 1.0, -w/h)
    self.dimg.SetLock(pango.Lock.LockLeft, pango.Lock.LockTop)

    self.texture = pango.GlTexture(self.w_i, self.h_i, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    self.img = np.ones((self.h_i, self.w_i, 3),'uint8')*255
    
    # Translation error graph
    self.log = pango.DataLog()
    self.labels = ['error_t', 'error_r']#, "error_euclidean"]
    self.log.SetLabels(self.labels)

    self.plotter = pango.Plotter(self.log,0.0, 1500, -1500, 2500,10, 0.5)
    self.plotter.SetBounds(0.0, self.h_i/h, 0.0, 1-self.w_i/w, -w/h)

    self.plotter.Track("$i", "")

    pango.DisplayBase().AddDisplay(self.plotter)
    self.errorlog_r, self.errorlog_t = [], []

  def viewer_refresh(self, q_poses, q_gt, q_img, q_errors):
    while not q_poses.empty():
      self.state = q_poses.get()
    if not q_img.empty():
      self.img = q_img.get()
      self.img = self.img[::-1, :]
      self.img = cv2.resize(self.img, (self.w_i, self.h_i))

    while not q_gt.empty():
      self.state_gt = q_gt.get()

    while not q_errors.empty():
      error_t, error_r = q_errors.get()
      #self.log.Log(errors[0], errors[1], errors[2])

      self.errorlog_t.append(error_t)
      self.errorlog_r.append(error_r)
      #print(np.shape(self.errorlog))
      self.log.Log(np.sum(self.errorlog_t), np.sum(self.errorlog_r))

    # Clear and Activate Screen (we got a real nice shade of gray
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.15, 0.15, 0.15, 0.0)
    #gl.glClearColor(0.0,0.0, 0.0, 0.0)
    self.dcam.Activate(self.scam)

    # Render
    if self.state is not None:
      gl.glLineWidth(1)
      # Render current pose
      if self.state_gt[0].shape[0] >= 1:
        gl.glColor3f(1.0, 1.0, 1.0)
        pango.DrawCameras(self.state_gt)
      # Render previous keyframes
      if self.state[0].shape[0] >= 2:
        gl.glColor3f(1.0, 0.0, 1.0)
        pango.DrawCameras(self.state[:-1])

      # Render current pose
      if self.state[0].shape[0] >= 1:
        gl.glColor3f(0.2, 1.0, 0.2)
        pango.DrawCameras(self.state[-1:])

    #print(self.img.shape)
    #cv2.imshow("test", self.img)
    
    self.texture.Upload(self.img, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

    self.dimg.Activate()
    gl.glColor3f(1.0, 1.0, 1.0)

    self.texture.RenderToViewport()

    pango.FinishFrame()

  def update(self, vo=None, gt = np.eye(4)):
    '''
    Add new data to queue
    '''

    if self.q_img is None or self.q_poses is None:
      return

    error_t = sum(abs((vo.poses[-1][:3, -1] - vo.poses[-2][:3,-1]) - (self.gt[-1][:3,-1] - gt[:3,-1])))
    
    gt_prev_qt = quaternion.from_rotation_matrix(self.gt[-1][:3, :3])
    gt_qt = quaternion.from_rotation_matrix(gt[:3, :3])
    gt_tform = gt_prev_qt * gt_qt.inverse()
    
    cur_qt = quaternion.from_rotation_matrix(vo.poses[-1][:3, :3])
    prev_qt = quaternion.from_rotation_matrix(vo.poses[-2][:3, :3])

    qt_tform = prev_qt * cur_qt.inverse()

    error_r = np.sum(np.rad2deg(quaternion.rotation_intrinsic_distance(gt_tform, qt_tform)))


    self.poses.append(vo.poses[-1])
    self.gt.append(gt)
    self.q_img.put(vo.annotate_frames())
    self.q_gt.put(np.array(self.gt))
    self.q_poses.put(np.array(self.poses))
    self.q_errors.put((error_t, error_r))
  def stop(self):
    self.vt.terminate()
    self.vt.join()
    qtype = type(Queue())
    for x in self.__dict__.values():
      if isinstance(x, qtype):
        while not x.empty():
          _ = x.get()
    print("viewer stopped")
