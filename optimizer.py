import g2o
import numpy as np

def reprojection_error(angles, t, x0, x1):
    # xEx' = 0
    # E = RS = R[t]_x
    # compose Essential Matrix
    angles = params[0]
    t = params[1]
    x0 = params[2]
    x1 = params[3]
    R = euler2rot3d(angles)
    S = np.array([[0., -t[2], t[1]],
                  [t[2], 0., -t[2]],
                  [-t[1], t[0], 0.]])
    E = R @ S
    # cost
    return np.concatenate(((x0@ E.T @ x1), (x1@ np.linalg.inv(E).T @ x0)))

class PoseGraph3D(object):
  nodes = []
  edges = []
  nodes_optimized = []
  edges_optimized = []

  def __init__(self, verbose=False):
    self.solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    self.solver=  g2o.OptimizationAlgorithmLevenberg(self.solver)

    self.optimizer = g2o.SparseOptimizer()
    self.optimizer.set_verbose(verbose)
    self.optimizer.set_algorithm(self.solver)

  def add_vertex(self, id, pose, is_fixed=False):
    # Rt (pose) matrix, absolute
    v = g2o.VertexSE3()
    v.set_id(id)
    v.set_estimate(g2o.Isometry3d(pose))
    v.set_fixed(is_fixed)

    self.optimizer.add_vertex(v)
    self.nodes.append(v)

  def add_edge(self, vertices, measurement=None, information=np.eye(6), robust_kernel=None):
    edge = g2o.EdgeSE3()
    for i, vertex in enumerate(vertices):
    
    # check to see if we're passing in actual vertices or just the vertex ids
    
      if isinstance(vertex, int): 
        vertex = self.optimizer.vertex(vertex)

      edge.set_vertex(i, vertex)
    
    edge.set_measurement(g2o.Isometry3d(measurement)) # relative pose transformation between frames
    
    # information matrix/precision matrix
    # represents uncertainty of measurement error
    # inverse of covariance matrix
    edge.set_information(information) 
    if robust_kernel is not None:
      edge.set_robust_kernel(robust_kernel)
    
    self.optimizer.add_edge(edge)
  
  def optimize(self, max_iter=15):
    self.optimizer.initialize_optimization()
    self.optimizer.optimize(max_iter)

    self.optimizer.save("data/out.g2o")

    self.edges_optimized = []
    if False:
      for edge in self.optimizer.edges():
        self.edges_optimized = [(edge.vertices()[0].estimate().matrix(), edge.vertices()[1].estimate().matrix())for edge in self.optimizer.edges()]
    self.nodes_optimized = np.array([i.estimate().matrix() for i in self.optimizer.vertices().values()])
    self.nodes_optimized = np.array(self.nodes_optimized)
    self.edges_optimized = np.array(self.edges_optimized)
