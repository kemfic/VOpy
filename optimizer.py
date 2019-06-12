import g2o
import numpy as np

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
    v = g2o.VertexSE3()
    v.set_id(id)
    v.set_estimate(pose)
    v.set_fixed(is_fixed)

    self.optimizer.add_vertex(v)
    self.nodes.append(v)

  def add_edge(self, nodes, measurement=None, information=np.eye(6), robust_kernel=None):
    edge = g2o.EdgeSE3()
    for i, vertex in enumerate(vertices):
      if isinstance(node_id, int):
        node = self.vertex(node_id)

      edge.set_vertex(i, node)
  def optimize(self, iterations=1):
    self.optimizer.initialize_optimization()
    self.optimizer.optimize(iterations)

    self.optimizer.save("data/out.g2o")
    self.edges_optimized = []
    for edge in self.optimizer.edges():
      self.edges_optimized.append([edge.vertices()[0].estimate().matrix(), edge.vertices()[1].estimate().matrix()])

    self.nodes_optimized = np.array([i.estimate().matrix() for i in self.optimizer.vertices().values()])
    self.nodes_optimized = np.array(self.nodes_optimized)
    self.edges_optimized = np.array(self.edges_optimized)
