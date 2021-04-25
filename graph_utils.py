


def estimate_graph_statistics(graph_dataset):
    """
    returns mean and variance of each color channel
    """
    mean_k = 0
    var_k = 0
    K = 0

    M = 0
    N = 0
    # dimensions of the image, along which mean and std are computed for the channels
    dim = (0,1)
    for graph in graph_dataset:
        
        node_features = graph.node_features()
          
        
        N = node_features.shape[0]
        curr_mean = graph.node_features().mean(axis=0)#img.mean(dim)#axis=(0,1))

        old_stat_weight = M/(M+N)
        new_stat_weight = N/(M+N)
        mean_k_new = mean_k*old_stat_weight + curr_mean*new_stat_weight


        curr_var = node_features.var(axis=0)#axis=(0,1))
        var_k = old_stat_weight*var_k +new_stat_weight*curr_var + old_stat_weight*new_stat_weight*(mean_k-curr_mean)**2

        mean_k = mean_k_new
        M += N
    return mean_k, var_k


class ConnectedComponentCV2(object):
    def __init__(self, num_components, labels, stats, centroids):
        self.num_components = num_components
        self.labels = labels 
        self.stats = stats
        self.centroids = centroids
        
        
class GraphItemLogs(object):
  def __init__(self, op_sequence=None,computation_time=None):
    self.op_sequence = op_sequence
    self.computation_time = computation_time

  def get_opseq(self):
    return self.op_sequence  
  def get_time(self):
    return self.computation_time

class GraphItem(object):
    
    def __init__(self, image,
                nodes_cc,
                edges_cc,
                joints_cc,
                joint_label_summarized_position,
                 existing_edges_idx_to_vert_indices_map,
                 stellar_graph,
                 torch_geom_data,
                adjacencyMatrix,
                nx_graph,
                graph_creation_logs=None):
        self.image = image
        
        
        self.nodes_cc = nodes_cc
        self.edges_cc = edges_cc
        self.joints_cc = joints_cc
        
        self.joint_label_summarized_position = joint_label_summarized_position
        
        self.existing_edges_idx_to_vert_indices_map = existing_edges_idx_to_vert_indices_map
        
        
        self.stellar_graph = stellar_graph
        self.torch_geom_data = torch_geom_data
        self.adjacencyMatrix = adjacencyMatrix

        self.nx_graph = nx_graph

        self.graph_creation_logs = graph_creation_logs
    def edge_idxto_verts(self):
        return self.existing_edges_idx_to_vert_indices_map
              