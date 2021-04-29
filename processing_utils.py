import torch
from scipy.sparse import csr_matrix
import networkx as nx
import stellargraph as sg
from torch_geometric.data import Data
from torch_geometric.utils.undirected import to_undirected
from collections import OrderedDict
import numpy as np
import time
import pandas as pd
from torch_geometric.data import Data
import cv2

from graph_utils import GraphItem, GraphItemLogs, ConnectedComponentCV2
from image_utils import remove_isolated, is_on
import matplotlib.pyplot as plt
import imageio
from skimage.morphology import skeletonize 






class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        new_tensor = tensor.detach().clone()
        for t, m, s in zip(new_tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return new_tensor





class ToGraphTransform(object):
    
    def __init__(self, skeleton_kernel_element=None,
                         opening_kernel_element = None,
                         dilate_skel_edges_kernel = None,
                        SQUARE_IMAGE_SIZE = 512,
                        ):
        
        #assert SQUARE_IMAGE_SIZE == 512 or SQUARE_IMAGE_SIZE == 256, "Image size must be either 256x256 or 512x512"
        if skeleton_kernel_element is None:
            self.skeleton_kernel_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        else:
            self.skeleton_kernel_element = skeleton_kernel_element
            
            
        if opening_kernel_element is None:
            if SQUARE_IMAGE_SIZE == 256:
              self.opening_kernel_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                
            else:
              self.opening_kernel_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        else:
            self.opening_kernel_element = opening_kernel_element
            
            
        if dilate_skel_edges_kernel is None:
            if SQUARE_IMAGE_SIZE == 256:
                self.dilate_skel_edges_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            else:
              self.dilate_skel_edges_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
              
        else:
            self.dilate_skel_edges_kernel = dilate_skel_edges_kernel
            
        
    def graph_transform(self, image, log_procedure=False):
        """
        image -- numpy array grayscale binarized image 512x512 or 256x256

        Adaptation with few modifications to "Automatic extraction of graph-like structures
        from binary images" Marcin Iwanowski.
        This algorithm extracts features that allow the construction of a graph data structure,
        from  images that resemble graph like structures. By employing image morphological operators 
        such as dilation, erosion and combinations of the two, nodes, joints and edges are extracted from 
        the image.
        Nodes and Edges are extracted such that a small overlap region exists between the two.
        Such overlapping regions are named joints and allow for the referencing and creation of the relationships
        between nodes and edges. This makes it possible to generate in the final steps an adjancency matrix.
        Nodes, Edges and Joints objects are separated in their own image matrix in order to facilitate the next steps.  
        In this implemented solution, in order to build the required abstraction of the image structure that
        allows to create the adjacency matrix, CONTIGUOUS areas of pixels that represent a unique object are 
        GROUPED together by means of a connected components labeling algorithm.
        This is done for all three types of objects.
        Having obtained a labeling for all nodes, edges and joints, they can be referenced in the final 
        step of the algorithm, where such identification is employed to generate the adjacency matrix
        """
        logger = OrderedDict()
        logger['0:op_init'] = image 

        start_processing = time.time()
        # identify node like objects in original by means of the morphological
        # operation of opening
        element = self.opening_kernel_element #cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        logger['1:op_opening'] = opening 
      
        # represent nodes on their own feature matrix
        nodes = np.zeros(image.shape, np.uint8)
        nodes[opening == 255] = 255
       
        #cv2_imshow(nodes)
        #cv2plot(nodes)
        #cv2plot(image)
        # obtain skeleton of original image in order to obtain a minimal representation
        # of the connections between nodes
        skel = (skeletonize(image//255)*255).astype(np.uint8)#self.get_skeleton_image(image)
        logger['2:op_skeletonize'] = skel 
        # remove isolated white pixels (isolation is checked by controlling the 8 neighborhood around a white pixel )
        cleaned_skel = remove_isolated(skel)
        logger['3:op_skeletonized_cleanup'] = cleaned_skel 

        # pixels related to nodes are now turned off and only features related to edges remain
        # in the resulting image
        edge_skeleton = np.zeros(image.shape,np.uint8)
        edge_skeleton[ (cleaned_skel == 255) & (nodes==0) ] = 255
        logger['4:op_extract_edge_skeleton'] = edge_skeleton 
        #cv2plot(edge_skeleton)

        # In order to obtain a more robust representation of edge features 
        # a dilation is applied to the edge_skeleton feature matrix.
        # This way, areas representing edges are slightly enlarged in order to 
        # obtain the overlapping needed for the extraction of the Joint objects
        dil_ker = self.dilate_skel_edges_kernel# cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        dil_edges = cv2.dilate(edge_skeleton, dil_ker, iterations = 1)
        logger['5:op_dilate_edge_skeleton'] = dil_edges
        #cv2plot(dil_edges)

        # Joint objects can now be extracted
        joints = np.zeros(image.shape, np.uint8)
        # intersection of dilated edges and nodes position is now not null,
        #i can extract the "joints" between nodes and edges
        # IT IS IMPORTANT TO NOTE THAT DUE TO THIS CONSTRUCTION, FOR EACH PIXEL THAT CORRESPONDS TO A JOINT, THERE IS ALSO AN UNDERLYING
        # NODE AND EDGE, WHICH WILL BE LABELED AS A CONNECTED COMPONENT.
        # CV2 connected component labeling function provides matrices which associate locations in the image with the label of the connected component.
        # This allows, for the reason stated in uppercase above, to FIND THE LABEL OF A NODE AND THAT OF THE EDGE ATTACHED TO IT.
        joints[ (dil_edges == 255) & (nodes == 255)] = 255
        logger['6.1:op_extracted_joints'] = joints
        # following, there are useful visualizations and data structure that can be plotted in order
        # to observe the disjoint sets of nodes, edges and joint objects
        #cv2plot(joints)
        # edge objects purified from joints or nodes overlapping areas
        edges = np.zeros(image.shape, np.uint8)
        edges[(dil_edges == 255) & (nodes == 0) & (joints== 0)] = 255
        logger['6.2:op_extract_edges_no_overlap'] = edges
        #cv2plot(edges)

        # nodes without the joint overlapping areas
        nodes_no_joints = np.zeros(image.shape, np.uint8)
        nodes_no_joints[(nodes==255) & (joints==0)] = 255
        logger['6.3:op_extract_nodes_no_overlap'] = nodes_no_joints
        #cv2plot(nodes_no_joints)

        # visualization to understand with different colors the relationship between edges, nodes and joints
        #viz_data = edges + (nodes_no_joints*0.2).astype(np.uint8) + (joints*0.7).astype(np.uint8)


        # CONNECTED COMPONENTS EXTRACTION
        # this has to be done on the matrices: nodes, joints and dil_edges as these are built such that
        # every pixel location in the intersection of nodes and edges matrices corresponds a specific joint component  
        # this is useful for optimization purposes in the following MAPPING PHASE
        node_components, node_labels, node_stats, node_centroids = cv2.connectedComponentsWithStats(nodes, connectivity=8)
        joint_components, joint_labels, joint_stats, joint_centroids = cv2.connectedComponentsWithStats(joints, connectivity=8)
        edge_components, edge_labels, edge_stats, edge_centroids = cv2.connectedComponentsWithStats(dil_edges, connectivity=8)
        logger['7.1:op_node_components'] = node_labels
        logger['7.2:op_joint_labels'] = joint_labels
        logger['7.3:op_edge_components'] = edge_labels
          
        end_processing = time.time()
        #print("processing: ", end_processing - start_processing)
        
        start = time.time()
        
        
        h,w = image.shape
        # account for zero-th component which in opencv2 is the background, so for each
        # set of components, only num_labels-1 are actually meaningful
        t1 = time.time()
        # all number of components account for the background components as the 0-th 
        # therefore the actual objects represented by a connected component are num_components - 1
        # furthermore labels are treated starting from 1 and when used for indexing purposes are used as 
        # zero based indices for the arrays, by subtracting 1 unit to the label "value"

        V_L = np.zeros((joint_components-1, 1), np.uint8)
        E_L = np.zeros((joint_components-1, 1), np.uint8)
        Adj = np.zeros((node_components-1, node_components-1), np.uint8)
        t2 = time.time()
        
        
        #print("init time: ",t2-t1)
        # MAPPING PHASE: NODES AND EDGES ARE MAPPED TO THE CORRESPONDING JOINT
        # From the labeled regions generated by means of the connected components algorithm,
        # an inner cycle goes through all pixels and
        # if we find the pixel with label corresponding to the current joint,
        # we TAKE ALSO THE LABEL ASSIGNED TO THE EDGE AND THE NODE THAT IS LOCATED AT THAT PIXEL corresponding to the current joint.
        # By the previously performed construction of edge, node and joint objects,
        # a meaningful association between (joint - edge) and (joint - node) is guaranteed to exist.
        # This is because joints have been built in order to represent regions
        # where nodes and edges do overlap.



        t3 = time.time()
        # fast implementation of the previous mapping phase (from the original 50 seconds to 0.01, both for 512x512 images)
        # query all row,col locations that correspond to an identified joint (THE ZERO-TH COMPONENT IS ALWAYS THE BACKGROUND in opencv2).
        # Thus, this query gives all pixel positions that are not the background component for the node-edge joints 
        # np.where() returns locations as row_pos, col_pos  
        row_loc, col_loc = np.where(joint_labels != 0)
        query = np.where(joint_labels != 0)


        #for i,j in zip(query[0], query[1]):
        #    label = joint_labels[i][j]
        #    V_L[label-1] = node_labels[i][j]
        #    E_L[label-1] = edge_labels[i][j]

        # for each joint, store one of its pixel positions (needed to reference the underlying overlapping edge and node)
        # as stated in the reasons provided above the line of code: 'joints[ (dil_edges == 255) & (nodes == 255)] = 255'
        joint_labels_unique_dict = {joint_labels[i][j] : (i,j) for i,j in zip(query[0], query[1])}
        #edge_labels_to_joints_dict = {}
        
        t4 = time.time()
        
        edge_label_to_joint_label_dict = {}
        
        # also in this case, the number of edge components includes the background component
        # so it has to be removed from the number of connected components in the case of the edges.
        # Since edge labels go from 0 to (edge_components-1), with 0 as the background component for a total of (edge_components) 
        # reducing by 1 the num of components and ZERO INDEXING THE remaining ones (which have all labels in the range [1, edge_components-1] )
        # the edge_idx_to_vertices_idx stores the proper edge label, but zero indexed.
        # A -1 in any location corresponds to a non existing node.
        # This way, we find all edges that have both vertices in the image, by verifying that both vertex indices are NON-NEGATIVE
        # in the EDGE_IDX_TO_VERTICES_IDX matrix (num_edges X 2 matrix, where 2 is to store the two vertex ids). 
        edge_idx_to_vertices_idx = np.zeros((edge_components-1,2), np.int)-1
        
        edges_source = []
        edges_target = []
        for dict_item in joint_labels_unique_dict.items():
            joint_label, (i,j) = dict_item
            V_L[joint_label-1] = node_labels[i][j]
            E_L[joint_label-1] = edge_labels[i][j]


            edge_idx = edge_labels[i][j]-1


            joints_list = edge_label_to_joint_label_dict.get(edge_idx+1, None)
            if joints_list is None:
                edge_label_to_joint_label_dict[edge_idx+1] = [joint_label]
            else:
                edge_label_to_joint_label_dict[edge_idx+1].append(joint_label)

                
            edge_vertex_pair = edge_idx_to_vertices_idx[edge_idx]

            if edge_vertex_pair[0] == -1:
                node_idx = node_labels[i][j]-1
                edge_vertex_pair[0] = node_idx
            elif edge_vertex_pair[1] == -1:
                node_idx = node_labels[i][j]-1
                edge_vertex_pair[1] = node_idx
                v1 = edge_vertex_pair[0]
                Adj[v1][node_idx] = 1
                Adj[node_idx][v1] = 1
                
                edges_source.append(v1)
                edges_target.append(node_idx)
                
        
                
        end = time.time()

        computation_time = end-start_processing
        #print("end mapping: ", end - t4)
        #print("end: ", end-start_processing)
        # allows to retrieve which are the actual edge indices and labels that are used in the final graph
        # some edges are not valid since they are at the borders of the image, thus they are missing the remaining edge
        # to which they should be connected to. These are discarded from the final graph:
        # as an example, of 437 edges only 26 are invalid for this reason in the first image in the RCC dataset 
        existing_edges_idx_to_vert_indices_map = { edge_idx:(verts_idx[0],verts_idx[1]) for edge_idx, verts_idx in  enumerate(edge_idx_to_vertices_idx)\
                                                          if verts_idx[0] >= 0 and verts_idx[1] >= 0}
        
        nx_graph = None #nx.from_numpy_matrix(Adj)
        
        # connected components stats are 
        # 1 leftmost pixel of the bounding box of the area of the CC (with respect to the width of the image)
        # 2 top pixel of the bounding box of the area of the CC (with respect to the height of the image)
        # 3 width of the connected component
        # 4 heigh of the connected component
        # 5 area of the connected component in pixels
        # for the purpose of creating node features, the last three are at the moment the most useful for the analysis 
        node_stats_selected = node_stats[1:,2:]
        graph_cc_mean = np.array([ 12.66008916 , 12.10070891, 110.54037857])
        graph_cc_std = np.array([ 5.04678162 , 4.65336656, 72.05804868])

        node_stats_standardized =  (node_stats_selected - graph_cc_mean)/graph_cc_std
        nodes_cc = ConnectedComponentCV2( node_components-1, node_labels, node_stats_selected, node_centroids[1:,:])
        edges_cc = ConnectedComponentCV2( edge_components-1, edge_labels, edge_stats[1:,2:], edge_centroids[1:,:])
        joints_cc = ConnectedComponentCV2( joint_components-1, joint_labels, joint_stats[1:,2:], joint_centroids[1:,:])
        
        

            
        edges_dataframe = pd.DataFrame({"source":edges_source, 
                                        "target":edges_target})
        stellargraph = sg.StellarGraph(nodes=node_stats_standardized, edges=edges_dataframe)

        edge_index = torch.tensor([edges_source,
                                  edges_target], dtype=torch.long)
        edge_index = to_undirected(edge_index, node_components-1)
        #standardized node features for torch graph data
        x = torch.tensor(node_stats_standardized, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)
        
        if log_procedure:
          logs = GraphItemLogs(op_sequence=logger,computation_time=computation_time)
        else:
          logs = None

        graph_item = GraphItem(image,
                nodes_cc,
                edges_cc,
                joints_cc,
                joint_labels_unique_dict,
                existing_edges_idx_to_vert_indices_map,
                stellargraph,
                data,
                Adj,
                nx_graph,
                logs)
        
    
        return graph_item

    def get_skeleton_image(self, img):
        size = np.size(img)
        skel = np.zeros(img.shape, np.uint8)

        # Get a Cross Shaped Kernel
        element = self.skeleton_kernel_element

        # Repeat steps 2-4
        while True:
            #Step 2: Open the image
            open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            #Step 3: Substract open from the original image
            temp = cv2.subtract(img, open)
            #Step 4: Erode the original image and refine the skeleton
            eroded = cv2.erode(img, element)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()
            # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
            if cv2.countNonZero(img)==0:
                break
        return skel
    @staticmethod 
    def plot(graph_item, comp_type='node', highlight_grayscale_color=255):

        node_labels = graph_item.nodes_cc.labels
        edge_labels = graph_item.edges_cc.labels
        joint_labels = graph_item.joints_cc.labels
        
        edge_idxto_verts = graph_item.edge_idxto_verts()
        
        highlight_image = np.zeros(image.shape)
        
        # highlight all nodes
        if comp_type == 'node':
            highlight_image[ node_labels > 0 ] = highlight_grayscale_color

        # else highlight all edges
        elif comp_type == 'edge':
            for edge_idx, verts in edge_idxto_verts.items():
                highlight_image[ edge_labels == edge_idx+1 ] = highlight_grayscale_color
        
        # else highlight all joints
        elif comp_type == 'joint':
            for edge_idx, verts in edge_idxto_verts.items():
                highlight_image[ edge_labels == edge_idx+1 ] = highlight_grayscale_color
            highlight_image[(highlight_image == highlight_grayscale_color) & (joint_labels > 0) ] = highlight_grayscale_color
        return highlight_image
    @staticmethod
    def plot_all(graph_item, highlight_grayscale_color=255):
        node_labels = graph_item.nodes_cc.labels
        edge_labels = graph_item.edges_cc.labels
        joint_labels = graph_item.joints_cc.labels

        highlight_image = np.zeros(graph_item.image.shape)
        highlight_image[  (node_labels > 0) & (joint_labels == 0)  ] = highlight_grayscale_color//4
        highlight_image[  (edge_labels > 0) & (joint_labels == 0)  ] = highlight_grayscale_color//2
        highlight_image[   (joint_labels > 0)  ] = highlight_grayscale_color
        
        return highlight_image
    @staticmethod
    def highlight_component(graph_item,
                            component_idx, 
                            comp_type='node', 
                            highlight_image=None, 
                            highlight_grayscale_color=255):
        
      
        if highlight_image is None:
            highlight_image = np.zeros(graph_item.image.shape)
        if comp_type == 'node':
            node_labels = graph_item.nodes_cc.labels
            highlight_image[ node_labels == component_idx+1] = highlight_grayscale_color
        if comp_type == 'node_edges':
            node_labels = graph_item.nodes_cc.labels
            edge_labels = graph_item.edges_cc.labels
        
            node_edges_labels = edge_labels[(node_labels == component_idx+1) & (edge_labels > 0)]
            build_query = None
            for q in set(node_edges_labels):
                if build_query is None:
                    build_query = (edge_labels == q)
                else:
                    build_query |= (edge_labels == q)
            highlight_image[build_query] = highlight_grayscale_color
            highlight_image[node_labels == component_idx+1] = highlight_grayscale_color//2
          
        if comp_type == 'node_joints':
            node_labels = graph_item.nodes_cc.labels
            edge_labels = graph_item.edges_cc.labels
            joint_labels = graph_item.joints_cc.labels  

            node_joints_labels = joint_labels[(node_labels == component_idx+1) & (joint_labels > 0)]
            build_query = None
            for q in set(node_joints_labels):
                if build_query is None:
                    build_query = (joint_labels == q)
                else:
                    build_query |= (joint_labels == q)
            highlight_image[build_query] = highlight_grayscale_color
            highlight_image[(node_labels == component_idx+1) & (highlight_image == 0) ] = highlight_grayscale_color//2
        if comp_type == 'edge':
            # edge between v1 and v2
            v1, v2 = component_idx

            node_labels = graph_item.nodes_cc.labels
            edge_labels = graph_item.edges_cc.labels
            joint_labels = graph_item.joints_cc.labels

            # obtain v1 location and that of all its edges
            v1_highlight = highlight_component(graph_item,
                            v1, 
                            comp_type='node_edges', 
                            highlight_image=None, 
                            highlight_grayscale_color=255)
            # obtain label of edges of v1 that also happen to intersect with the node location of v2
            # it is only one since this is not a multigraph
            common_edge_label = edge_labels[  (v1_highlight >0) & (node_labels==v2+1) ]
        
            build_query = None
            for q in set(common_edge_label):
                if build_query is None:
                    build_query = (edge_labels == q)
                else:
                    build_query |= (edge_labels == q)
            if build_query is None:
              #no match found
              return None
           
            highlight_image[build_query] = highlight_grayscale_color
            highlight_image[node_labels == v1+1] = highlight_grayscale_color//2
            highlight_image[node_labels == v2+1] = highlight_grayscale_color//4
        return highlight_image

#graph_item = gt.graph_transform( (seg_gt*255.).astype(np.uint8), log_procedure=True)

    @staticmethod
    def plot_graph_data_structure(graph_item,figsize=(10,10)):
        import networkx as nx
        G = nx.Graph(graph_item.adjacencyMatrix)
        plt.figure(figsize=figsize)
        nx.draw(G)
        plt.axis('equal')
    @staticmethod
    def plot_graph_creation_phases(graph_item, duration=1.0, gif_filepath="./", gif_filename="opseq.gif"):
        import imageio
        op_sequence = graph_item.graph_creation_logs.get_opseq()
        if op_sequence is not None:
            kargs = { 'duration': duration}
            git_path = os.path.join(gif_filepath,gif_filename)
            imageio.mimsave(git_path, graph_item.graph_creation_logs.get_opseq().values()[:-3],'GIF',**kargs)

