import os
import numpy as np

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])   
def my_compute_box_3d(center, size, heading_angle):
    R = rotz(-1*heading_angle)
    l,w,h = size
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)
def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0
def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:,0:3], box3d)
    return pc[box3d_roi_inds,:], box3d_roi_inds 
# def get_box3d_dim_statistics(folder_path):
#     """ Collect 3D bounding box statistics.
#     Used for computing mean box sizes. """
#     data_idx_list = []
#     for file in os.listdir(folder_path):
#         if file.endswith("_bbox.npy") and not file.endswith("_2d_bbox.npy"):
#             print(file)
#             data_idx_list.append(file)
#     print("data_idx_list", data_idx_list)
#     for data_idx in data_idx_list:
#         #print(os.path.exists(os.path.join(folder_path, '%06d_votes.npz'%(int(data_idx.split('_')[0])))))
#         # if os.path.exists(os.path.join(folder_path, '%06d_votes.npz'%(int(data_idx.split('_')[0])))):
#         #     print("pass",data_idx)
#         #     continue
#         print('------------- ', data_idx)
#         # 构建完整的文件路径
#         data_file_path = os.path.join(folder_path, data_idx)
#         # 使用np.load加载.npy文件
#         objects = np.load(data_file_path)
#         # Skip scenes with 0 object

#         # if  len(objects)==0:
#         #         continue
            
#         pc_path = os.path.join(folder_path, data_idx.replace('_bbox.npy','_pc.npy'))
#         pc = np.load(pc_path)
#         pc_upright_depth_subsampled = random_sampling(pc,50000)
        
#         N = pc_upright_depth_subsampled.shape[0]
#         point_votes = np.zeros((N,10)) # 3 votes and 1 vote mask 
#         point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
#         indices = np.arange(N)
#         for obj in objects:
#             try:
#                 # Find all points in this object's OBB
#                 box3d_pts_3d = my_compute_box_3d(obj[0:3],
#                     np.array([obj[3],obj[4],obj[5]]), obj[6])
#                 pc_in_box3d,inds = extract_pc_in_box3d(\
#                     pc_upright_depth_subsampled, box3d_pts_3d)
#                 # Assign first dimension to indicate it is in an object box
#                 point_votes[inds,0] = 1
#                 # Add the votes (all 0 if the point is not in any object's OBB)
#                 votes = np.expand_dims(obj[0:3],0) - pc_in_box3d[:,0:3]
#                 sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
#                 for i in range(len(sparse_inds)):
#                     j = sparse_inds[i]
#                     point_votes[j, int(point_vote_idx[j]*3+1):int((point_vote_idx[j]+1)*3+1)] = votes[i,:]
#                     # Populate votes with the fisrt vote
#                     if point_vote_idx[j] == 0:
#                         point_votes[j,4:7] = votes[i,:]
#                         point_votes[j,7:10] = votes[i,:]
#                 point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds]+1)
#             except:
#                     print('ERROR ----',  data_idx, obj[7])
#         print("output_folder",os.path.join(folder_path, '%s_votes.npz'%(data_idx.split('_bbox')[0])))
#         # np.savez_compressed(os.path.join(folder_path, '%s_votes.npz'%(data_idx.split('_bbox')[0]))),
#         #     point_votes = point_votes)

def worker(data_idx, folder_path):

    #print(os.path.exists(os.path.join(folder_path, '%06d_votes.npz'%(int(data_idx.split('_')[0])))))
    # if os.path.exists(os.path.join(folder_path, '%06d_votes.npz'%(int(data_idx.split('_')[0])))):
    #     print("pass",data_idx)
    #     continue
    print('------------- ', data_idx)
    # 构建完整的文件路径
    data_file_path = os.path.join(folder_path, data_idx)
    # 使用np.load加载.npy文件
    objects = np.load(data_file_path)
    # Skip scenes with 0 object

    # if  len(objects)==0:
    #         continue
        
    pc_path = os.path.join(folder_path, data_idx.replace('_bbox.npy','_pc.npy'))
    pc = np.load(pc_path)
    pc_upright_depth_subsampled = random_sampling(pc,50000)
    np.savez_compressed(os.path.join(folder_path, '%s_pc.npz'%(data_idx.split('_bbox')[0])),
            pc=pc_upright_depth_subsampled)
    N = pc_upright_depth_subsampled.shape[0]
    point_votes = np.zeros((N,10)) # 3 votes and 1 vote mask 
    point_vote_idx = np.zeros((N)).astype(np.int32) # in the range of [0,2]
    indices = np.arange(N)
    for obj in objects:
        try:
            # Find all points in this object's OBB
            box3d_pts_3d = my_compute_box_3d(obj[0:3],
                np.array([obj[3],obj[4],obj[5]]), obj[6])
            pc_in_box3d,inds = extract_pc_in_box3d(\
                pc_upright_depth_subsampled, box3d_pts_3d)
            # Assign first dimension to indicate it is in an object box
            point_votes[inds,0] = 1
            # Add the votes (all 0 if the point is not in any object's OBB)
            votes = np.expand_dims(obj[0:3],0) - pc_in_box3d[:,0:3]
            sparse_inds = indices[inds] # turn dense True,False inds to sparse number-wise inds
            for i in range(len(sparse_inds)):
                j = sparse_inds[i]
                point_votes[j, int(point_vote_idx[j]*3+1):int((point_vote_idx[j]+1)*3+1)] = votes[i,:]
                # Populate votes with the fisrt vote
                if point_vote_idx[j] == 0:
                    point_votes[j,4:7] = votes[i,:]
                    point_votes[j,7:10] = votes[i,:]
            point_vote_idx[inds] = np.minimum(2, point_vote_idx[inds]+1)
        except:
                print('ERROR ----',  data_idx, obj[7])
    print("output_folder",os.path.join(folder_path, '%s_votes.npz'%(data_idx.split('_bbox')[0])))
    np.savez_compressed(os.path.join(folder_path, '%s_votes.npz'%(data_idx.split('_bbox')[0])),point_votes = point_votes)

import multiprocessing
def main_process(folder_path):
    """ Collect 3D bounding box statistics.
    Used for computing mean box sizes. """
    data_idx_list = [file for file in os.listdir(folder_path) if file.endswith("_bbox.npy") and not file.endswith("_2d_bbox.npy")]
    print("data_idx_list", data_idx_list)
    
    num_processes = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(worker, [(data_idx, folder_path) for data_idx in data_idx_list])


folder_path = "/data2/timingyang/dataset_new/dataset"
main_process(folder_path)