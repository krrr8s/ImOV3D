import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import os
import cv2


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
from multiprocessing import Pool, cpu_count
import multiprocessing
def check_rotation_matrix(R):
    # Step 3: Check matrix integrity and presence of None or NaN values
    has_none = np.any(np.isnan(R))
    has_nan = np.any(np.isnan(R))

    if has_none or has_nan:
        # Return identity matrix if matrix has None or NaN values
        return np.eye(3)

    return R
def get_PCA(pc):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(pc)
    pcd.colors = o3d.utility.Vector3dVector(np.zeros(np.array(pcd.colors).shape))
    downpcd = pcd.voxel_down_sample(voxel_size=5e-2)


    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.geometry.PointCloud.orient_normals_to_align_with_direction(downpcd,
                                                                orientation_reference=np.array([0.0, 0.0, 1.0]))

    # Converts normal vectors to numpy arrays
    normals = np.array(downpcd.normals)
    # Using k-means clustering algorithm
    kmeans = KMeans(n_clusters=1,n_init=10)  # Set the number of clusters to look for to 1, that is, find a unique normal vector
    kmeans.fit(normals)
    # Get the center point of the cluster, that is, the unique normal vector
    unique_normal = kmeans.cluster_centers_[0]
    z = np.array([0, 0, 1])
    # Calculate the rotation matrix
    rotation_matrix = rotation_matrix_from_vectors(unique_normal, z)

    '''
    print(rotation_matrix)
    unique_normal = np.dot(rotation_matrix,unique_normal)
    print(unique_normal)


    # Create LineSet object
    line_set = o3d.geometry.LineSet()

    # Define start and end points for the line segment
    start_point = np.array([0, 0, 0])  # Starting point of the line
    end_point = unique_normal  # Endpoint of the line, using the unique_normal obtained from K-means

    # Set line vertices and colors
    line_set.points = o3d.utility.Vector3dVector(np.vstack((start_point, end_point)))
    line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    line_set.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]]))  # Set color to red
    o3d.io.write_line_set("unique_normal_line.ply", line_set)

    # Visualize the LineSet
    #o3d.visualization.draw_geometries([downpcd, line_set], window_name="Unique Normal", width=800, height=600)
    '''
    return check_rotation_matrix(rotation_matrix)

def worker(folder_list, calib_files,folder_path,calib_folder,calib_save_path):
    for file_name, calib_file in zip(folder_list, calib_files):
      
        if not file_name.lower().endswith(".ply"):
            continue


        file_path = os.path.join(folder_path, file_name)
        print(file_path)

        check_pc = o3d.io.read_point_cloud(file_path)
        if len(check_pc.points) ==0:
            output = np.eye(3)
        else:
            output = get_PCA(file_path)
        print(output)
        '''
        if file_name.split("_")[0] == '000001':
            # Read point cloud data
            point_cloud = o3d.io.read_point_cloud("output/test_pointclouds/000001_pointcloud.ply")

            # Define rotation matrix
            rotation_matrix = output
            print(output)
            # Perform rotation
            rotated_point_cloud = point_cloud.rotate(rotation_matrix)

            # Save the rotated point cloud as a PLY file
            o3d.io.write_point_cloud("output_cloud_tr.ply", rotated_point_cloud)
        '''
        Rvector = output.transpose().reshape(1, 9)
        calib_path = os.path.join(calib_folder, calib_file)

        with open(calib_path, "r") as file:
            lines = file.readlines()

        print("old",lines)
        Rvector_str = ' '.join(['%.5f' % x for x in Rvector.reshape(-1)])
        lines[0] = Rvector_str + "\n"
        lines[0] = Rvector_str.strip('[]') + "\n"
        print("new",lines)
        # Write the changes back to the file
        calib_save_path_fianl = os.path.join(calib_save_path, file_name.split('_')[0] +'.txt')
        with open(calib_save_path_fianl, "w") as file:
            file.writelines(lines)


if __name__ == "__main__":
    pc_path = "./demo/output/visual/000001_bbox_0.ply"  # Replace with your actual point cloud file path
    rotation_matrix = get_PCA(pc_path)
    rotation_matrix = rotation_matrix.T
    print(rotation_matrix)

'''
Check the extrinsic of PC_LIFTING_BBOX_GEN_Module/IMG2PC/demo/calib/000001.txt, they are same
[[ 0.98373094  0.01961008  0.17857456]
 [ 0.01961008  0.97636278 -0.21524675]
 [-0.17857456  0.21524675  0.96009372]]
'''
