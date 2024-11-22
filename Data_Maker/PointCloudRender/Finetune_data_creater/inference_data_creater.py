import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
from collections import defaultdict
from PIL import Image  # Import the PIL library for image saving
import os


class partial_view(object):
    def __init__(self):
        # Initialize the class properties here
        self.attribute = "partial_view"
        self.Rtilt = np.eye(3)
        #self.W  = random.choice([480, 540, 600, 640])
        self.factors_1 = 380
        self.factors_2 = 240
        self.factors = 0
        self.f = 0.5 * self.factors / np.tan(0.5 * 55 * np.pi / 180.0)
        #hidden_removal
        self.rotation_angle_v = [0]
        self.rotation_angle_h = [0]
        
        self.hidden_removal_factors = 10000
        self.render_angle = [75,60,45,30,15,0,-15,-30,-45,-60,-75]
       
        
        
    def depth_to_points(self,depth, R=None, t=None):
        K = self.K
        Kinv = np.linalg.inv(K)
        angle_x = np.radians(-90)
        Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
        # angle_y = np.radians(-10)
        # Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],[0, 1, 0],[-np.sin(angle_y), 0, np.cos(angle_y)]])
        R = self.Rtilt@Rx
        # t = np.array([0, -1, 0.5])
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)

        # M converts from your coordinate to PyTorch3D's coordinate system
        M = np.eye(3)
        M[0, 0] = -1.0
        M[1, 1] = -1.0

        height, width = depth.shape[1:3]

        x = np.arange(width)
        y = np.arange(height)
        coord = np.stack(np.meshgrid(x, y), -1)
        coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
        coord = coord.astype(np.float32)
        # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
        coord = coord[None]  # bs, h, w, 3

        D = depth[:, :, :, None, None]

        scales = 256
        # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
        pts3D_1 = D / scales * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
        # pts3D_1 live in your coordinate system. Convert them to Py3D's
        # pts3D_1 = M[None, None, None, ...] @ pts3D_1
        # from reference to targe tviewpoint
        pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
        # pts3D_2 = pts3D_1
        # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
        return pts3D_2[:, :, :, :3, 0][0]

    def depth2pc(self,color_image, depth_image):

        # 归一化每个像素的深度值
        depth_image=(depth_image*self.depth_factor*256).astype('uint16')

        pts3d = self.depth_to_points(depth_image[None])
        pts3d = pts3d.reshape(-1, 3)

        # Get RGB image
        rgb = np.array(color_image)
        # Convert to Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d)
        pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)

        o3d.io.write_point_cloud("./output.ply", pcd, write_ascii=True)

    def flip_axis_to_camera(self,pc):
        ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
            Input and output are both (N,3) array
        '''
        pc2 = np.copy(pc)
        pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
        pc2[:, 1] *= -1
        return pc2
    
    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        angle_list = []
        R_list = []
        centroid_distance_list = []
        for angle in self.render_angle:
            angle_list.append(angle)
            R = np.array([[1, 0, 0],
                            [0, np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                            [0, np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
            R_list.append(R)
            pc2_temp = np.dot(np.transpose(R), np.transpose(pc[:,0:3])) # (3,n)
            centroid = np.mean(pc2_temp[:, :3].T, axis=0)
            distances = np.sqrt(centroid[0] ** 2 + centroid[2] ** 2)
            centroid_distance_list.append(distances)
        # Find the index of the smallest centroid_distance_list value
        min_distance_index = np.argmin(centroid_distance_list)
        # Get the corresponding R matrix
        best_R = R_list[min_distance_index]
        best_angle = angle_list[min_distance_index]
        # Print the minimum distance and corresponding R matrix
        print(" Minimum distance :", centroid_distance_list[min_distance_index])
        print(" corresponding R matrix :", best_R,best_angle)
        self.Rtilt = best_R
        pc2 = np.dot(np.transpose(self.Rtilt), np.transpose(pc[:, 0:3]))  # (3,n)
        return self.flip_axis_to_camera(np.transpose(pc2))
    
    # def project_upright_depth_to_camera(self, pc):
    #     ''' project point cloud from depth coord to camera coordinate
    #         Input: (N,3) Output: (N,3)
    #     '''
    #     # Project upright depth to depth coordinate
    #     pc2 = np.dot(np.transpose(self.Rtilt), np.transpose(pc[:,0:3])) # (3,n)
    #     return self.flip_axis_to_camera(np.transpose(pc2))

    def scale_court(self,pc2,factors,cx,cy):
        f = 0.5 * factors / np.tan(0.5 * 55 * np.pi / 180.0)
        K = np.array([[f, 0, cx/2],
                     [0, f, cy/2],
                     [0, 0, 1]])
    
        uv = np.dot(pc2, np.transpose(K)) # (n,3)

        uv[:,0] /= uv[:,2]
        uv[:,1] /= uv[:,2]

        min_uv = np.min(uv, axis=0)
        uv -= min_uv
        K[0][2] -= min_uv[0]
        K[1][2] -= min_uv[1]
        uv = uv[:,0:2]
        depth =  pc2[:,2]
        h, w = int(np.max(uv[:, 1])) + 1, int(np.max(uv[:, 0])) + 1
        depth_image = np.zeros((h, w))
        
        for i in range(uv.shape[0]):
            u, v = int(uv[i, 0]), int(uv[i, 1])
            depth_image[v, u] = depth[i]
        return depth_image.shape[0],depth_image.shape[1]

    def project_upright_depth_to_image(self, pc):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc_xyz = pc[:,:3]
        pc_rgb = pc[:,3:]

        pc2 = self.project_upright_depth_to_camera(pc_xyz)

        # Extract x and y coordinates
        x_coordinates = pc_xyz[:, 0]
        z_coordinates = pc_xyz[:, 2]

        # Calculate the range in the x and y directions
        x_min = np.min(x_coordinates)
        x_max = np.max(x_coordinates)
        z_min = np.min(z_coordinates)
        z_max = np.max(z_coordinates)

        # Calculate the length and width of the x and y directions
        x_length = x_max - x_min
        z_length = z_max - z_min
        
        if z_length/x_length<1:
            self.factors = self.factors_1
        else:
            self.factors = self.factors_2
        # If the larger value is greater than 800, gradually reduce the self.factors
        k02, k12 = self.scale_court(pc2,self.factors,self.factors, self.factors*(z_length/x_length))
        print("1",k02, k12)
        while max(k02, k12) > 800:
            k02, k12 = self.scale_court(pc2,self.factors,self.factors, self.factors*(z_length/x_length))
            print('dndn',self.factors)
            self.factors *= 0.95 # Here you can reduce self.factors by 10% each time, you can adjust this value as needed

        k02, k12 = self.scale_court(pc2,self.factors,self.factors, self.factors*(z_length/x_length))
        print("2",k02, k12)
        # If the smaller value is less than 320, gradually increase self.factors
        while min(k02, k12) < 300:
            k02, k12 = self.scale_court(pc2,self.factors,self.factors, self.factors*(z_length/x_length))
            print('upup',self.factors)
            self.factors *= 1.05  # Here you increase self.factors by 10% each time, you can adjust this value as needed

        self.f = 0.5 * self.factors / np.tan(0.5 * 55 * np.pi / 180.0)
        self.K = np.array([[self.f, 0, self.factors/2],
                     [0, self.f, self.factors*(z_length/x_length)/2],
                     [0, 0, 1]])

        uv = np.dot(pc2, np.transpose(self.K)) # (n,3)

        uv[:,0] /= uv[:,2]
        uv[:,1] /= uv[:,2]

        self.min_uv = np.min(uv, axis=0)
        uv -= self.min_uv
        self.K[0][2] -= self.min_uv[0]
        self.K[1][2] -= self.min_uv[1]
        return uv[:,0:2], pc2[:,2], pc_rgb

    def rotation(self,angle_v,angle_h):
        # Create a rotation transformation matrix
        R_v = np.array([[1, 0, 0],
                        [0, np.cos(np.radians(angle_v)), -np.sin(np.radians(angle_v))],
                        [0, np.sin(np.radians(angle_v)), np.cos(np.radians(angle_v))]])

        R_h = np.array([[np.cos(np.radians(angle_h)), -np.sin(np.radians(angle_h)), 0],
                        [np.sin(np.radians(angle_h)), np.cos(np.radians(angle_h)), 0],
                        [0, 0, 1]])
        return R_v,R_h

    def display_point_cloud_after_hidden_removal(self,points):
        # 1. Create an Open3D point cloud object
        pcd_hidden = o3d.geometry.PointCloud()
        pcd_hidden.points = o3d.utility.Vector3dVector(points[:,:3])

        # 2. Define parameters for hidden point removal
        diameter = np.linalg.norm(
            np.asarray(pcd_hidden.get_max_bound()) - np.asarray(pcd_hidden.get_min_bound()))
        camera = [0, 0, 0]
        radius = diameter * self.hidden_removal_factors  # float

        # 3. Get all points visible from a given viewpoint
        _, pt_map = pcd_hidden.hidden_point_removal(camera_location=camera, radius=radius)  # pt_map: idx list

        return pt_map
    def one_step(self,pc):
        uv, depth, pc_rgb = self.project_upright_depth_to_image(pc)

        # Calculate the image size based on uv coordinates
        h, w = int(np.max(uv[:, 1])) + 1, int(np.max(uv[:, 0])) + 1

        # Create empty arrays for depth and color images
        depth_image = np.zeros((h, w))
        color_image = np.zeros((h, w, 3), dtype=np.uint8)

        # Populate the images
        for i in range(uv.shape[0]):
            u, v = int(uv[i, 0]), int(uv[i, 1])
            depth_image[v, u] = depth[i]
            color_image[v, u] = pc_rgb[i]

        # Normalize the depth image for visualization
        self.depth_factor = np.max(depth_image) - np.min(depth_image)
        depth_image = (depth_image - np.min(depth_image)) / self.depth_factor
        return color_image,depth_image

    def forward(self,pc,rotation_flag=True):

        if rotation_flag:
            angle_v = random.choice(self.rotation_angle_v)
            angle_h = random.choice(self.rotation_angle_h)
            # print(angle_v,angle_h)
            R_v,R_h = self.rotation(angle_v,angle_h)
            centroid = np.mean(pc[:,:3], axis=0)
            # Pan to the origin, rotate, pan back
            R = np.dot(R_v, R_h)
            pc_backup =pc.copy()
            pc_backup[:,:3] = np.dot(pc_backup[:,:3] - centroid, R.T) + centroid
            unique_indices =  self.display_point_cloud_after_hidden_removal(pc_backup[:,:3])
            # print(pc.shape,len(unique_indices))
            color_image, depth_image = self.one_step(pc[unique_indices])
            
        else:
            color_image, depth_image = self.one_step(pc)

        #self.depth2pc(color_image, depth_image)
        print("self.Rtilt",self.Rtilt)
        return color_image, depth_image,self.Rtilt,self.K

def process_point_clouds_single(pcd_directory, depth_directory, color_directory,calib_directory):
    # Create the partial_view object outside the loop
    pv = partial_view()
    
    # List all files in the directory
    file_list = os.listdir(pcd_directory)
    
    for file_name in file_list:
        # Construct the full file path
        file_path = os.path.join(pcd_directory, file_name)
        print("file_name:", file_name)
        
        # Ensure we're processing a .ply file
        if not file_path.endswith('.ply'):
            continue
        
        # Load the PLY file using Open3D
        pcd = o3d.io.read_point_cloud(file_path)
        
        # Extract the point cloud and color data
        pc = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255  # Assuming colors are in [0,1]. Convert to [0,255]
        pc_full = np.hstack([pc, colors])

        # Use the partial_view class to project the point cloud to an image
        color_image, depth_image,Rtilt,K = pv.forward(pc_full)
        
        # Save depth and color images as separate files
        depth_image = (depth_image * 255).astype(np.uint8)  # Convert to 8-bit
        print("depth_image",depth_image.shape)
        depth_image = Image.fromarray(depth_image)
        save_filename=file_name.split("_")[0]
        depth_image.save(os.path.join(depth_directory, f'{save_filename}_depth.png'))
        
        color_image = Image.fromarray(color_image)
        color_image.save(os.path.join(color_directory, f'{save_filename}_color.png'))
        
        R_flattened = Rtilt.T.flatten()
        K_flattened = K.T.flatten()
        # Save R and K matrices
        calib_data = np.vstack([R_flattened, K_flattened])
        np.savetxt(os.path.join(calib_directory, f'{save_filename}.txt'), calib_data)

import multiprocessing
from scipy.io import loadmat
def process_single_file(file_name, pcd_directory, depth_directory, color_directory, calib_directory):
    pv = partial_view()  # If this class has no side effects, we can instantiate it in each process
    
    # Construct the full file path
    file_path = os.path.join(pcd_directory, file_name)
    print("file_name:", file_name,file_name.split('.')[0])
    
    # Ensure we're processing a .ply file
    if not file_path.endswith('_pc.ply'):
        return
    
    # Load the PLY file using Open3D
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Extract the point cloud and color data
    pc = np.asarray(pcd.points)

    colors = np.asarray(pcd.colors) * 255  # Assuming colors are in [0,1]. Convert to [0,255]

    pc_full = np.hstack([pc, colors])

    
    # Use the partial_view class to project the point cloud to an image
    color_image, depth_image,Rtilt,K = pv.forward(pc_full)
    
    # Save depth and color images as separate files
    depth_image = (depth_image * 255).astype(np.uint8)  # Convert to 8-bit
    print("depth_image",depth_image.shape)
    depth_image = Image.fromarray(depth_image)
    save_filename=file_name.split("_pc.")[0]
    depth_image.save(os.path.join(depth_directory, f'{save_filename}.png'))
    
    color_image = Image.fromarray(color_image)
    color_image.save(os.path.join(color_directory, f'{save_filename}_color.png'))
    R_flattened = Rtilt.T.flatten()
    K_flattened = K.T.flatten()
    # Save R and K matrices
    calib_data = np.vstack([R_flattened, K_flattened])
    np.savetxt(os.path.join(calib_directory, f'{save_filename}.txt'), calib_data)
    print(save_filename,"done")

def process_point_clouds(pcd_directory, depth_directory, color_directory,calib_directory):
    # List all files in the directory
    file_list =  [f for f in os.listdir(pcd_directory) if f.endswith('.ply')]
    
    # Define the number of processes based on your machine's capability
    num_processes = multiprocessing.cpu_count()  # This gets the number of CPU cores in your machine
    
    # Create a Pool and use it to parallelize the processing
    with multiprocessing.Pool(num_processes) as pool:
        pool.starmap(process_single_file, [(f, pcd_directory, depth_directory, color_directory,calib_directory) for f in file_list])

pcd_directory = "./demo_input/pointcloud_inference"
depth_directory = "./demo_output/source_inference"
color_directory = "./demo_output/target_inference"
calib_directory = "./demo_output/calib_inference"
if not os.path.exists(depth_directory):
    os.makedirs(depth_directory)
if not os.path.exists(color_directory):
    os.makedirs(color_directory)
if not os.path.exists(calib_directory):
    os.makedirs(calib_directory)
process_point_clouds_single(pcd_directory, depth_directory, color_directory,calib_directory)
#process_point_clouds(pcd_directory, depth_directory, color_directory,calib_directory)