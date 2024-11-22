import os
import cv2
import json
from PIL import Image
import open3d as o3d
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from math import gcd
import trimesh
import scipy.io as sio
import threading

#COCO_label= {0: 'bed', 1: 'table', 2: 'sofa', 3: 'chair', 4: 'toilet', 5: 'desk', 6: 'dresser', 7: 'night_stand', 8: 'bookshelf', 9: 'bathtub'}
COCO_label = {"bed": 0, "dresser": 1,"night_stand": 2, "ottoman": 3, "dresser_mirror": 4 }
COCO_label = {value: key for key, value in COCO_label.items()}

def check_rotation_matrix(R):
    # Step 3: Check matrix integrity and presence of None or NaN values
    has_none = np.any(np.isnan(R))
    has_nan = np.any(np.isnan(R))

    if has_none or has_nan:
        # Return identity matrix if matrix has None or NaN values
        return np.eye(3)

    return R

def heading2rotmat(heading_angle):
    rotmat = np.zeros((3,3))
    rotmat[2,2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2,0:2] = np.array([[cosval, -sinval],[sinval, cosval]])
    return rotmat

def compute_bbox(in_pc):
    pca = PCA(2)
    pca.fit(in_pc[:,:2])
    yaw_vec = pca.components_[0,:]
    yaw = np.arctan2(yaw_vec[1],yaw_vec[0])
    in_pc_tmp = in_pc.copy()
    in_pc_tmp = heading2rotmat(-yaw) @ in_pc_tmp[:,:3].T
    x_min = in_pc_tmp[0,:].min()
    x_max = in_pc_tmp[0,:].max()
    y_min = in_pc_tmp[1,:].min()
    y_max = in_pc_tmp[1,:].max()
    z_min = in_pc_tmp[2,:].min()
    z_max = in_pc_tmp[2,:].max()
    dx = x_max-x_min
    dy = y_max-y_min
    dz = z_max-z_min
    bbox = heading2rotmat(yaw) @ np.array([[x_min,y_min,z_min],[x_max,y_max,z_max]]).T
    bbox = bbox.T
    x_min,y_min,z_min = bbox[0]
    x_max,y_max,z_max = bbox[1]
    cx = (x_min+x_max)/2
    cy = (y_min+y_max)/2
    cz = (z_min+z_max)/2
    rst_bbox = np.expand_dims(np.array([cx, cy, cz, dx/2, dy/2, dz/2, -1*yaw]), axis=0)
    sunrgbdformat_bbox = np.expand_dims(np.array([cx, cy, cz, dy / 2, dx / 2, dz / 2,yaw_vec[0],yaw_vec[1]]), axis=0)
    #print(rst_bbox.shape)
    #write_oriented_bbox(rst_bbox, "rst.ply")
    #write_ply(in_pc[:,:3], "pc.ply")
    #print(cx, cy, cz, dx, dy, dz, yaw)
    #exit()
    return rst_bbox,sunrgbdformat_bbox


def write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes
    Args:
        scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
            and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename: (string) filename
    """

    def heading2rotmat(heading_angle):
        pass
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to ply file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='ply')

    return





def get_intrinsics(H, W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)

    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])


def depth_to_points(depth, R=None,K=None, t=None):
    #if K is None:
    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    angle_x = np.radians(-90)
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    R = R@Rx
    #R = Rx
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
    #print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D/scales * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    #pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0]

def depth_edges_mask(depth,thr):
    """Returns a mask of edges in the depth map.
    Args:
    depth: 2D numpy array of shape (H, W) with dtype float32.
    Returns:
    mask: 2D numpy array of shape (H, W) with dtype bool.
    """
    # Compute the x and y gradients of the depth map.
    depth_dx, depth_dy = np.gradient(depth)
    # Compute the gradient magnitude.
    depth_grad = np.sqrt(depth_dx ** 2 + depth_dy ** 2)
    # Compute the edge mask.
    mask = depth_grad > thr  # 0.01
    return mask


def normalize_depth(depth, min_depth=None, max_depth=None):

    if min_depth is None:
        min_depth = np.min(depth)
    if max_depth is None:
        max_depth = np.max(depth)

    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    return normalized_depth



def get_pointcloud_and_bbox(image, depth,Rtilt,K, output_filepath,thr =0.05,remove_edges=True,point_cloud=True):
    #image.thumbnail((1024, 1024))  # limit the size of the input image
    depth = np.array(depth).astype(np.uint16)

    if remove_edges:
        # Compute the edge mask.
        depth_remove =normalize_depth(depth)
        mask = depth_edges_mask(depth_remove,thr)
        # Filter the depth map using the edge mask.
        depth[mask] = 0.0

    pts3d = depth_to_points(depth[None],R=Rtilt,K = K)
    pts3d = pts3d.reshape(-1, 3)

    # Get RGB image
    rgb = np.array(image.convert('RGB'))
    # Convert to Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3) / 255.0)

    #sunrgbd_format.
    y_values = pts3d[:, 1]  # Extract the Y-axis value
    nonzero_indices = np.nonzero(y_values) # Find an index with a non-zero y value
    pcd = pcd.select_by_index(nonzero_indices[0]) # Delete the corresponding point by index

    if point_cloud:

        # Save as ply
        ply_path = output_filepath
        o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)

        point_cloud_data = {
            'instance': np.hstack([np.asarray(pcd.points), np.asarray(pcd.colors)]).astype(np.float32),
        }

       # Save as MAT file
        file_name = os.path.splitext(os.path.basename(output_filepath))[0].replace("_pointcloud","")
        print(file_name)
        mat_filepath = f"./output/sunrgbd_trainval/depth/{file_name}.mat"
        sio.savemat(mat_filepath, point_cloud_data)
        print("Number of points in point cloud:", len(pts3d))
        print("Path to saved ply file:", ply_path)
        return pcd
    else:
        # Get RGB channel
        colors = np.asarray(pcd.colors)
        # Get the index of the black dot
        black_indices = np.where(np.all(colors == [0, 0, 0], axis=1))[0]
        # Delete the black dot
        pcd_without_black = pcd.select_by_index(np.delete(np.arange(len(colors)), black_indices))
        # Save as ply
        ply_path = output_filepath
        #o3d.io.write_point_cloud(ply_path, pcd_without_black, write_ascii=True)
        return pcd_without_black




def process_single_image_pair(rgb_filenames, depth_filenames,calib_filenames, rgb_folder, depth_folder,calib_folder , output_folder, bbox_output_folder, annotations):
    for rgb_filename, depth_filename, calib_filename in zip(rgb_filenames, depth_filenames,calib_filenames):

        if (rgb_filename.split('.')[0] != depth_filename.split('.')[0]) or (rgb_filename.split('.')[0] != calib_filename.split('.')[0]):
            print("rgb_filename, depth_filename", rgb_filename, depth_filename,calib_filename)
            raise ValueError("If the file name is different, the operation is terminated.")

        rgb_filepath = os.path.join(rgb_folder, rgb_filename)
        depth_filepath = os.path.join(depth_folder, depth_filename)
        calib_filepath = os.path.join(calib_folder, calib_filename)


        image = Image.open(rgb_filepath)
        image = image.convert('RGB')
        if depth_filepath.endswith('.pfm'):
           
            depth = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)           

            depth = Image.fromarray(depth)
        elif depth_filepath.endswith('.png'):
           
            depth = Image.open(depth_filepath)
        else:
            raise ValueError('Unknown file format: {}'.format(depth_filepath))

        lines = [line.rstrip() for line in open(calib_filepath)]
        Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        Rtilt = np.reshape(Rtilt, (3, 3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        K = np.reshape(K, (3, 3), order='F')
        print("R",Rtilt)
        print("K",K)

       
        output_filename = f"{Path(rgb_filename).stem}_pointcloud.ply"
        output_filepath = os.path.join(output_folder, output_filename)

       
        bbox_output_filename = f"{Path(rgb_filename).stem}_bbox.ply"
        bbox_output_filepath = os.path.join(bbox_output_folder, bbox_output_filename)
        # Generate a point cloud file
        pcd_rgb = get_pointcloud_and_bbox(image, depth,Rtilt,K, output_filepath,point_cloud=True)

        # Extract segmentation information
        num_str = rgb_filepath.split("/")[-1].split(".")[0]

        # Create empty mask
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Create empty masks arrayget_pointcloud
        H,W = image.shape[0], image.shape[1]
        image_anns = [ann for ann in annotations['annotations'] if ann['image_id'] == int(num_str)]
        masks = np.zeros((len(image_anns), H, W), np.uint8)
        classes=[]
        bounding_boxes2d = []
        '''
        for i, ann in enumerate(image_anns):
            if isinstance(ann['segmentation'], dict):
                continue
            bbox2d = ann['bbox']
            bounding_boxes2d.append(bbox2d)
            seg = ann['segmentation'][0]
            classes.append(ann['category_id'])
            pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(masks[i,:,:],[pts], 255)
        '''
        for i, ann in enumerate(image_anns):
            bbox2d = ann['bbox']
            bounding_boxes2d.append(bbox2d)
            classes.append(ann['category_id'])
            # If your segmentation data contains multiple lists,
            # iterate over each list (polygon) in the segmentation data
            for seg in ann['segmentation']:
                pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
                # The fillPoly function is called for each polygon
                cv2.fillPoly(masks[i,:,:],[pts], 255)
        

        print("calsses",classes)
        instances=[]
        instances_sample=[]
        instances_DBSCAN = []
        bboxs=[]
        valid_ins=[]
        sunrgbdformat_bboxs = []
        DBSCAN_flag=True
        for i, mask in enumerate(masks):
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = Image.fromarray(mask)
            pcd= get_pointcloud_and_bbox(mask, depth,Rtilt,K, bbox_output_filepath,thr=0.005,point_cloud=False)
            instances.append(pcd)
           
            _, unique_indices = np.unique(pcd.points, axis=0, return_index=True)
            pc_for_bbox = np.hstack([np.asarray(pcd.points), np.asarray(pcd.colors)])[unique_indices]
            if DBSCAN_flag:
                step_interval = max((1, int(pc_for_bbox.shape[0] / 3000)))
                cur_ins_pc = pc_for_bbox[0:pc_for_bbox.shape[0]:step_interval, :]

                if cur_ins_pc.shape[0] < 100:
                    print("loss")
                    continue
                instances_sample.append(cur_ins_pc)

                db = DBSCAN(eps=0.3, min_samples=100).fit(cur_ins_pc)

                cur_ins_pc_remove_outiler = []
                for cluster in np.unique(db.labels_):
                    if cluster < 0:
                        continue

                    cluster_ind = np.where(db.labels_ == cluster)[0]

                    if cluster_ind.shape[0] / cur_ins_pc.shape[0] < 0.2 or cluster_ind.shape[0] <= 100:
                        continue
                    cur_ins_pc_remove_outiler.append(cur_ins_pc[cluster_ind, :])

                if len(cur_ins_pc_remove_outiler) < 1:
                    continue

                valid_ins.append(i)

                pc_for_bbox = np.concatenate(cur_ins_pc_remove_outiler, axis=0)

                instances_DBSCAN.append(pc_for_bbox)

            bbox,sunrgbdformat_bbox = compute_bbox(pc_for_bbox)
            bboxs.append(bbox)
            sunrgbdformat_bboxs.append(sunrgbdformat_bbox)


      
        print("len(bboxs)",len(bboxs))
        if len(bboxs)>=1:

            bboxs = np.concatenate(bboxs, axis=0)
            if DBSCAN_flag==False:
                valid_ins = list(range(len(classes)))
            valid_classes = [classes[i] for i in valid_ins]
            print("classes",classes)
            print("valid_classes",valid_classes)
            print("valid_ins",valid_ins)
            bboxs = np.concatenate([bboxs, np.expand_dims(valid_classes, axis=1)], axis=1)
            pseudo_label_filename = os.path.join(bbox_output_folder, "%s_%s_bbox" % (num_str,"_"))
            np.save(pseudo_label_filename, bboxs)

            txt_path = "./output/sunrgbd_trainval/label"
            txt_file_path = os.path.join(txt_path, f"{num_str}.txt")
            #create_SUNRGBDformat_label
            txt_write=[]
            for i,ins in enumerate(valid_ins):
                a= COCO_label[valid_classes[i]].replace(' ','_')
                b = str(bounding_boxes2d[ins]).replace("[", "").replace("]", "").replace(",", "")
                c = str(sunrgbdformat_bboxs[i][0][:]).replace("[", "").replace("]", "").replace("\n", "")
                sunrgbdformat_label = f"{a} {b} {c}\n"
                txt_write.append(sunrgbdformat_label.replace("  ", " ").replace("  ", " "))
            txt_str = ''.join(txt_write)
            with open(txt_file_path, 'w') as file:
                file.write(txt_str)

            bboxs[:, 3:6] *= 2
            bboxs[:, 6] *= -1
            print(bboxs.shape)
            write_oriented_bbox(bboxs[:, :7], "%s.ply" % (pseudo_label_filename))

        print(f"Saved point cloud for {depth_filename} {calib_filename} {rgb_filename} to {output_filepath}")

def key_callback(event):
    global pressed_q
    if event.name == 'q':
        pressed_q = True

import multiprocessing
def process_images(rgb_folder, depth_folder, calib_folder, instance_json_path, output_folder, bbox_output_folder):
    rgb_filenames = sorted(os.listdir(rgb_folder))
    depth_filenames = sorted(os.listdir(depth_folder))
    calib_filenames = sorted(os.listdir(calib_folder))
    
    # Load the instance JSON file
    with open(instance_json_path, "r") as f:
        annotations = json.load(f)

    num_processes = multiprocessing.cpu_count()
    processes = []

    num_files = len(rgb_filenames)
    batch_size = num_files // num_processes

    for i in range(num_processes):
        start = i * batch_size
        end = (i + 1) * batch_size if i < num_processes - 1 else num_files

        process = multiprocessing.Process(
            target=process_single_image_pair,
            args=(rgb_filenames[start:end], depth_filenames[start:end], calib_filenames[start:end], rgb_folder, depth_folder, calib_folder, output_folder, bbox_output_folder, annotations)
        )

        processes.append(process)
        process.start()

    for process in processes:
        process.join()


import time
if __name__ == "__main__":

    rgb_folder = './image'
    depth_folder = './depth' #from depth estimation[ZoeDepth]
    calib_folder = './calib' #after revised extrinsic, and intrinsic is not used
    instance_json_path = './data_cocoformat.json' #COCO format json
    output_folder = './output/test_pointclouds'
    bbox_output_folder = './output/test_bbox_pointclouds'

    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(bbox_output_folder).mkdir(parents=True, exist_ok=True)

    sunrgbd_trainval = './output/sunrgbd_trainval'
    sunrgbd_trainval_depth = sunrgbd_trainval+'/depth'
    sunrgbd_trainval_label = sunrgbd_trainval+'/label'
    os.makedirs(sunrgbd_trainval, exist_ok=True)
    os.makedirs(sunrgbd_trainval_depth, exist_ok=True)
    os.makedirs(sunrgbd_trainval_label, exist_ok=True)

    start_time = time.time()
    process_images(rgb_folder, depth_folder, calib_folder,instance_json_path, output_folder, bbox_output_folder)
    end_time = time.time()
    execution_time = end_time - start_time

    print("代码执行时间：", execution_time, "秒")

