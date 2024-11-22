# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for 3D object detection on SUN RGB-D (with additional support for ImVoteNet).

A lvis oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Modified by: Xinlei Chen
Date: 2020

"""
import os
import sys
import numpy as np
import tqdm
from torch.utils.data import Dataset
import scipy.io as sio # to load .mat files for depth points
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import lvis_utils
from model_util_lvis import lvisDatasetConfig
import pickle
DC = lvisDatasetConfig() # dataset specific config
MAX_NUM_OBJ = 64 # maximum number of objects allowed per scene
MEAN_COLOR_RGB = np.array([0.5,0.5,0.5]) # lvis color is in 0~1

NUM_CLS = 1204 # lvis number of classes
CLIP_TEXT_FEATRUE = 512 # lvis number of classes
MAX_NUM_2D_DET = 100 # maximum number of 2d boxes per image
MAX_NUM_PIXEL = 1570*800 # maximum number of pixels per image


class lvisDetectionVotesDataset(Dataset):
    def __init__(self, 
                 split_set='train',
                 num_points=20000,
                 use_color=False,
                 use_height=False,
                 use_imvote=False,
                 max_imvote_per_pixel=3,
                 augment=False,
                 scan_idx_list=None):

        assert(num_points<=50000)
        self.split_set =split_set
        self.train = split_set == 'train'

        self.data_path = os.path.join(ROOT_DIR,
            'lvis/lvis_pc_bbox_votes_%s'%(split_set))
        self.bbox_2d_path = os.path.join(ROOT_DIR,
            'lvis/lvis_2d_bbox_%s'%(split_set))

        if self.train:
            print("training set!!")
            self.raw_data_path = os.path.join(ROOT_DIR, 'lvis/lvis_trainval_train')
       
            with open(os.path.join(ROOT_DIR, 'datasetlist/train_lvis.txt'), 'r') as file:
                file_ids = [line.strip() for line in file]
            self.scan_names = file_ids
       
        else:
            self.raw_data_path = os.path.join(ROOT_DIR, 'lvis/lvis_trainval_eval')
          
            with open(os.path.join(ROOT_DIR, 'datasetlist/sunrgbd.txt'), 'r') as file:
                file_ids = [line.strip() for line in file]
            self.scan_names = file_ids
        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.use_imvote = use_imvote
        self.max_imvote_per_pixel = max_imvote_per_pixel
        self.vote_dims = 1+self.max_imvote_per_pixel*4
        # Total feature dimensions: geometric(5)+semantic(NUM_CLS)+texture(3) 
        self.image_feature_dim = CLIP_TEXT_FEATRUE + 5
        
        #self.pre_load_2d_bboxes()
        if self.train:
            self.cache_file = "lvis_train.pkl"  # Set the cache file path
        else:
            self.cache_file = "lvis_eval.pkl" 
        
        if os.path.exists(self.cache_file):
            self.load_from_cache()
        else:
            self.pre_load_2d_bboxes()
            self.save_to_cache()
        
    def save_to_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump((self.cls_id_map, self.cls_score_map, self.bbox_2d_map), f)
    def load_from_cache(self):
        with open(self.cache_file, 'rb') as f:
            self.cls_id_map, self.cls_score_map, self.bbox_2d_map = pickle.load(f)


    def get_intrinsics(self,H, W):
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


    def pre_load_2d_bboxes(self):
        self.cls_id_map = {}
        self.cls_score_map = {}
        self.bbox_2d_map = {}
        
        print("pre-loading 2d boxes from: " + self.bbox_2d_path)
        for scan_name in tqdm.tqdm(self.scan_names):
            # Read 2D object detection boxes and scores
            cls_id_list = []
            cls_score_list = []
            bbox_2d_list = []
            for line in open(os.path.join(self.bbox_2d_path, scan_name+'.txt'), 'r'):
                det_info = line.rstrip().split(" ")
                prob = float(det_info[-1])
                # Filter out low-confidence 2D detections
                if prob < 0.3:
                    continue
                cls_id_list.append(lvis_utils.type2class[det_info[0]])
                cls_score_list.append(prob)
                bbox_2d_list.append(np.array([float(det_info[i]) for i in range(4,8)]).astype(np.int32))
            self.cls_id_map[scan_name] = cls_id_list
            self.cls_score_map[scan_name] = cls_score_list
            self.bbox_2d_map[scan_name] = bbox_2d_list
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """
        scan_name = self.scan_names[idx]
        point_cloud = np.load(os.path.join(self.data_path, scan_name)+'_pc.npz')['pc'] # Nx6
        if self.train:
            revise_3dbox_path = './label_revise_extrinsic/' 
            bboxes = np.load(os.path.join(revise_3dbox_path, scan_name)+'_bbox.npy') # K,8

        else:
            bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy') # K,8
        bbox_num = bboxes.shape[0]
        if bbox_num > 64:
            bboxes=bboxes[:64,:]
            bbox_num = bboxes.shape[0]
        point_votes = np.load(os.path.join(self.data_path, scan_name)+'_votes.npz')['point_votes'] # Nx10
        if self.use_imvote:

            # Read camera parameters
            calib_lines = [line for line in open(os.path.join(self.raw_data_path, 'calib', scan_name+'.txt')).readlines()]
            calib_Rtilt = np.reshape(np.array([float(x) for x in calib_lines[0].rstrip().split(' ')]), (3,3), 'F')
            calib_K = np.reshape(np.array([float(x) for x in calib_lines[1].rstrip().split(' ')]), (3,3), 'F')

            if self.train:
                #optional: before render to 2D space, we first change it to original status
                extrinsic_revise_calib_path = './surface_normal_uncertainty-main/output_normal_scannetpth_green230_new/calib'
                calib_lines_ext = [line for line in open(os.path.join(extrinsic_revise_calib_path, scan_name+'.txt')).readlines()]
                calib_Rtilt_ext = np.reshape(np.array([float(x) for x in calib_lines_ext[0].rstrip().split(' ')]), (3,3), 'F')
                calib_Rtilt = np.dot(calib_Rtilt_ext,calib_Rtilt) 


            # Read image
            full_img = lvis_utils.load_image(os.path.join(self.raw_data_path, 'image', scan_name+'.png'))
            full_img_height = full_img.shape[0]
            full_img_width = full_img.shape[1]
            # ------------------------------- 2D IMAGE VOTES ------------------------------
            cls_id_list = self.cls_id_map[scan_name][:MAX_NUM_2D_DET]
            
            cls_score_list = self.cls_score_map[scan_name][:MAX_NUM_2D_DET]
            
            bbox_2d_list = self.bbox_2d_map[scan_name][:MAX_NUM_2D_DET]
            
            obj_img_list = []
            for i2d, (cls2d, box2d) in enumerate(zip(cls_id_list, bbox_2d_list)):
                xmin, ymin, xmax, ymax = box2d
                # During training we randomly drop 2D boxes to reduce over-fitting
                if self.train and np.random.random()>0.5:
                    continue

                obj_img = full_img[ymin:ymax, xmin:xmax, :]
                obj_h = obj_img.shape[0]
                obj_w = obj_img.shape[1]
                # Bounding box coordinates (4 values), class id, index to the semantic cues
                meta_data = (xmin, ymin, obj_h, obj_w, cls2d, i2d)
                if obj_h == 0 or obj_w == 0:
                    continue

                # Use 2D box center as approximation
                uv_centroid = np.array([int(obj_w/2), int(obj_h/2)])
                uv_centroid = np.expand_dims(uv_centroid, 0)

                v_coords, u_coords = np.meshgrid(range(obj_h), range(obj_w), indexing='ij')
                img_vote = np.transpose(np.array([u_coords, v_coords]), (1,2,0))
                img_vote = np.expand_dims(uv_centroid, 0) - img_vote 

                obj_img_list.append((meta_data, img_vote))

            full_img_votes = np.zeros((full_img_height,full_img_width,self.vote_dims), dtype=np.float32)
            # Empty votes: 2d box index is set to -1
            full_img_votes[:,:,3::4] = -1.
            for obj_img_data in obj_img_list:
                meta_data, img_vote = obj_img_data
                u0, v0, h, w, cls2d, i2d = meta_data
                for u in range(u0, u0+w):
                    for v in range(v0, v0+h):
                        iidx = int(full_img_votes[v,u,0])
                        if iidx >= self.max_imvote_per_pixel: 
                            continue
                        full_img_votes[v,u,(1+iidx*4):(1+iidx*4+2)] = img_vote[v-v0,u-u0,:]
                        full_img_votes[v,u,(1+iidx*4+2)] = cls2d
                        full_img_votes[v,u,(1+iidx*4+3)] = i2d + 1 # add +1 here as we need a dummy feature for pixels outside all boxes
                full_img_votes[v0:(v0+h), u0:(u0+w), 0] += 1

            full_img_votes_1d = np.zeros((MAX_NUM_PIXEL*self.vote_dims), dtype=np.float32)
            full_img_votes_1d[0:full_img_height*full_img_width*self.vote_dims] = full_img_votes.flatten()

            # Semantic cues: one-hot vector for class scores
            cls_score_feats = np.zeros((1+MAX_NUM_2D_DET,NUM_CLS), dtype=np.float32)
            # First row is dumpy feature
            len_obj = len(cls_id_list)
            if len_obj:
                ind_obj = np.arange(1,len_obj+1)
                ind_cls = np.array(cls_id_list)
                cls_score_feats[ind_obj, ind_cls] = np.array(cls_score_list)

            # Texture cues: normalized RGB values
            full_img = (full_img - 128.) / 255.
            # Serialize data to 1D and save image size so that we can recover the original location in the image
            full_img_1d = np.zeros((MAX_NUM_PIXEL*3), dtype=np.float32)
            full_img_1d[:full_img_height*full_img_width*3] = full_img.flatten()

        if not self.use_color:
            point_cloud = point_cloud[:,0:3]
        else:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)

        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        scale_ratio = 1.
        if self.augment:
            flip_flag = (np.random.random()>0.5)
            if flip_flag:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                bboxes[:,0] = -1 * bboxes[:,0]
                bboxes[:,6] = np.pi - bboxes[:,6]
                point_votes[:,[1,4,7]] = -1 * point_votes[:,[1,4,7]]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            rot_mat = lvis_utils.rotz(rot_angle)

            point_votes_end = np.zeros_like(point_votes)
            point_votes_end[:,1:4] = np.dot(point_cloud[:,0:3] + point_votes[:,1:4], np.transpose(rot_mat))
            point_votes_end[:,4:7] = np.dot(point_cloud[:,0:3] + point_votes[:,4:7], np.transpose(rot_mat))
            point_votes_end[:,7:10] = np.dot(point_cloud[:,0:3] + point_votes[:,7:10], np.transpose(rot_mat))

            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            bboxes[:,0:3] = np.dot(bboxes[:,0:3], np.transpose(rot_mat))
            bboxes[:,6] -= rot_angle
            point_votes[:,1:4] = point_votes_end[:,1:4] - point_cloud[:,0:3]
            point_votes[:,4:7] = point_votes_end[:,4:7] - point_cloud[:,0:3]
            point_votes[:,7:10] = point_votes_end[:,7:10] - point_cloud[:,0:3]

            if self.use_imvote:
                R_inverse = np.copy(np.transpose(rot_mat))
                if flip_flag:
                    R_inverse[0,:] *= -1
                # Update Rtilt according to the augmentation
                # R_inverse (3x3) * point (3x1) transforms an augmented depth point
                # to original point in upright_depth coordinates
                calib_Rtilt = np.dot(np.transpose(R_inverse), calib_Rtilt) 
                
            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:,3:6] + MEAN_COLOR_RGB
                rgb_color *= (1+0.4*np.random.random(3)-0.2) # brightness change for each channel
                rgb_color += (0.1*np.random.random(3)-0.05) # color shift for each channel
                rgb_color += np.expand_dims((0.05*np.random.random(point_cloud.shape[0])-0.025), -1) # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0])>0.3,-1)
                point_cloud[:,3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random()*0.3+0.85
            if self.use_imvote:
                calib_Rtilt = np.dot(np.array([[scale_ratio,0,0],[0,scale_ratio,0],[0,0,scale_ratio]]), calib_Rtilt)
            scale_ratio_expand = np.expand_dims(np.tile(scale_ratio,3),0)
            point_cloud[:,0:3] *= scale_ratio_expand
            bboxes[:,0:3] *= scale_ratio_expand
            bboxes[:,3:6] *= scale_ratio_expand
            point_votes[:,1:4] *= scale_ratio_expand
            point_votes[:,4:7] *= scale_ratio_expand
            point_votes[:,7:10] *= scale_ratio_expand
            if self.use_height:
                point_cloud[:,-1] *= scale_ratio

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0],:] = bboxes

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in lvis_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here 
            box3d_size = bbox[3:6]*2
            if self.train: 
                size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            else:
                size_class, size_residual = DC.size2class_eval(box3d_size, DC.class2type_eval[semantic_class])
            box3d_centers[i,:] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i,:] = box3d_size

        target_bboxes_mask = label_mask 
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = lvis_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:,0])
            ymin = np.min(corners_3d[:,1])
            zmin = np.min(corners_3d[:,2])
            xmax = np.max(corners_3d[:,0])
            ymax = np.max(corners_3d[:,1])
            zmax = np.max(corners_3d[:,2])
            target_bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin])
            target_bboxes[i,:] = target_bbox

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        point_votes_mask = point_votes[choices,0]
        point_votes = point_votes[choices,1:]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:,0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:,-1] # from 0 to 9
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes
        if self.use_imvote:
            ret_dict['scale'] = np.array(scale_ratio).astype(np.float32)
            ret_dict['calib_Rtilt'] = calib_Rtilt.astype(np.float32)
            ret_dict['calib_K'] = calib_K.astype(np.float32)
            ret_dict['full_img_width'] = np.array(full_img_width).astype(np.int64)
            ret_dict['cls_score_feats'] = cls_score_feats.astype(np.float32)
            ret_dict['full_img_votes_1d'] = full_img_votes_1d.astype(np.float32)
            ret_dict['full_img_1d'] = full_img_1d.astype(np.float32)

        return ret_dict

def viz_votes(pc, point_votes, point_votes_mask):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]
    pc_obj_voted2 = pc_obj + point_votes[inds,3:6]
    pc_obj_voted3 = pc_obj + point_votes[inds,6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1.ply')
    pc_util.write_ply(pc_obj_voted2, 'pc_obj_voted2.ply')
    pc_util.write_ply(pc_obj_voted3, 'pc_obj_voted3.ply')

def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = DC.class2angle(angle_classes[i], angle_residuals[i])
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs.ply')
    pc_util.write_ply(label[mask==1,:], 'gt_centroids.ply')

def get_sem_cls_statistics():
    """ Compute number of objects for each semantic class """
    d = lvisDetectionVotesDataset(use_height=True, use_color=True, use_v1=True, augment=True)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i%10==0: print(i)
        sample = d[i]
        pc = sample['point_clouds']
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in sem_cls:
            if mask[j] == 0: continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)

if __name__=='__main__':
    d = lvisDetectionVotesDataset(use_height=True, use_color=True, use_v1=True, augment=True)
    sample = d[200]
    print(sample['vote_label'].shape, sample['vote_label_mask'].shape)
    pc_util.write_ply(sample['point_clouds'], 'pc.ply')
    viz_votes(sample['point_clouds'], sample['vote_label'], sample['vote_label_mask'])
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
        sample['heading_class_label'], sample['heading_residual_label'],
        sample['size_class_label'], sample['size_residual_label'])
