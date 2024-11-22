# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes
import pointnet2_utils
import clip


from config import get_flags,global_flag
FLAGS = get_flags(global_flag)
if FLAGS.dataset == 'sunrgbd':
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()

elif FLAGS.dataset == 'scannet':
    from model_util_scannet import scannetDatasetConfig
    DATASET_CONFIG = scannetDatasetConfig()

elif FLAGS.dataset == 'lvis':
    from model_util_lvis import lvisDatasetConfig
    DATASET_CONFIG = lvisDatasetConfig()




def decode_scores(net, end_points, num_class, num_heading_bin, num_size_cluster, mean_size_arr, key_prefix):
    net_transposed = net.transpose(2,1) # (batch_size, 1024, ..)
    batch_size = net_transposed.shape[0]
    num_proposal = net_transposed.shape[1]

    objectness_scores = net_transposed[:,:,0:2]
    end_points[key_prefix+'objectness_scores'] = objectness_scores
    
    base_xyz = end_points[key_prefix+'aggregated_vote_xyz'] # (batch_size, num_proposal, 3)
    center = base_xyz + net_transposed[:,:,2:5] # (batch_size, num_proposal, 3)
    end_points[key_prefix+'center'] = center

    heading_scores = net_transposed[:,:,5:5+num_heading_bin]
    heading_residuals_normalized = net_transposed[:,:,5+num_heading_bin:5+num_heading_bin*2]
    end_points[key_prefix+'heading_scores'] = heading_scores # Bxnum_proposalxnum_heading_bin
    end_points[key_prefix+'heading_residuals_normalized'] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
    end_points[key_prefix+'heading_residuals'] = heading_residuals_normalized * (np.pi/num_heading_bin) # Bxnum_proposalxnum_heading_bin

    size_scores = net_transposed[:,:,5+num_heading_bin*2:5+num_heading_bin*2+num_size_cluster]
    size_residuals_normalized = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster:5+num_heading_bin*2+num_size_cluster*4].view([batch_size, num_proposal, num_size_cluster, 3]) # Bxnum_proposalxnum_size_clusterx3
    end_points[key_prefix+'size_scores'] = size_scores
    end_points[key_prefix+'size_residuals_normalized'] = size_residuals_normalized
    end_points[key_prefix+'size_residuals'] = size_residuals_normalized * torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)

    sem_cls_scores = net_transposed[:,:,5+num_heading_bin*2+num_size_cluster*4:] # Bxnum_proposalx10
    end_points[key_prefix+'sem_cls_scores'] = sem_cls_scores
    return end_points


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256, key_prefix='pc_only_'):
        super().__init__() 
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.key_prefix = key_prefix
        self.pre_OV_head_dim = 256

        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes( 
                npoint=self.num_proposal,
                radius=0.3,
                nsample=16,
                mlp=[self.seed_feat_dim, 128, 128, 128],
                use_xyz=True,
                normalize_xyz=True
            )
    
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.conv1 = torch.nn.Conv1d(128,128,1)
        self.conv2 = torch.nn.Conv1d(128,128,1)
        #self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+self.num_class,1)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        
        ###############################OV - part-start###############################
        self.conv3 = torch.nn.Conv1d(128,2+3+num_heading_bin*2+num_size_cluster*4+ self.pre_OV_head_dim,1)
        
        self.clip_header = nn.Sequential(
            nn.Conv1d(in_channels=self.pre_OV_head_dim, out_channels=512, kernel_size=1, bias=True), 
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, bias=True)
            )

        self.text = ["a photo of " + str(item) for item in DATASET_CONFIG.type2class.keys()]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_clip, self.preprocess_clip = clip.load("ViT-B/32", device=device)
        text = clip.tokenize(self.text).to(device)
        self.text_feats = self.batch_encode_text(text)
        # self.text_feats = self.img_model.encode_text(text).detach()
        self.text_num = self.text_feats.shape[0]
        self.text_label = torch.arange(self.text_num, dtype=torch.int).to(device)
        
        self.eval_text = ["a photo of " + str(item) for item in DATASET_CONFIG.type2class_eval.keys()]
        self.eval_text.append("a photo of unknown category")
        self.eval_text.append("a photo of unclassified category")
        self.eval_text.append("a photo of other category")
        eval_text = clip.tokenize(self.eval_text).to(device)
        self.eval_text_feats = self.batch_encode_text(eval_text)\
        # self.text_feats = self.img_model.encode_text(text).detach()

        self.eval_text_num = self.eval_text_feats.shape[0] - 3
        self.eval_text_label = torch.arange(self.eval_text_num, dtype=torch.int).to(device)

    def batch_encode_text(self, text):
        batch_size = 20

        text_num = text.shape[0]
        cur_start = 0
        cur_end = 0

        all_text_feats = []
        while cur_end < text_num:
            # print(cur_end)
            cur_start = cur_end
            cur_end += batch_size
            if cur_end >= text_num:
                cur_end = text_num
            
            cur_text = text[cur_start:cur_end,:]
            cur_text_feats = self.model_clip.encode_text(cur_text).detach()
            all_text_feats.append(cur_text_feats)

        all_text_feats = torch.cat(all_text_feats, dim=0)
        # print(all_text_feats.shape)
        return all_text_feats
    
    def classify(self, image_features, text_features):
        device = image_features.device
        text_features = text_features.to(device)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logit_scale_ = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        logit_scale = logit_scale_.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def classify_pc(self, pc_query_feat, text_feat, text_num):
        query_num, batch_size, feat_dim = pc_query_feat.shape
        pc_query_feat = pc_query_feat.reshape([-1, feat_dim])

        logits_per_image, logits_per_text = self.classify(pc_query_feat.half(), text_feat)
        logits_per_image = logits_per_image.reshape([query_num, batch_size, -1]).permute(1,0,2)
        logits_per_text = logits_per_text.reshape([query_num, batch_size, -1]).permute(1,0,2)
        return logits_per_image.float(), logits_per_text.float()


        ###############################OV - part-end###############################

    def forward(self, xyz, features, end_points):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4) 
        """
        if self.sampling == 'vote_fps':
            # Farthest point sampling (FPS) on votes
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)
            sample_inds = fps_inds
        elif self.sampling == 'seed_fps': 
            # FPS on seed and choose the votes corresponding to the seeds
            # This gets us a slightly better coverage of *object* votes than vote_fps (which tends to get more cluster votes)
            sample_inds = pointnet2_utils.furthest_point_sample(end_points['seed_xyz'], self.num_proposal)
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        elif self.sampling == 'random':
            # Random sampling from the votes
            num_seed = end_points['seed_xyz'].shape[1]
            batch_size = end_points['seed_xyz'].shape[0]
            sample_inds = torch.randint(0, num_seed, (batch_size, self.num_proposal), dtype=torch.int).cuda()
            xyz, features, _ = self.vote_aggregation(xyz, features, sample_inds)
        else:
            log_string('Unknown sampling strategy: %s. Exiting!'%(self.sampling))
            exit()
        end_points[self.key_prefix+'aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
        end_points[self.key_prefix+'aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal

        # --------- PROPOSAL GENERATION ---------
        net = F.relu(self.bn1(self.conv1(features))) 
        net = F.relu(self.bn2(self.conv2(net))) 
        net = self.conv3(net) # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)
        ###############################OV - part-start###############################
        cls_clip_input = net[:, -self.pre_OV_head_dim:, :]
        cls_clip_output = self.clip_header(cls_clip_input)
        pc_query_feat = cls_clip_output.permute(2, 0, 1)
        pc_query_feat = pc_query_feat / pc_query_feat.norm(dim=2, keepdim=True)
        
        ##
        if self.training:
            #print("Using training Clip text")
            text_output = self.text_feats / self.text_feats.norm(dim=1, keepdim=True)
            cls_logits , _ = self.classify_pc(pc_query_feat, text_output, self.text_num)
        else:
          
            eval_text_output = self.eval_text_feats / self.eval_text_feats.norm(dim=1, keepdim=True)
            cls_logits , _ = self.classify_pc(pc_query_feat, eval_text_output, self.eval_text_num)

            self.num_class = DATASET_CONFIG.num_eval_class

        cls_concat = cls_logits.permute(0, 2, 1)
        net = torch.cat((net[:, :-self.pre_OV_head_dim, :], cls_concat), dim=1)
            
        ###############################OV - part-end###############################

        end_points = decode_scores(net, end_points, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr, self.key_prefix)
        return end_points

if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    net = ProposalModule(DC.num_class, DC.num_heading_bin,
        DC.num_size_cluster, DC.mean_size_arr,
        128, 'seed_fps').cuda()
    end_points = {'seed_xyz': torch.rand(8,1024,3).cuda()}
    out = net(torch.rand(8,1024,3).cuda(), torch.rand(8,256,1024).cuda(), end_points)
    for key in out:
        print(key, out[key].shape)
