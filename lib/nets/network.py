# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy
import sys
from PIL import Image, ImageDraw
import os

import utils.timer

from layer_utils.snippets import generate_anchors_pre
from layer_utils.generate_3d_anchors import GridAnchor3dGenerator
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer, anchor_target_layer_torch
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes
from model.bbox_transform import bbox_transform_inv, lidar_bbox_transform_inv, lidar_3d_bbox_transform_inv, clip_boxes

from torchvision.ops import RoIAlign, RoIPool
from torchvision import transforms
from model.config import cfg
import utils.bbox as bbox_utils
import tensorboardX as tb

from scipy.misc import imresize

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._predictions      = {}
        self._mc_run_output    = {}
        self._mc_run_results   = {}
        self._losses           = {}
        self._cum_losses       = {}
        self._anchor_targets   = {}
        self._proposal_targets = {}
        self._layers  = {}
        self.timers   = {}
        self._gt_image = None
        self._act_summaries       = {}
        self._score_summaries     = {}
        self._val_event_summaries = {}
        self._event_summaries     = {}
        self._gt_summaries        = {}
        self._cnt              = 0
        self._anchor_cnt       = 0
        self._proposal_cnt     = 0
        self._device           = 'cuda'
        self._cum_loss_keys    = ['total_loss','rpn_cross_entropy','rpn_loss_box']
        #self._cum_loss_keys    = ['total_loss','rpn_cross_entropy','rpn_loss_box','cross_entropy','loss_box']
        if(cfg.ENABLE_FULL_NET):
            self._cum_loss_keys.append('loss_box')
            self._cum_loss_keys.append('cross_entropy')
            if(cfg.NET_TYPE == 'lidar'):
                self._cum_loss_keys.append('ry_loss')
            if(cfg.ENABLE_ALEATORIC_CLS_VAR):
                self._cum_loss_keys.append('a_cls_var')
                self._cum_loss_keys.append('a_cls_entropy')
            if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
                self._cum_loss_keys.append('a_bbox_var')
        self._cum_gt_entries                  = 0
        self._batch_gt_entries                = 0
        self._cum_im_entries                  = 0
        self._num_mc_run                      = 1
        self._num_aleatoric_samples           = cfg.NUM_ALEATORIC_SAMPLE
        self._bev_extents                     = [cfg.LIDAR.X_RANGE,cfg.LIDAR.Y_RANGE,cfg.LIDAR.Z_RANGE]
        self._net_type                        = cfg.NET_TYPE
        #Set on every forward pass for use with proposal target layer
        self._gt_boxes      = None
        self._true_gt_boxes = None
        self._gt_boxes_dc   = None

    def _add_gt_image(self):
        # add back mean
        image = ((self._gt_summaries['frame']))*cfg.PIXEL_STDDEVS + cfg.PIXEL_MEANS
        #Flip info from (xmin,xmax,ymin,ymax) to (ymin,ymax,xmin,xmax) due to frame being rotated
        frame_range = [self._info[3] - self._info[2] + 1, self._info[1] - self._info[0] + 1]
        image = imresize(image[0], frame_range / self._info[6])
        # BGR to RGB (opencv uses BGR)
        #print(image)
        #image = image[:,:,:,cfg.PIXEL_ARRANGE]
        self._gt_image = image[np.newaxis, :, :, cfg.PIXEL_ARRANGE_BGR].copy(order='C')
        #print(self._gt_image.shape)
        #self._gt_image = image[np.newaxis, :, :, :].copy(order='C')

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        self._add_gt_image()
        image = draw_bounding_boxes(\
                          self._gt_image, self._gt_summaries['gt_boxes'], self._gt_summaries['info'][6])

        return tb.summary.image('GROUND_TRUTH',
                                image[0].astype('float32') / 255.0, dataformats='HWC')

    def _add_act_summary(self, key, tensor):
        return tb.summary.histogram(
            'ACT/' + key + '/activations',
            tensor.data.cpu().numpy(),
            bins='auto'),
        tb.summary.scalar('ACT/' + key + '/zero_fraction',
                          (tensor.data == 0).float().sum() / tensor.numel())

    def _add_score_summary(self, key, tensor):
        return tb.summary.histogram(
            'SCORE/' + key + '/scores', tensor.data.cpu().numpy(), bins='auto')

    def _add_train_summary(self, key, var):
        return tb.summary.histogram(
            'TRAIN/' + key, var.data.cpu().numpy(), bins='auto')

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred):
        rois, rpn_scores, anchors = proposal_top_layer(\
                                        rpn_cls_prob, rpn_bbox_pred, self._info,
                                         self._feat_stride, self._anchors, self._num_anchors)
        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
        rois, rpn_scores, anchors_3d = proposal_layer(\
                                        rpn_cls_prob, rpn_bbox_pred, self._info, self._mode,
                                         self._feat_stride, self._anchors, self._anchors_3d, self._num_anchors)
        return rois, rpn_scores, anchors_3d

    def _anchor_target_layer(self, rpn_cls_score):
        #Remove rotation element if LiDAR
        # map of shape (..., H, W)
        height, width = rpn_cls_score.data.shape[1:3]

        #gt_boxes = self._gt_boxes.data.cpu().numpy()
        #gt_boxes_dc = self._gt_boxes_dc.data.cpu().numpy()

        #.data is used to pull a tensor from a pytorch variable. Deprecated, but it grabs a copy of the data that will not be tracked by gradients
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_torch(self._gt_boxes, self._gt_boxes_dc, self._info, self._feat_stride, self._anchors, self._num_anchors, height, width, self._device)
            # bbox_outside_weights

        #rpn_labels = torch.from_numpy(rpn_labels).float().to(
        #    self._device)  #.set_shape([1, 1, None, None])
        #rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets).float().to(
        #    self._device)  #.set_shape([1, None, None, self._num_anchors * 4])
        #rpn_bbox_inside_weights = torch.from_numpy(
        #    rpn_bbox_inside_weights).float().to(
        #        self._device)  #.set_shape([1, None, None, self._num_anchors * 4])
        #rpn_bbox_outside_weights = torch.from_numpy(
        #    rpn_bbox_outside_weights).float().to(
        #        self._device)  #.set_shape([1, None, None, self._num_anchors * 4])

        rpn_labels = rpn_labels.long()
        self._anchor_targets['rpn_labels']               = rpn_labels
        self._anchor_targets['rpn_bbox_targets']         = rpn_bbox_targets
        self._anchor_targets['rpn_bbox_inside_weights']  = rpn_bbox_inside_weights
        self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        for k in self._anchor_targets.keys():
            self._score_summaries[k] = self._anchor_targets[k]

    def _proposal_target_layer(self, rois, roi_scores, anchors_3d):
        labels, rois, anchors_3d, roi_scores, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer(rois, roi_scores, anchors_3d, self._gt_boxes, self._true_gt_boxes, self._gt_boxes_dc, self._num_classes)

        self._proposal_targets['rois']                 = rois
        self._proposal_targets['labels']               = labels.long()
        self._proposal_targets['bbox_targets']         = bbox_targets
        self._proposal_targets['bbox_inside_weights']  = bbox_inside_weights
        self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

        for k in self._proposal_targets.keys():
            self._score_summaries[k] = self._proposal_targets[k]

        return rois, roi_scores, anchors_3d

    def _anchor_component(self, height, width):
        # just to get the shape right
        #height = int(math.ceil(self._info.data[0, 0] / self._feat_stride[0]))
        #width = int(math.ceil(self._info.data[0, 1] / self._feat_stride[0]))
        if(self._net_type == 'image'):
            anchors, anchor_length = generate_anchors_pre(\
                                                height, width,
                                                self._feat_stride, self._anchor_scales, self._anchor_ratios)
            #TODO: Unused, shouldn't use unless lidar. fix please.
            self._anchors_3d = torch.from_numpy(anchors).to(self._device)
        elif(self._net_type == 'lidar'):
            anchor_generator = GridAnchor3dGenerator()
            anchor_length, anchors = anchor_generator._generate(height, width, self._feat_stride, self._anchor_scales, self._anchor_ratios)
            self._anchors_3d = torch.from_numpy(anchors).to(self._device)
            anchors = bbox_utils.bbaa_graphics_gems(anchors, (width)*self._feat_stride-1, (height)*self._feat_stride-1)
        self._anchor_targets['anchors'] = torch.from_numpy(anchors).to(self._device)
        self._anchors = torch.from_numpy(anchors).to(self._device)
        self._anchor_length = anchor_length

    def _huber_loss(self,pred, targets, huber_delta, sigma):
        sigma_2 = sigma**2
        box_diff = pred - targets
        abs_in_box_diff = torch.abs(box_diff)
        smoothL1_sign = (abs_in_box_diff < huber_delta / sigma_2).detach().float()
        above_one = (abs_in_box_diff - (0.5 * huber_delta / sigma_2)) * (1. - smoothL1_sign)
        below_one = torch.pow(box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign
        in_loss_box = below_one + above_one
        return in_loss_box

    def _smooth_l1_loss(self,
                        stage,
                        bbox_pred,
                        bbox_targets,
                        bbox_var,
                        bbox_inside_weights,
                        bbox_outside_weights,
                        sigma=1.0,
                        dim=[1]):
        if((stage == 'RPN' and cfg.ENABLE_RPN_BBOX_VAR) or (stage == 'DET' and cfg.ENABLE_ALEATORIC_BBOX_VAR)):
            bbox_var_en = True
        else:
            bbox_var_en = False
        #Ignore diff when target is not a foreground target
        # a mask array for the foreground anchors (called “bbox_inside_weights”) is used to calculate the loss as a vector operation and avoid for-if loops.
        bbox_pred = bbox_pred*bbox_inside_weights
        bbox_targets = bbox_targets*bbox_inside_weights
        #torch.set_printoptions(profile="full")
        #print('from _smooth_l1_loss')
        #print(bbox_targets)
        #print(bbox_inside_weights)
        #torch.set_printoptions(profile="default")
        if(self._net_type == 'lidar' and stage == 'DET'):
            bbox_shape   = [bbox_pred.shape[0],bbox_pred.shape[1]]
            elem_rm      = int(bbox_shape[1]/7)
            bbox_pred_aa = bbox_pred.reshape(-1,7)[:,0:6].reshape(-1,bbox_shape[1]-elem_rm)
            targets_aa   = bbox_targets.reshape(-1,7)[:,0:6].reshape(-1,bbox_shape[1]-elem_rm)
            #TODO: Sum across elements
            loss_box     = self._huber_loss(bbox_pred_aa,targets_aa,1.0,sigma)
            #TODO: Do i need to compute the sin of the difference here?
            sin_pred     = bbox_pred.reshape(-1,7)[:,6:7].reshape(-1,elem_rm)
            #Convert to sin to normalize, targets will be in degrees off of anchor
            sin_targets  = bbox_targets.reshape(-1,7)[:,6:7].reshape(-1,elem_rm)
            ry_loss      = self._huber_loss(sin_pred,sin_targets,1.0/9.0,sigma)
            self._losses['ry_loss'] = torch.mean(torch.sum(ry_loss,dim=1))
            in_loss_box  = torch.cat((loss_box.reshape(-1,6),ry_loss.reshape(-1,1)),dim=1).reshape(-1,bbox_shape[1])
            #bbox_outside_weights = torch.mean(bbox_outside_weights,axis=1)
        else:
            in_loss_box = self._huber_loss(bbox_pred,bbox_targets,1.0,sigma)

        if(bbox_var_en):
            #Don't need covariance matrix as it collapses itself in the end anyway
            in_loss_box = 0.5*in_loss_box*torch.exp(-bbox_var) + 0.5*torch.exp(bbox_var)
            in_loss_box = in_loss_box*bbox_inside_weights
        #Used to normalize the predictions, this is only used in the RPN
        #By default negative(background) and positive(foreground) samples have equal weighting
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = out_loss_box
        #Condense down to 1D array, each entry is the box_loss for an individual box, array is batch size of all predicted boxes
        #[loss,y,x,num_anchor]
        for i in sorted(dim, reverse=True):
            loss_box = loss_box.sum(i)
        #print(loss_box.size())
        #TODO: Could it be mean is taken at a different level between rpn and 2nd stage??
        loss_box = loss_box.mean()
        return loss_box

    #Determine losses for single batch image
    def _add_losses(self, sigma_rpn=3.0):
        # RPN, class loss
        #View rearranges the matrix to match specified dimension -1 is inferred from other dims, probably OBJ/Not OBJ
        rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)
        #What is the target label out of the RPN
        rpn_label     = self._anchor_targets['rpn_labels'].view(-1)
        #Remove all non zeros to get an index list of target objects, not dontcares
        #.nonzero() returns indices
        rpn_select    = (rpn_label.data != -1).nonzero().view(-1)
        #TODO: RPN class score is indexed by the GT entries. That doesnt seem right. Need to investigate
        #Upon further investigation it appears based upon the entries in the generated anchors. If anchors and targets have an associated size then 
        #That would make this make sense.
        rpn_cls_score = rpn_cls_score.index_select(
            0, rpn_select).contiguous().view(-1, 2)
        #Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
        rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)
        #torch.nn.functional
        #Compare labels from anchor_target_layer and rpn_layer
        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, reduction='mean')

        # RPN, bbox loss

        #Pretty sure these are delta's at this point
        rpn_bbox_pred = self._predictions['rpn_bbox_pred']
        rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = self._anchor_targets[
            'rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = self._anchor_targets[
            'rpn_bbox_outside_weights']
        rpn_loss_box = self._smooth_l1_loss(
            'RPN',
            rpn_bbox_pred,
            rpn_bbox_targets,
            [],
            rpn_bbox_inside_weights,
            rpn_bbox_outside_weights,
            sigma=sigma_rpn,
            dim=[1, 2, 3])
        if(cfg.ENABLE_FULL_NET):
            # RCNN, class loss, performed on class score logits
            cls_score            = self._predictions['cls_score']
            cls_prob             = self._predictions['cls_prob']
            label                = self._proposal_targets['labels'].view(-1)
            # RCNN, bbox loss
            bbox_pred            = self._predictions['bbox_pred']
            #This should read bbox_target_deltas
            bbox_targets         = self._proposal_targets['bbox_targets']
            bbox_inside_weights  = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            #Flag to only handle a_bbox_var if enabled
            #Compute aleatoric class entropy
            if(cfg.ENABLE_ALEATORIC_CLS_VAR):
                a_cls_var  = self._predictions['a_cls_var']
                cross_entropy, a_cls_mutual_info = self._bayesian_cross_entropy(cls_score, a_cls_var, label,cfg.NUM_CE_SAMPLE)
                self._losses['a_cls_entropy'] = torch.mean(a_cls_mutual_info)
                self._losses['a_cls_var']     = torch.mean(a_cls_var)
                #Classical entropy w/out logit sampling
                #self._losses['a_cls_entropy'] = torch.mean(self._categorical_entropy(cls_prob))
            else:
                cross_entropy = F.cross_entropy(cls_score, label)
                self._losses['a_cls_entropy'] = torch.tensor(0)
            #Compute aleatoric bbox variance
            if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
                #Grab a local variable for a_bbox_var
                a_bbox_var  = self._predictions['a_bbox_var']
                #network output comes out as log(a_bbox_var), so it needs to be adjusted
                #TODO: Should i only care about top output variance? Thats what bbox_inside_weights does
                self._losses['a_bbox_var'] = (torch.exp(a_bbox_var)*bbox_inside_weights).mean()
            else:
                a_bbox_var = None
                self._losses['a_bbox_var'] = torch.tensor(0)
            #Compute loss box
            loss_box = self._smooth_l1_loss('DET', bbox_pred, bbox_targets, a_bbox_var, bbox_inside_weights, bbox_outside_weights)
            #Assign computed losses to be tracked in tensorboard
            self._losses['cross_entropy']     = cross_entropy
            self._losses['loss_box']          = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box']      = rpn_loss_box
            #Total loss
            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        else:
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box']      = rpn_loss_box
            #Total loss
            loss = rpn_cross_entropy + rpn_loss_box
        self._losses['total_loss'] = loss
        return loss

    def _compute_bbox_cov(self,bbox_samples):
        mc_bbox_mean = torch.mean(bbox_samples,dim=0)
        mc_bbox_pred = bbox_samples.unsqueeze(3)
        mc_bbox_var = torch.mean(torch.matmul(mc_bbox_pred,mc_bbox_pred.transpose(2,3)),dim=0)
        mc_bbox_mean = mc_bbox_mean.unsqueeze(2)
        mc_bbox_var = mc_bbox_var - torch.matmul(mc_bbox_mean,mc_bbox_mean.transpose(1,2))
        mc_bbox_var = mc_bbox_var*torch.eye(mc_bbox_var.shape[-1]).cuda()
        mc_bbox_var = torch.sum(mc_bbox_var,dim=-1)
        #mc_bbox_var = torch.diag_embed(mc_bbox_var,offset=0,dim1=1,dim2=2)
        return mc_bbox_var.clamp_min(0.0)

    def _compute_bbox_var(self,bbox_samples):
        n = bbox_samples.shape[0]
        mc_bbox_mean = torch.pow(torch.sum(bbox_samples,dim=0),2)
        mc_bbox_var = torch.sum(torch.pow(bbox_samples,2),dim=0)
        mc_bbox_var += -mc_bbox_mean/n
        mc_bbox_var = mc_bbox_var/(n-1)
        return mc_bbox_var.clamp_min(0.0)

    def _categorical_entropy(self,cls_prob):
        #Compute entropy for each class(y=c)
        cls_entropy = cls_prob*torch.log(cls_prob)
        #Sum across classes
        total_entropy = -torch.sum(cls_entropy,dim=1)
        #true_cls      = torch.gather(cls_score,1,labels.unsqueeze(1)).squeeze(1)
        #softmax = torch.exp(true_cls)/torch.mean(torch.exp(cls_score),dim=1)
        return total_entropy
    #input: cls_score (T,N,C) T-> Samples, N->Batch, C-> Classes
    def _categorical_mutual_information(self,cls_score):
        cls_prob = F.softmax(cls_score,dim=2)
        avg_cls_prob = torch.mean(cls_prob,dim=0)
        total_entropy = self._categorical_entropy(avg_cls_prob)
        #Take sum of entropy across classes
        mutual_info = torch.sum(cls_prob*torch.log(cls_prob),dim=2)
        #Get expectation over T forward passes
        mutual_info = torch.mean(mutual_info,dim=0)
        mutual_info += total_entropy
        return mutual_info.clamp_min(0.0)

    def _bayesian_cross_entropy(self,cls_score,cls_var,targets,num_sample):
        


        #cls_var comes in as true variance. Network output is log(var) but exp is applied at network output.
        #true_var       = torch.gather(cls_var,1,targets.unsqueeze(1)).squeeze(1)
        #undistorted_ce = F.cross_entropy(cls_score, targets,reduction='none')

        #Distorted loss - generate a normal distribution.
        #Change where it is sampled from? Maybe from output of CE?
        #preprocess

        #cls_score_mask = torch.zeros((cls_score.shape[0],cls_score.shape[1]),device='cuda').scatter(1, targets, 1)
        #for i,mask_row in enumerate(cls_score_mask):
        #    mask_row[targets[i]] = 1
        cls_score_mask = torch.zeros_like(cls_score).scatter_(1, targets.unsqueeze(1), 1)
        cls_score_shifted = cls_score + cls_score_mask
        #for i in range(cls_score.shape[0]):
        #    for j in range(cls_score.shape[1]):
        #        if(targets[i] == j):
        #            cls_score_shifted[i,j] = cls_score[i,j] + 1
        #        else:
        #            cls_score_shifted[i,j] = cls_score[i,j]

        #cls_score[:,targets] = cls_score[:,targets] + 1
        #cls_score_shifted = torch.zeros((cls_score.shape[0],cls_score.shape[1],device='cuda').scatter_()
        #cls_score_shifted = cls_score[:,targets] + 1
        #Step 1: Get a set of distorted logits sampled from a gaussian distribution
        distribution     = torch.distributions.Normal(0,torch.sqrt(cls_var))
        cls_score_resize = cls_score_shifted.repeat(num_sample,1,1)
        logit_samples    = distribution.sample((num_sample,)) + cls_score_resize
        #logit_samples    = logit_samples.permute(1,2,0)
        #Step 2: Pass logit samples through cross entropy loss
        #targets_resize  = targets.repeat(num_sample,1)
        
        #ce_loss_samples = F.cross_entropy(logit_samples, targets_resize,reduction='none')
        #Step 3: Undo NLL
        softmax_samples  = F.softmax(logit_samples,dim=2)
        #softmax_samples = torch.exp(-ce_loss_samples)
        #Step 4: Take average of T samples
        avg_softmax     = torch.mean(softmax_samples,dim=0)
        #Step 5: Redo NLL
        ce_loss         = F.nll_loss(torch.log(avg_softmax),targets)
        #Step 6: Add regularizer
        ce_loss         += 0.01*torch.mean(cls_var)
        #ce_loss         = -torch.log(avg_softmax)
        a_mutual_info   = self._categorical_mutual_information(logit_samples)
        #targets_resize = targets.repeat(num_sample,1)
        #distorted_ce = -F.cross_entropy(logits, targets_resize,reduction='none')
        #ce_noise     = torch.log(distorted_ce) - undistorted_ce.repeat(num_sample,1)
        #ce_noise_var = torch.var(ce_noise)
        #ce_noise_var = torch.pow(-torch.log(torch.mean(torch.exp(ce_noise),dim=0)),2)
        #ce_elu       = -1*torch.nn.functional.elu(-ce_noise)
        #ce_var_loss  = ce_elu + undistorted_ce.repeat(num_sample,1)
        #ce_var_loss  = torch.log(torch.mean(torch.exp(ce_var_loss),dim=0)+1e-9)
        #regularizer   = torch.mean(ce_noise_mean)
        #regularizer   = regularizer - 1
        #regularizer   = torch.mean(torch.pow(ce_noise,2))
        #distorted_ce = -torch.nn.functional.elu(-distorted_ce)
        #ce_loss = -torch.log(torch.mean(torch.exp(-distorted_ce),dim=0))
        #ce_loss = torch.mean(ce_loss)
        #print('ce loss {}'.format(ce_loss))
        #print('ce_var_loss {}'.format())
        #print('reg {}'.format(regularizer))
        #print('cls_var {}'.format(torch.mean(cls_var)))
        return ce_loss, a_mutual_info

    def _region_proposal(self, net_conv):
        #this links net conv -> rpn_net -> relu
        rpn = F.relu(self.rpn_net(net_conv))
        #print('RPN result')
        #print(rpn)
        self._act_summaries['rpn'] = rpn
        dropout_layer = nn.Dropout(0.1)
        rpn_d = dropout_layer(rpn)
        rpn_cls_score = self.rpn_cls_score_net(
            rpn)  # batch * (num_anchors * 2) * h * w
        #print(rpn_cls_score.size())
        # change it so that the score has 2 as its channel size for softmax
        rpn_cls_score_reshape = rpn_cls_score.view(
            1, 2, -1,
            rpn_cls_score.size()[-1])  # batch * 2 * (num_anchors*h) * w
        #print(rpn_cls_score_reshape[0,:,1,1])
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        #print(rpn_cls_prob_reshape[0,:,1,1])
        # Move channel to the last dimenstion, to fit the input of python functions
        rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(
            0, 2, 3, 1)  # batch * h * w * (num_anchors * 2)
        rpn_cls_score = rpn_cls_score.permute(
            0, 2, 3, 1)  # batch * h * w * (num_anchors * 2)
        rpn_cls_score_reshape = rpn_cls_score_reshape.permute(
            0, 2, 3, 1).contiguous()  # batch * (num_anchors*h) * w * 2
        rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), dim=1)[1]

        rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
        rpn_bbox_pred = rpn_bbox_pred.permute(
            0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

        if self._mode == 'TRAIN':
            #At this point, rpn_bbox_pred is a normalized delta
            #self.timers['proposal'].tic()
            rois, roi_scores, anchors_3d = self._proposal_layer(
                rpn_cls_prob, rpn_bbox_pred)  # rois, roi_scores are varible
            #self.timers['proposal'].toc()
            #targets for first stage loss computation (against the RPN predictions)
            #self.timers['anchor_t'].tic()
            self._anchor_target_layer(rpn_cls_score)
            #self.timers['anchor_t'].toc()
            #N.B. - ROI's passed into proposal_target_layer have been pre-transformed and are true bounding boxes
            #Generate final detection targets from ROI's generated from the RPN
            #self.timers['proposal_t'].tic()
            if(cfg.ENABLE_FULL_NET):
                rois, _, anchors_3d = self._proposal_target_layer(rois, roi_scores, anchors_3d)
            #self.timers['proposal_t'].toc()
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _, anchors_3d = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
            elif cfg.TEST.MODE == 'top':
                rois, _, anchors_3d = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
            else:
                raise NotImplementedError

        self._predictions['rpn_cls_score'] = rpn_cls_score
        self._predictions['rpn_cls_score_reshape'] = rpn_cls_score_reshape
        self._predictions['rpn_cls_prob'] = rpn_cls_prob
        self._predictions['rpn_cls_pred'] = rpn_cls_pred
        self._predictions['rpn_bbox_pred'] = rpn_bbox_pred
        self._predictions['rois'] = rois
        self._predictions['roi_scores'] = roi_scores
        self._predictions['anchors_3d'] = anchors_3d

        return rois

    #Used to dynamically change batch size depending on eval or train
    def set_num_mc_run(self,num_mc_run):
        self._num_mc_run = num_mc_run

    def _region_classification(self, fc7):
        raise NotImplementedError

    def _input_to_head(self):
        raise NotImplementedError

    def _head_to_tail(self, pool5):
        raise NotImplementedError

    def create_architecture(self,
                            num_classes,
                            tag=None,
                            anchor_scales=(8, 16, 32),
                            anchor_ratios=(0.5, 1, 2)):
        self._tag = tag

        self._num_classes = num_classes
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self._num_anchors = self._num_scales * self._num_ratios

        assert tag != None

        # Initialize layers
        self._init_modules()

    def _run_summary_op(self, val=False, summary_size=1):
        """
            Run the summary operator: feed the placeholders with corresponding network outputs(activations)
        """
        summaries = []
        # Add image gt
        if(self._net_type == 'image'):
            summaries.append(self._add_gt_image_summary())
        #elif(self._net_type == 'lidar'):
        #    print('vg summaries on tensorboard not supported yet.')
        # Add event_summaries
        if not val:
            for key, var in self._event_summaries.items():
                #print("adding summary for key {:s} with value {:f} and summary size divisor {:d}".format(key,var,summary_size))
                summaries.append(tb.summary.scalar(key, var/float(summary_size)))
            #Reset summary val
            self._event_summaries = {}
            # Add score summaries
            for key, var in self._score_summaries.items():
                summaries.append(self._add_score_summary(key, var))
            self._score_summaries = {}
            # Add act summaries
            for key, var in self._act_summaries.items():
                summaries += self._add_act_summary(key, var)
            self._act_summaries = {}
            # Add train summaries
            for k, var in dict(self.named_parameters()).items():
                if var.requires_grad:
                    summaries.append(self._add_train_summary(k, var))

            self._gt_summaries = {}
        else:
            for key, var in self._val_event_summaries.items():
                #print("adding validation summary for key {:s} with value {:f} and summary size divisor {:d}".format(key,var,summary_size))
                summaries.append(tb.summary.scalar(key, var/float(summary_size)))
            #Reset summary val
            self._val_event_summaries = {}

        return summaries

    def _draw_and_save(self,im,gt_boxes):
        datapath = os.path.join(cfg.DATA_DIR, 'waymo','debug')
        out_file = os.path.join(datapath,'{}.png'.format(self._cnt))
        im = (im + cfg.PIXEL_MEANS)*cfg.PIXEL_STDDEVS
        im = im.astype('uint8')[0,:,:,:]
        print(im.shape)
        source_img = Image.fromarray(im)
        draw = ImageDraw.Draw(source_img)
        for det in gt_boxes:
            draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(0,0,0))
        print('Saving file at location {}'.format(out_file))
        source_img.save(out_file,'PNG')  
        self._cnt += 1

    def _predict(self):
        # This is just _build_network in tf-faster-rcnn
        #self.timers['net'].tic()
        torch.backends.cudnn.benchmark = False
        net_conv = self._input_to_head(self._frame)
        #print(net_conv)
        # build the anchors for the image
        #self.timers['anchor_gen'].tic()
        self._anchor_component(net_conv.size(2), net_conv.size(3))
        #self.timers['anchor_gen'].toc()
        #print('run region proposal network')
        #numpy_out = net_conv.cpu().detach().numpy()[0, :, :, :]
        #print(numpy_out.shape)
        #for i in range(0,1000):
        #    numpy.savetxt('/home/mat/Thesis/train_net_conv_out_feature_{:d}_.txt'.format(i), numpy_out[i,:,:], delimiter=',')
        rois = self._region_proposal(net_conv)
        #print('_predict ROIs')
        #print(rois)
        if(cfg.ENABLE_FULL_NET):
            if cfg.POOLING_MODE == 'align':
                pool5 = self._roi_align_layer(net_conv, rois)
            else:
                pool5 = self._roi_pool_layer(net_conv, rois)
            #del net_conv
            if self._mode == 'TRAIN':
                #Find best algo
                #self._num_mc_run = 1
                torch.backends.cudnn.benchmark = True  # benchmark because now the input size are fixed
            #elif(self._mode == 'TEST'):
            #    self._num_mc_run = 10
            if(cfg.ENABLE_EPISTEMIC_BBOX_VAR or cfg.ENABLE_EPISTEMIC_CLS_VAR):
                dropout_en = True
            else:
                dropout_en = False
            if(cfg.ENABLE_CUSTOM_TAIL):
                fc7 = self._custom_tail(pool5,dropout_en)
            else:
                fc7 = self._head_to_tail(pool5,dropout_en)
                fc7 = fc7.unsqueeze(0).repeat(self._num_mc_run,1,1)

            self._region_classification(fc7)
            #self.timers['net'].toc()
        for k in self._predictions.keys():
            self._score_summaries[k] = self._predictions[k]

    def forward(self, frame, info=None, gt_boxes=None, gt_boxes_dc=None, mode='TRAIN'):
        self._gt_summaries['frame'] = frame
        self._gt_summaries['gt_boxes'] = gt_boxes
        self._gt_summaries['gt_boxes_dc'] = gt_boxes_dc
        self._gt_summaries['info'] = info
        self._info = info  # No need to change; actually it can be an list
        scale = info[6]
        self._frame = torch.from_numpy(frame.transpose([0, 3, 1,
                                                        2])).to(self._device)
        if(self._net_type == 'image'):
            true_gt_boxes = gt_boxes

        elif(self._net_type == 'lidar'):
            #TODO: Should info contain bev extants? Seems like the cleanest way
            gt_box_labels = gt_boxes[:, -1, np.newaxis]
            gt_bboxes     = gt_boxes[:, :-1]
            gt_boxes      = bbox_utils.bbaa_graphics_gems(gt_bboxes,info[1],info[3])
            #gt_boxes      = bbox_utils.bbox_bev_to_voxel_grid(gt_boxes,self._bev_extants,info)
            gt_boxes      = np.concatenate((gt_boxes, gt_box_labels),axis=1)
            #Still in 3D format
            #bev_extents   = [cfg.LIDAR.X_RANGE[0],cfg.LIDAR.Y_RANGE[0],cfg.LIDAR.Z_RANGE[0],cfg.LIDAR.X_RANGE[1],cfg.LIDAR.Y_RANGE[1],cfg.LIDAR.Z_RANGE[1]]
            #bev_gt_bboxes = bbox_utils.bbox_voxel_grid_to_pc(gt_bboxes,bev_extents,info)
            true_gt_boxes = np.concatenate((gt_bboxes, gt_box_labels),axis=1)
            #Dont care areas
            #gt_boxes_dc   = bbox_utils.bbox_3d_to_bev_axis_aligned(gt_boxes_dc)
            gt_boxes_dc = None
            #gt_boxes_dc   = bbox_utils.bbox_bev_to_voxel_grid(gt_boxes,self._bev_extants,info)

        self._true_gt_boxes = torch.from_numpy(true_gt_boxes).to(
            self._device) if true_gt_boxes is not None else None
        self._gt_boxes = torch.from_numpy(gt_boxes).to(
            self._device) if gt_boxes is not None else None
        self._gt_boxes_dc = torch.from_numpy(np.empty(0)).to(self._device)
        #torch.from_numpy(gt_boxes_dc).to(
        #    self._device) if gt_boxes is not None else None
        self._mode = mode
        #This overrides the mode so configuration parameters used in train can also be used for val
        if(mode == 'VAL'):
            self._mode = 'TRAIN'

        self._predict()
        #ENABLE to draw all anchors
        if(cfg.DEBUG.DRAW_ANCHORS):
            self._draw_and_save_anchors(frame,
                                        self._anchor_targets['anchors'],
                                        self._net_type)

        #ENABLE to draw all anchor targets
        if(cfg.DEBUG.DRAW_ANCHOR_T):
            self._draw_and_save_targets(frame,
                                        self._anchor_targets['rpn_bbox_targets'],
                                        self._anchor_targets['anchors'],
                                        None,
                                        self._anchor_targets['rpn_labels'],
                                        self._anchor_targets['rpn_bbox_inside_weights'],
                                        'anchor',
                                        self._net_type)

        #ENABLE to draw all proposal targets
        if(cfg.DEBUG.DRAW_PROPOSAL_T):
            self._draw_and_save_targets(frame,
                                        self._proposal_targets['bbox_targets'],
                                        self._proposal_targets['rois'],
                                        self._predictions['anchors_3d'],
                                        self._proposal_targets['labels'],
                                        self._proposal_targets['bbox_inside_weights'],
                                        'proposal',
                                        self._net_type)

        if(mode == 'VAL' or mode == 'TRAIN'):
            #self.timers['losses'].tic()
            self._add_losses()  # compute losses
            #self.timers['losses'].toc()
        if(mode == 'VAL' or mode == 'TEST'):
            if(cfg.ENABLE_FULL_NET):
                bbox_pred = self._predictions['bbox_pred']
                rois      = self._predictions['rois'][:,1:]
                #bbox_targets are pre-normalized for loss, so modifying here.
                #Expand as -> broadcast 
                stds = bbox_pred.data.new(cfg.TRAIN[self._net_type.upper()].BBOX_NORMALIZE_STDS).repeat(
                    self._num_classes).unsqueeze(0).expand_as(bbox_pred)
                means = bbox_pred.data.new(cfg.TRAIN[self._net_type.upper()].BBOX_NORMALIZE_MEANS).repeat(
                    self._num_classes).unsqueeze(0).expand_as(bbox_pred)
                #Denormalize bbox target predictions
                bbox_mean = bbox_pred.mul(stds).add(means)
                #bbox_var = self._predictions['a_bbox_var']
                if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
                    bbox_gaussian = torch.distributions.Normal(0,torch.sqrt(torch.exp(self._predictions['a_bbox_var'])))
                    bbox_samples = bbox_gaussian.sample((self._num_aleatoric_samples,)) + bbox_mean
                    mean_bbox_inv = bbox_transform_inv(rois,bbox_mean,scale)
                    #TODO: Maybe detach here?
                    roi_coords = rois.unsqueeze(0).repeat(self._num_aleatoric_samples,1,1)
                    roi_coords = roi_coords.view(-1,roi_coords.shape[2])
                    bbox_samples = bbox_samples.view(-1,bbox_samples.shape[2])
                    bbox_inv_samples = bbox_transform_inv(roi_coords,bbox_samples,scale)
                    bbox_inv_samples = bbox_inv_samples.view(self._num_aleatoric_samples,-1,bbox_inv_samples.shape[1])
                    bbox_inv_var = self._compute_bbox_var(bbox_inv_samples)
                    self._predictions['bbox_inv_pred']  = mean_bbox_inv
                    self._predictions['a_bbox_inv_var'] = bbox_inv_var
                else:
                    if(self._net_type == 'image'):
                        mean_bbox_inv = bbox_transform_inv(rois,bbox_mean,scale)
                    elif(self._net_type == 'lidar'):
                        #roi_height = cfg.LIDAR.ANCHORS[0][2]
                        #roi_zc     = cfg.LIDAR.ANCHORS[0][2]/2
                        #mean_bbox_inv = lidar_bbox_transform_inv(rois,roi_height,roi_zc,bbox_mean,scale)
                        mean_bbox_inv = lidar_3d_bbox_transform_inv(self._predictions['anchors_3d'],bbox_mean,scale)
                    
                    self._predictions['bbox_inv_pred'] = mean_bbox_inv

        #Reset to mode == VAL
        self._mode = mode

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, image):
        feat = self._layers["head"](torch.from_numpy(
            image.transpose([0, 3, 1, 2])).to(self._device))
        return feat

    # only useful during testing mode
    def test_frame(self, frame, info):
        self.eval()
        scale = info[6]
        with torch.no_grad():
            self.forward(frame, info, None, None, mode='TEST')
        cls_score, cls_prob, bbox_pred, rois = self._predictions["cls_score"].data.cpu().detach(), \
                                                         self._predictions['cls_prob'].data.detach(), \
                                                         self._predictions['bbox_inv_pred'].data.detach(), \
                                                         self._predictions['rois'].data.detach()

        a_bbox_var, e_bbox_var, a_cls_entropy, a_cls_var, e_cls_mutual_info = self._uncertainty_postprocess(bbox_pred,cls_prob,rois,scale)

        return cls_score, cls_prob, a_cls_entropy, a_cls_var, e_cls_mutual_info, bbox_pred, a_bbox_var, e_bbox_var, rois

    def delete_intermediate_states(self):
        # Delete intermediate result to save memory
        for d in [
                self._losses, self._predictions, self._anchor_targets,
                self._proposal_targets
        ]:
            for k in list(d):
                del d[k]
                
    #Eval summary required
    def run_eval(self, blobs, sum_size, update_summaries=False):
        #Flip between eval and train mode -> gradient doesnt accumulate?
        self.eval() # model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
        with torch.no_grad():
            self.forward(blobs['data'], blobs['info'], blobs['gt_boxes'], blobs['gt_boxes_dc'], mode='VAL')
        self.train()
        if(cfg.ENABLE_FULL_NET):
            summary           = None
            bbox_pred         = self._predictions['bbox_inv_pred'].data.detach() #(self._fc7_channels, self._num_classes * 4)
            cls_prob          = self._predictions['cls_prob'].data.detach() #(self._fc7_channels, self._num_classes)
            rois              = self._predictions['rois'].data.detach()
            roi_labels        = self._proposal_targets['labels'].data.detach()

            uncertainties = self._uncertainty_postprocess(bbox_pred,cls_prob,rois,blobs['info'][4])
        else:
            summary    = None
            bbox_pred  = self._predictions['rois'][:,1:5]
            cls_prob   = self._predictions['roi_scores']
            rois       = self._gt_boxes[:, :4]
            roi_labels = self._gt_boxes[:,4:5]
            uncertainties = None
        for k in self._losses.keys():
            if(k in self._val_event_summaries):
                self._val_event_summaries[k] += self._losses[k].item()
            else:
                self._val_event_summaries[k] = self._losses[k].item()
        if(update_summaries is True):
            summary = self._run_summary_op(True,sum_size)
        self.delete_intermediate_states()
        return summary, rois, roi_labels, cls_prob, bbox_pred, uncertainties

    def _uncertainty_postprocess(self,bbox_pred,cls_prob,rois,im_scale):
        uncertainties = {}
        if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
            uncertainties['a_bbox_var'] = self._predictions['a_bbox_inv_var'].data.detach() #(self._fc7_channels, self._num_classes * 4)
        else:
            uncertainties['a_bbox_var'] = None
        if(cfg.ENABLE_ALEATORIC_BBOX_VAR):
            cls_prob = self._predictions['cls_prob'] #(self._fc7_channels, self._num_classes * 4)
            #TODO: This should not be taking the mean yet, we need to filter by top indices
            #a_cls_entropy = -torch.mean(a_cls_entropy)*torch.log(torch.mean(a_cls_entropy)) - (1-torch.mean(a_cls_entropy))*torch.log(1-torch.mean(a_cls_entropy))
            a_cls_entropy                   = self._categorical_entropy(cls_prob)
            uncertainties['a_cls_entropy']  = a_cls_entropy.data.detach()
            uncertainties['a_cls_var']      = self._predictions['a_cls_var']
            #Compute class entropy
        else:
            uncertainties['a_cls_entropy'] = None
            uncertainties['a_cls_var']     = None
        #if(cfg.ENABLE_ALEATORIC_CLS_VAR):
        #    cls_var = self._predictions['cls_var'].data.detach().cpu().numpy() #(self._fc7_channels, self._num_classes * 4)
        #else:
        #    cls_var = None
        #For tensorboard
        if(cfg.ENABLE_EPISTEMIC_CLS_VAR):
            e_cls_score = self._mc_run_output['cls_score'].detach()
            #Compute average entropy via mutual information
            e_cls_mutual_info = self._categorical_mutual_information(e_cls_score)
            uncertainties['e_cls_mutual_info'] = e_cls_mutual_info
            self._mc_run_results['e_cls_mutual_info'] = torch.mean(e_cls_mutual_info)
            self._mc_run_output['e_cls_mutual_info'] = e_cls_mutual_info
        else:
            uncertainties['e_cls_mutual_info'] = torch.tensor([0])

        if(cfg.ENABLE_EPISTEMIC_BBOX_VAR):
            #All of this to simply get the predictions from [M,N,C] to [M*N,C] interleaved.
            #This is to not change bbox_transform_inv
            mc_bbox_pred = self._mc_run_output['bbox_pred']
            mc_bbox_pred = mc_bbox_pred.view(-1,mc_bbox_pred.shape[2])
            roi_sampled = rois[:,1:]
            roi_sampled = roi_sampled.unsqueeze(0).repeat(self._num_mc_run,1,1)
            roi_sampled = roi_sampled.view(-1,roi_sampled.shape[2])
            mc_bbox_pred = bbox_transform_inv(roi_sampled,mc_bbox_pred,im_scale)
            mc_bbox_pred = mc_bbox_pred.view(self._num_mc_run,-1,mc_bbox_pred.shape[1])
            #Way #1 to compute bbox var
            #mc_bbox_covar = self._compute_bbox_cov(mc_bbox_pred)
            #Way #2 to compute bbox var
            e_bbox_var   = self._compute_bbox_var(mc_bbox_pred)
            #Way #3 to compute bbox var
            #Doesnt work??
            #e_bbox_var = torch.var(mc_bbox_pred,dim=0)
            uncertainties['e_bbox_var'] = e_bbox_var
            self._mc_run_output['e_bbox_var'] = e_bbox_var
            self._mc_run_results['e_bbox_var'] = torch.mean(e_bbox_var)
            #Compute average variance
        else:
            uncertainties['e_bbox_var'] = torch.tensor([0])
        if(cfg.ENABLE_EPISTEMIC_BBOX_VAR or cfg.ENABLE_EPISTEMIC_CLS_VAR):
            for k in self._mc_run_results.keys():
                if(k in self._val_event_summaries):
                    self._val_event_summaries[k] += self._mc_run_results[k].item()
                else:
                    self._val_event_summaries[k] = self._mc_run_results[k].item()
        return uncertainties

    def train_step(self, blobs, train_op, update_weights=False):
        #Computes losses for single image
        self.forward(blobs['data'], blobs['info'], blobs['gt_boxes'], blobs['gt_boxes_dc'])
        #.item() converts single element of type pytorch.tensor to a primitive float/int
        loss = self._losses['total_loss'].item()
        #utils.timer.timer.tic('backward')
        #self.timers['backprop'].tic()
        self._losses['total_loss'].backward()
        #self.timers['backprop'].toc()
        #utils.timer.timer.toc('backward')
        for key in self._cum_loss_keys:
            if(key in self._cum_losses):
                self._cum_losses[key] += self._losses[key].item()
            else:
                self._cum_losses[key] = self._losses[key].item()

        self._batch_gt_entries                += len(blobs['gt_boxes'])
        #Pseudo batching, only one image on the GPU at a time, but weights are updated at intervals

        if(update_weights):
            #Clip gradients
            torch.nn.utils.clip_grad_norm_([x[1] for x in self.named_parameters()],cfg.GRAD_MAX_CLIP)
            train_op.step()
            train_op.zero_grad()
            for k in self._cum_losses.keys():
                if(k in self._event_summaries):
                    self._event_summaries[k] += self._cum_losses[k]
                else:
                    self._event_summaries[k] = self._cum_losses[k]
            self._cum_losses        = {}
            self._batch_gt_entries  = 0
            self._cum_gt_entries   += self._batch_gt_entries
        #Should actually be divided by batch size, but whatever
        self._cum_im_entries       += 1
        self.delete_intermediate_states()

        return loss

    def train_step_with_summary(self, blobs, train_op, sum_size, update_weights=False):
        loss = self.train_step(blobs, train_op, update_weights)
        summary = self._run_summary_op(False, self._cum_im_entries)
        self._cum_gt_entries = 0
        self._cum_im_entries = 0
        return loss, summary

    def train_step_no_return(self, blobs, train_op):
        self.forward(blobs['data'], blobs['info'], blobs['gt_boxes'], blobs['gt_boxes_dc'])
        train_op.zero_grad()
        self._losses['total_loss'].backward()
        train_op.step()
        self.delete_intermediate_states()

    def print_cumulative_loss(self, batch_start_iter, iter, max_iters, lr):
        div = float(self._batch_gt_entries)
        if(div == 0):
            return
        print('iter: %d - %d / %d, total batch loss: %.6f\n >>> rpn_loss_cls: %.6f\n '
                '>>> rpn_loss_box: %.6f\n >>> loss_cls: %.6f\n >>> loss_box: %.6f\n >>> lr: %f' % \
                (batch_start_iter, iter, max_iters, self._cum_losses['total_loss']/div, self._cum_losses['rpn_cross_entropy']/div, self._cum_losses['rpn_loss_box']/div, self._cum_losses['cross_entropy']/div, self._cum_losses['loss_box']/div, lr))


    def load_state_dict(self, state_dict):
        """
    Because we remove the definition of fc layer in resnet now, it will fail when loading 
    the model trained before.
    To provide back compatibility, we overwrite the load_state_dict
    """
        nn.Module.load_state_dict(
            self, {k: v
                   for k, v in state_dict.items() if k in self.state_dict()}
            )

    """
    Function: _draw_and_save_targets
    Useful for debug, place within forward() loop. Allows visualization of anchor(1st stage) or proposal(2nd stage) targets over the voxelgrid/image array
    Arguments:
    ----------
    frame       -> the input frame
    info        -> size of the frame (x_min,x_max,y_min,y_max,z_min,z_max)
    targets     -> (NxKx4 or NxKx7) where the targets are either for a 3d box or a 2d axis aligned box
    rois        -> (Nx4) axis aligned rois (BEV or FV), either anchors (1st stage) or true ROI's (2nd stage)
    labels      -> (N) label for each target
    mask        -> (NxKx4 or NxKx7) dictating which bbox targets are to be used to transform ROI's (background dets do not have associated targets)
    target_type -> anchor (1st stage) or proposal (2nd stage)
    net_type    -> image or lidar
    output:
    -------
    draws png files to a specific subfolder, dictated by cfg.DATA_DIR
    """
    def _draw_and_save_targets(self,frame,targets,rois,anchors_3d,labels,mask,target_type,net_type):
        datapath = os.path.join(cfg.DATA_DIR,'debug')
        if(target_type == 'anchor'):
           cnt = self._anchor_cnt
        elif(target_type == 'proposal'):
           cnt = self._proposal_cnt
        out_file = os.path.join(datapath,'{}_{}_target_{}.png'.format(cnt,target_type,net_type))
        if(net_type == 'lidar'):
            self._draw_and_save_lidar_targets(frame,targets,rois,anchors_3d,labels,mask,target_type,out_file)
        elif(net_type == 'image'):
            self._draw_and_save_image_targets(frame,targets,rois,labels,mask,target_type,out_file)
        print('Saving target file at location {}'.format(out_file))  
        if(target_type == 'anchor'):
            self._anchor_cnt += 1 
        elif(target_type == 'proposal'):
            self._proposal_cnt += 1

    def _draw_and_save_lidar_targets(self,frame,targets,rois,anchors_3d,labels,mask,target_type,out_file):
        voxel_grid = frame[0]
        voxel_grid_rgb = np.zeros((voxel_grid.shape[0],voxel_grid.shape[1],3))
        voxel_grid_rgb[:,:,0] = np.max(voxel_grid[:,:,0:cfg.LIDAR.NUM_SLICES],axis=2)
        max_height = np.max(voxel_grid_rgb[:,:,0])
        min_height = np.min(voxel_grid_rgb[:,:,0])
        voxel_grid_rgb[:,:,0] = np.clip(voxel_grid_rgb[:,:,0]*(255/(max_height - min_height)),0,255)
        voxel_grid_rgb[:,:,1] = voxel_grid[:,:,cfg.LIDAR.NUM_SLICES]*(255/voxel_grid[:,:,cfg.LIDAR.NUM_SLICES].max())
        voxel_grid_rgb[:,:,2] = voxel_grid[:,:,cfg.LIDAR.NUM_SLICES+1]*(255/voxel_grid[:,:,cfg.LIDAR.NUM_SLICES+1].max())
        voxel_grid_rgb        = voxel_grid_rgb.astype(dtype='uint8')
        img = Image.fromarray(voxel_grid_rgb,'RGB')
        draw = ImageDraw.Draw(img)
        if(target_type == 'anchor'):
            mask   = mask.view(-1,4)
            labels = labels.permute(0,2,3,1).reshape(-1)
            targets = targets.view(-1,4)
            rois = rois.view(-1,4)
            anchors = bbox_transform_inv(rois,targets)
            anchors = anchors.data.cpu().numpy()
            rois    = rois.data.cpu().numpy()
        #if(target_type == 'anchor'):
        if(target_type == 'proposal'):
            #Target is in a (N,K*7) format, transform to (N,7) where corresponding label dictates what class bbox belongs to 
            sel_targets = torch.where(labels == 0, targets[:,0:7],targets[:,7:14])
            #Get subset of mask for specific class selected
            mask = torch.where(labels == 0, mask[:,0:7], mask[:,7:14])
            sel_targets = sel_targets*mask
            rois = rois[:,1:5]
            rois    = rois.data.cpu().numpy()
            #Extract XC,YC and L,W
            targets = sel_targets
            stds = targets.data.new(cfg.TRAIN.LIDAR.BBOX_NORMALIZE_STDS).unsqueeze(0).expand_as(targets)
            means = targets.data.new(cfg.TRAIN.LIDAR.BBOX_NORMALIZE_MEANS).unsqueeze(0).expand_as(targets)
            targets = targets.mul(stds).add(means)
            targets = targets.view(-1,7)
            anchors = lidar_3d_bbox_transform_inv(anchors_3d,targets)
            anchors = anchors.data.cpu().numpy()
            anchors = bbox_utils.bbaa_graphics_gems(anchors,voxel_grid_rgb.shape[1],voxel_grid_rgb.shape[0])
            #rois = anchors
        #label_mask = labels + 1
        #label_idx  = label_mask.nonzero().squeeze(1).data.cpu().numpy()
        #anchors_filtered = anchors[label_idx,:].reshape(-1,4)
        #else:
        #    anchors = bbox_3d_transform_inv_all_boxes(anchors_3d,targets)
            #anchors = 3d_to_bev(anchors)
        for i, bbox in enumerate(anchors):
            bbox_mask = mask[i]
            bbox_label = int(labels[i])
            roi        = rois[i]
            np_bbox = None
            #if(torch.mean(bbox_mask) > 0):
            if(bbox_label == 1):
                np_bbox = bbox
                draw.text((np_bbox[0],np_bbox[1]),"class: {}".format(bbox_label))
                draw.rectangle(np_bbox,width=1,outline='green')
                if(np_bbox[0] >= np_bbox[2]):
                    print('x1 {} x2 {}'.format(np_bbox[0],np_bbox[2]))
                if(np_bbox[1] >= np_bbox[3]):
                    print('y1 {} y2 {}'.format(np_bbox[1],np_bbox[3]))
            elif(bbox_label == 0):
                np_bbox = roi
                draw.text((np_bbox[0],np_bbox[1]),"class: {}".format(bbox_label))
                draw.rectangle(np_bbox,width=1,outline='red')
                if(np_bbox[0] >= np_bbox[2]):
                    print('x1 {} x2 {}'.format(np_bbox[0],np_bbox[2]))
                if(np_bbox[1] >= np_bbox[3]):
                    print('y1 {} y2 {}'.format(np_bbox[1],np_bbox[3]))
        img.save(out_file,'png')

    def _draw_and_save_image_targets(self,frame,targets,rois,labels,mask,target_type,out_file):
        frame = frame[0]*cfg.PIXEL_STDDEVS + cfg.PIXEL_MEANS
        frame = frame.astype(dtype=np.uint8)
        img = Image.fromarray(frame,'RGB')
        draw = ImageDraw.Draw(img)
        if(target_type == 'anchor'):
            mask   = mask.view(-1,4)
            labels = labels.permute(0,2,3,1).reshape(-1)
        #if(target_type == 'anchor'):
        if(target_type == 'proposal'):
            #Target is in a (N,K*7) format, transform to (N,7) where corresponding label dictates what class bbox belongs to 
            sel_targets = torch.where(labels == 0, targets[:,0:4],targets[:,4:8])
            #Get subset of mask for specific class selected
            mask = torch.where(labels == 0, mask[:,0:4], mask[:,4:8])
            sel_targets = sel_targets*mask
            rois = rois[:,1:5]
            #Extract XC,YC and L,W
            targets = sel_targets
            stds = targets.data.new(cfg.TRAIN.IMAGE.BBOX_NORMALIZE_STDS[0:4]).unsqueeze(0).expand_as(targets)
            means = targets.data.new(cfg.TRAIN.IMAGE.BBOX_NORMALIZE_MEANS[0:4]).unsqueeze(0).expand_as(targets)
            targets = targets.mul(stds).add(means)
        rois = rois.view(-1,4)
        targets = targets.view(-1,4)
        anchors = bbox_transform_inv(rois,targets)
        label_mask = labels + 1
        label_idx  = label_mask.nonzero().squeeze(1)
        anchors_filtered = anchors[label_idx,:]
        #else:
        #    anchors = bbox_3d_transform_inv_all_boxes(anchors_3d,targets)
            #anchors = 3d_to_bev(anchors)
        for i, bbox in enumerate(anchors.view(-1,4)):
            bbox_mask = mask[i]
            bbox_label = int(labels[i])
            roi        = rois[i]
            np_bbox = None
            #if(torch.mean(bbox_mask) > 0):
            if(bbox_label == 1):
                np_bbox = bbox.data.cpu().numpy()
                draw.text((np_bbox[0],np_bbox[1]),"class: {}".format(bbox_label))
                draw.rectangle(np_bbox,width=1,outline='green')
                if(np_bbox[0] >= np_bbox[2]):
                    print('x1 {} x2 {}'.format(np_bbox[0],np_bbox[2]))
                if(np_bbox[1] >= np_bbox[3]):
                    print('y1 {} y2 {}'.format(np_bbox[1],np_bbox[3]))
            elif(bbox_label == 0):
                np_bbox = roi.data.cpu().numpy()
                draw.text((np_bbox[0],np_bbox[1]),"class: {}".format(bbox_label))
                draw.rectangle(np_bbox,width=1,outline='red')
                if(np_bbox[0] >= np_bbox[2]):
                    print('x1 {} x2 {}'.format(np_bbox[0],np_bbox[2]))
                if(np_bbox[1] >= np_bbox[3]):
                    print('y1 {} y2 {}'.format(np_bbox[1],np_bbox[3]))
        img.save(out_file,'png')


    """
    Function: _draw_and_save_anchors
    Useful for debug, place within forward() loop. Allows visualization of a subset of anchors over the voxelgrid/image array
    Arguments:
    ----------
    frame   -> the input frame
    anchors -> (Nx4) axis aligned anchors (BEV or FV)
    output:
    -------
    draws png files to a specific subfolder, dictated by cfg.DATA_DIR
    """
    def _draw_and_save_anchors(self, frame, anchors, net_type):
        datapath = os.path.join(cfg.DATA_DIR,'debug')
        out_file = os.path.join(datapath,'{}_anchors_{}.png'.format(self._cnt,net_type))
        if(net_type == 'lidar'):
            img = self._draw_and_save_lidar_anchors(frame,anchors)
        elif(net_type == 'image'):
            img = self._draw_and_save_image_anchors(frame,anchors)
        else:
            print('Cannot draw and save anchors for net type: {}'.format(net_type))
            img = None
        draw = ImageDraw.Draw(img)
        for i, bbox in enumerate(anchors.data.cpu().numpy()):
            if(i%900 < 3):
                c = (255,255,255)
                if(i%900 == 0):
                    c = (255,0,0)
                if(i%900 == 1):
                    c = (0,255,0)
                if(i%900 == 2):
                    c = (0,0,255)
                draw.rectangle(bbox,width=1,outline=(255,0,0))
        img.save(out_file,'png')
        print('Saving file at location {}'.format(out_file))  
        self._cnt += 1 

    def _draw_and_save_lidar_anchors(self,frame,anchors):
        voxel_grid = frame[0]
        voxel_grid_rgb = np.zeros((voxel_grid.shape[0],voxel_grid.shape[1],3))
        voxel_grid_rgb[:,:,0] = np.max(voxel_grid[:,:,0:cfg.LIDAR.NUM_SLICES],axis=2)
        max_height = np.max(voxel_grid_rgb[:,:,0])
        min_height = np.min(voxel_grid_rgb[:,:,0])
        voxel_grid_rgb[:,:,0] = np.clip(voxel_grid_rgb[:,:,0]*(255/(max_height - min_height)),0,255)
        voxel_grid_rgb[:,:,1] = voxel_grid[:,:,cfg.LIDAR.NUM_SLICES]*(255/voxel_grid[:,:,cfg.LIDAR.NUM_SLICES].max())
        voxel_grid_rgb[:,:,2] = voxel_grid[:,:,cfg.LIDAR.NUM_SLICES+1]*(255/voxel_grid[:,:,cfg.LIDAR.NUM_SLICES+1].max())
        voxel_grid_rgb        = voxel_grid_rgb.astype(dtype='uint8')
        img = Image.fromarray(voxel_grid_rgb,'RGB')
        return img

    def _draw_and_save_image_anchors(self,frame,anchors):
        frame = frame[0]*cfg.PIXEL_STDDEVS + cfg.PIXEL_MEANS
        frame = frame.astype(dtype=np.uint8)
        source_img = Image.fromarray(frame)
        return source_img
