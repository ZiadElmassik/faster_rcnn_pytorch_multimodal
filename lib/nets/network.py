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
import sys
from PIL import Image, ImageDraw
import os
from collections import OrderedDict
import utils.timer
import utils.torchpoolers as torchpooler

from layer_utils.snippets import generate_anchors_pre
from layer_utils.generate_3d_anchors import GridAnchor3dGenerator
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer, anchor_target_layer_torch
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes
from model.bbox_transform import bbox_transform_inv, lidar_3d_bbox_transform_inv, clip_boxes, lidar_3d_uncertainty_transform_inv, uncertainty_transform_inv

from torchvision.ops import RoIAlign, RoIPool
from torchvision.ops.poolers import MultiScaleRoIAlign
from model.config import cfg
import utils.bbox as bbox_utils
import tensorboardX as tb
import utils.loss_utils as loss_utils
import nets.resnet as custom_resnet
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
        self._frame_scale      = 1.0
        self._proposal_cnt     = 0
        self._anchors          = None
        self._anchors_cache    = None
        self._anchors_3d_cache = None
        self._device           = 'cuda'
        self._net_type                        = cfg.NET_TYPE
        self._bbox_means = torch.tensor(cfg.TRAIN[self._net_type.upper()].BBOX_NORMALIZE_MEANS).to(device=self._device)
        self._bbox_stds = torch.tensor(cfg.TRAIN[self._net_type.upper()].BBOX_NORMALIZE_STDS).to(device=self._device)
        self._cum_loss_keys    = ['total_loss','rpn_cross_entropy','rpn_loss_box']
        #self._cum_loss_keys    = ['total_loss','rpn_cross_entropy','rpn_loss_box','cross_entropy','loss_box']
        if(cfg.ENABLE_FULL_NET):
            self._cum_loss_keys.append('loss_box')
            self._cum_loss_keys.append('cross_entropy')
            #if(cfg.NET_TYPE == 'lidar'):
            #    self._cum_loss_keys.append('ry_loss')
            if(cfg.UC.EN_CLS_ALEATORIC):
                self._cum_loss_keys.append('a_cls_var')
            if(cfg.UC.EN_BBOX_ALEATORIC):
                self._cum_loss_keys.append('a_bbox_var')
        self._cum_gt_entries                  = 0
        self._batch_gt_entries                = 0
        self._cum_im_entries                  = 0
        self._e_num_sample                    = 1
        #Set on every forward pass for use with proposal target layer
        self._gt_boxes      = None
        self._true_gt_boxes = None
        self._gt_boxes_dc   = None
        if(cfg.UC.EN_BBOX_EPISTEMIC or cfg.UC.EN_CLS_EPISTEMIC):
            self._dropout_en = True
        else:
            self._dropout_en = False

    def _add_gt_image(self):
        # add back mean
        image = ((self._gt_summaries['frame']))*cfg.PIXEL_STDDEVS + cfg.PIXEL_MEANS
        #Flip info from (xmin,xmax,ymin,ymax) to (ymin,ymax,xmin,xmax) due to frame being rotated
        frame_range = np.asarray([self._info[3] - self._info[2], self._info[1] - self._info[0]])
        resized = frame_range / self._info[6]
        resized = resized.astype(dtype=np.int64)
        image = imresize(image[0], resized)
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
                                        self._anchors, self._num_anchors)
        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
        rois, rpn_scores, anchors_3d = proposal_layer(\
                                        rpn_cls_prob, rpn_bbox_pred, self._info, self._mode,
                                        self._anchors, self._anchors_3d, self._num_anchors)
        return rois, rpn_scores, anchors_3d

    def _anchor_target_layer(self, rpn_cls_score):
        #Remove rotation element if LiDAR
        # map of shape (..., H, W)
        height, width = rpn_cls_score.data.shape[1:3]

        #gt_boxes = self._gt_boxes.data.cpu().numpy()
        #gt_boxes_dc = self._gt_boxes_dc.data.cpu().numpy()

        #.data is used to pull a tensor from a pytorch variable. Deprecated, but it grabs a copy of the data that will not be tracked by gradients
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_torch(self._gt_boxes, self._gt_boxes_dc, self._info, self._anchors, self._num_anchors, height, width, self._device)
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
        anchor_target_dict = ['rpn_labels','rpn_bbox_targets','rpn_bbox_inside_weights','rpn_bbox_outside_weights']
        for anchor_target in anchor_target_dict:
            if(anchor_target not in self._anchor_targets.keys()):
                self._anchor_targets[anchor_target] = []
        self._anchor_targets['rpn_labels'].append(rpn_labels)
        self._anchor_targets['rpn_bbox_targets'].append(rpn_bbox_targets)
        self._anchor_targets['rpn_bbox_inside_weights'].append(rpn_bbox_inside_weights)
        self._anchor_targets['rpn_bbox_outside_weights'].append(rpn_bbox_outside_weights)

        for k in self._anchor_targets.keys():
            self._score_summaries[k] = self._anchor_targets[k]

    def _proposal_target_layer(self, rois, roi_scores, anchors_3d):
        num_bbox_elem = cfg[cfg.NET_TYPE.upper()].NUM_BBOX_ELEM
        labels, rois, anchors_3d, roi_scores, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer(rois, roi_scores, anchors_3d, self._gt_boxes, self._true_gt_boxes, self._gt_boxes_dc, self._num_classes, num_bbox_elem)

        self._proposal_targets['rois']                 = rois
        self._proposal_targets['labels']               = labels.long()
        self._proposal_targets['bbox_targets']         = bbox_targets
        self._proposal_targets['bbox_inside_weights']  = bbox_inside_weights
        self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

        for k in self._proposal_targets.keys():
            self._score_summaries[k] = self._proposal_targets[k]

        return rois, roi_scores, anchors_3d

    def _anchor_component(self, height, width, feat_stride):
        if(self._anchors_cache is None):
            self._anchors_cache    = {}
            self._anchors_3d_cache = {}
        if(feat_stride not in self._anchors_cache.items()):
            if(self._net_type == 'image'):
                if(cfg.USE_FPN):
                    anchor_scales = [self._anchor_scales[0] * feat_stride/self._feat_stride]
                else:
                    anchor_scales = self._anchor_scales
                anchors, anchor_length = generate_anchors_pre(\
                                                    height, width,
                                                    feat_stride, anchor_scales, self._anchor_ratios, self._frame_scale)
                #TODO: Unused, shouldn't use unless lidar. fix please.
                self._anchors_3d_cache[feat_stride] = torch.from_numpy(anchors).to(self._device)
            elif(self._net_type == 'lidar'):
                anchor_generator = GridAnchor3dGenerator()
                anchor_length, anchors = anchor_generator._generate(height, width, feat_stride, self._anchor_scales, self._anchor_ratios, self._frame_scale)
                self._anchors_3d_cache[feat_stride] = torch.from_numpy(anchors).to(self._device)
                anchors = bbox_utils.bbaa_graphics_gems(anchors, (width)*feat_stride-1, (height)*feat_stride-1)
            self._anchors_cache[feat_stride] = torch.from_numpy(anchors).to(self._device)
            #self._anchors = torch.from_numpy(anchors).to(self._device)
        self._anchors       = self._anchors_cache[feat_stride]
        self._anchors_3d    = self._anchors_3d_cache[feat_stride]
        self._anchor_length = len(self._anchors_cache[feat_stride])
        if((cfg.DEBUG.DRAW_ANCHORS and cfg.DEBUG.EN) or cfg.DEBUG.DRAW_ANCHOR_T):
            self._anchor_targets['anchors'] = self._anchors

    def _add_rpn_losses(self,sigma_rpn,rpn_cls_score,rpn_label,rpn_bbox_pred,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights):
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

        rpn_loss_box = loss_utils.smooth_l1_loss(
            'RPN',
            rpn_bbox_pred,
            rpn_bbox_targets,
            [],
            rpn_bbox_inside_weights,
            rpn_bbox_outside_weights,
            sigma=sigma_rpn,
            dim=[1, 2, 3])
        return rpn_cross_entropy, rpn_loss_box

    #Determine losses for single batch image
    def _add_losses(self, sigma_rpn=3.0):
        # RPN, class loss
        rpn_cross_entropy = None
        rpn_loss_box      = None
        num_fpn_layers = len(self._predictions['rpn_bbox_pred'])
        for i,_ in enumerate(self._predictions['rpn_bbox_pred']):
            #View rearranges the matrix to match specified dimension -1 is inferred from other dims, probably OBJ/Not OBJ
            rpn_cls_score = self._predictions['rpn_cls_score_reshape'][i].view(-1, 2)
            #What is the target label out of the RPN
            rpn_label     = self._anchor_targets['rpn_labels'][i].view(-1)
            #Pretty sure these are delta's at this point
            rpn_bbox_pred            = self._predictions['rpn_bbox_pred'][i]
            rpn_bbox_targets         = self._anchor_targets['rpn_bbox_targets'][i]
            rpn_bbox_inside_weights  = self._anchor_targets['rpn_bbox_inside_weights'][i]
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights'][i]
            s_rpn_cross_entropy, s_rpn_loss_box = self._add_rpn_losses(sigma_rpn,rpn_cls_score,rpn_label,rpn_bbox_pred,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights)
            if(rpn_cross_entropy is None):
                rpn_cross_entropy = s_rpn_cross_entropy
            else:
                rpn_cross_entropy += s_rpn_cross_entropy
            if(rpn_loss_box is None):
                rpn_loss_box = s_rpn_loss_box
            else:
                rpn_loss_box += s_rpn_loss_box
        if(not cfg.USE_FPN):
            assert num_fpn_layers == 1
        rpn_cross_entropy = rpn_cross_entropy/num_fpn_layers
        rpn_loss_box      = rpn_loss_box/num_fpn_layers
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
            if(cfg.UC.EN_CLS_ALEATORIC):
                a_cls_var  = self._predictions['a_cls_var']
                cross_entropy, a_cls_mutual_info = loss_utils.bayesian_cross_entropy(cls_score, a_cls_var, label,cfg.UC.A_NUM_CE_SAMPLE)
                self._losses['a_cls_var']     = torch.mean(a_cls_var)
                #Classical entropy w/out logit sampling
                #self._losses['a_entropy'] = torch.mean(loss_utils.categorical_entropy(cls_prob))
            else:
                cross_entropy = F.cross_entropy(cls_score, label)
            #Compute aleatoric bbox variance
            if(cfg.UC.EN_BBOX_ALEATORIC):
                #Grab a local variable for a_bbox_var
                a_bbox_var  = self._predictions['a_bbox_var']
                #network output comes out as log(a_bbox_var), so it needs to be adjusted
                #TODO: Should i only care about top output variance? Thats what bbox_inside_weights does
                self._losses['a_bbox_var'] = (torch.exp(a_bbox_var)*bbox_inside_weights).mean()
            else:
                a_bbox_var = None
                #self._losses['a_bbox_var'] = torch.tensor(0)
            #Compute loss box
            loss_box = loss_utils.smooth_l1_loss('DET', bbox_pred, bbox_targets, a_bbox_var, bbox_inside_weights, bbox_outside_weights)
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

    def _region_proposal(self, net_conv):
        #this links net conv -> rpn_net -> relu
        rpn = F.relu(self.rpn_net(net_conv))
        #print('RPN result')
        #print(rpn)
        self._act_summaries['rpn'] = rpn
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
            #self.timers['proposal_t'].toc()
        else:
            if cfg.TEST.MODE == 'nms':
                rois, roi_scores, anchors_3d = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
            elif cfg.TEST.MODE == 'top':
                rois, roi_scores, anchors_3d = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
            else:
                raise NotImplementedError
        if(self._mode != 'TEST'):
            rpn_pred_dict = ['rpn_cls_score_reshape','rpn_bbox_pred']
            for rpn_pred in rpn_pred_dict:
                if(rpn_pred not in self._predictions.keys()):
                    self._predictions[rpn_pred] = []
            self._predictions['rpn_cls_score_reshape'].append(rpn_cls_score_reshape)
            self._predictions['rpn_bbox_pred'].append(rpn_bbox_pred)
        return rois, roi_scores, anchors_3d
    #Used to dynamically change batch size depending on eval or train
    def set_e_num_sample(self,e_num_sample):
        self._e_num_sample = e_num_sample


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

        if(cfg.USE_FPN):
            self._num_anchors = self._num_ratios
        else:
            self._num_anchors = self._num_scales * self._num_ratios

        assert tag != None

        # Initialize layers
        self._init_modules()

    def _build_resnet(self):
        # choose different blocks for different number of layers
        if self._num_resnet_layers == 50:
            resnet = custom_resnet.resnet50(dropout_en=False,drop_rate=self._resnet_drop_rate, batchnorm_en=self._batchnorm_en)

        elif self._num_resnet_layers == 34:
            resnet = custom_resnet.resnet34(dropout_en=False,drop_rate=self._resnet_drop_rate, batchnorm_en=self._batchnorm_en)
            
        elif self._num_resnet_layers == 101:
            resnet = custom_resnet.resnet101(dropout_en=False,drop_rate=self._resnet_drop_rate, batchnorm_en=self._batchnorm_en)

        elif self._num_resnet_layers == 152:
            resnet = custom_resnet.resnet152(dropout_en=False,drop_rate=self._resnet_drop_rate, batchnorm_en=self._batchnorm_en)

        else:
            # other numbers are not supported
            raise NotImplementedError
            return None
        return resnet

    #TODO: Maybe add another sub function here to be placed in respective sensor network classes
    def _init_modules(self):
        self._init_head_tail()

        # rpn
        self.rpn_net = nn.Conv2d(
            self._net_conv_channels, cfg.RPN_CHANNELS, [3, 3], padding=1)
        self.rpn_cls_score_net = nn.Conv2d(cfg.RPN_CHANNELS, self._num_anchors * 2, [1, 1])

        self.rpn_bbox_pred_net = nn.Conv2d(cfg.RPN_CHANNELS,
                                           self._num_anchors * 4, [1, 1])
        if(cfg.ENABLE_CUSTOM_TAIL):
            self.t_fc1           = nn.Linear(self._roi_pooling_channels,self._fc7_channels*4)
            self.t_fc2           = nn.Linear(self._fc7_channels*4,self._fc7_channels*2)
            self.t_fc3           = nn.Linear(self._fc7_channels*2,self._fc7_channels)
            self.t_relu          = nn.ReLU(inplace=True)

        #Epistemic dropout layers
        if(cfg.UC.EN_BBOX_EPISTEMIC):
            self.bbox_fc1        = nn.Linear(self._fc7_channels, self._det_net_channels*2)
            self.bbox_bn1        = nn.BatchNorm1d(self._det_net_channels*2)
            self.bbox_drop1      = nn.Dropout(self._bbox_drop_rate)
            self.bbox_fc2        = nn.Linear(self._det_net_channels*2, self._det_net_channels)
            self.bbox_bn2        = nn.BatchNorm1d(self._det_net_channels)
            self.bbox_drop2      = nn.Dropout(self._bbox_drop_rate)
        #    self.bbox_fc3        = nn.Linear(self._det_net_channels*2, self._det_net_channels)
        if(cfg.UC.EN_CLS_EPISTEMIC):
            self.cls_fc1        = nn.Linear(self._fc7_channels, self._det_net_channels*2)
            self.cls_bn1        = nn.BatchNorm1d(self._det_net_channels*2)
            self.cls_drop1      = nn.Dropout(self._cls_drop_rate)
            self.cls_fc2        = nn.Linear(self._det_net_channels*2, self._det_net_channels)
            self.cls_bn2        = nn.BatchNorm1d(self._det_net_channels)
            self.cls_drop2      = nn.Dropout(self._cls_drop_rate)
        #    self.cls_fc3        = nn.Linear(self._det_net_channels*2, self._det_net_channels)

        #Traditional outputs
        self.cls_score_net       = nn.Linear(self._det_net_channels, self._num_classes)
        self.bbox_pred_net       = nn.Linear(self._det_net_channels, self._num_classes * cfg[cfg.NET_TYPE.upper()].NUM_BBOX_ELEM)
        #self.bbox_z_pred_net     = nn.Linear(self._det_net_channels, self._num_classes * 2)
        #self.heading_pred_net    = nn.Linear(self._det_net_channels, self._num_classes)

        #Aleatoric leafs
        if(cfg.UC.EN_CLS_ALEATORIC):
            self.cls_al_var_net   = nn.Linear(self._det_net_channels,self._num_classes)
        if(cfg.UC.EN_BBOX_ALEATORIC):
            self.bbox_al_var_net  = nn.Linear(self._det_net_channels, self._num_classes * cfg[cfg.NET_TYPE.upper()].NUM_BBOX_ELEM)
        self.init_weights()

    #FYI this is a fancy way of instantiating a class and calling its main function
    def _roi_pool_layer(self, bottom, rois):
        #Has restriction on batch, only one dim allowed
        return RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE),
                       1.0 / float(self._feat_stride))(bottom, rois)

    #torchvision.ops.RoIAlign(output_size, spatial_scale, sampling_ratio)
    def _roi_align_layer(self, bottom, rois):
        return RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / float(self._feat_stride),
                        0)(bottom, rois)

    def _multiscale_roi_align_layer(self, bottom, rois):
        img_size = [(bottom['p2'].shape[2]*self._feat_stride, bottom['p2'].shape[3]*self._feat_stride)]
        rois = [rois[:,1:5]]
        return torchpooler.MultiScaleRoIAlign(['p2','p3','p4','p5'], (cfg.POOLING_SIZE, cfg.POOLING_SIZE), sampling_ratio=0)(bottom, rois, img_size)

    def _crop_pool_layer(self, bottom, rois):
        return Network._crop_pool_layer(self, bottom, rois,
                                        cfg.RESNET.MAX_POOL)

    def _input_to_head(self,frame):
        if(self._fpn_en):
            c1 = self._layers['head'](frame)
            c2 = self._layers['layer1'](c1)
            c3 = self._layers['layer2'](c2)
            c4 = self._layers['layer3'](c3)
            c5 = self._layers['layer4'](c4)
            p2, p3, p4, p5 = self._layers['fpn'](c2, c3, c4, c5)
            if(cfg.POOLING_MODE == 'multiscale'):
                net_conv = OrderedDict()
                net_conv['p2'] = p2
                net_conv['p3'] = p3
                net_conv['p4'] = p4
                net_conv['p5'] = p5
            else:
                net_conv = p2
        else:   
            net_conv = self._layers['head'](frame)

        self._act_summaries['conv'] = net_conv
        return net_conv

    def _head_to_tail(self, pool5):
        #pool5 = pool5.unsqueeze(0).repeat(self._e_num_sample,1,1,1,1)
        #pool5 = pool5.view(-1,pool5.shape[2],pool5.shape[3],pool5.shape[4])
        #Reshape due to limitation on nn.conv2d (only one dim can be batch)
        #pool5 = pool5.view(-1,pool5.shape[2],pool5.shape[3],pool5.shape[4])
        fc7 = self.resnet.layer4(pool5).mean(3).mean(2)  # average pooling after layer4
        #fc7 = fc7.unsqueeze(0).view(self._e_num_sample,-1,fc7.shape[1])
        #fc7 = torch.nn.ReLU(fc7)
        return fc7


    def _region_classification(self, fc7):

        #First pass through custom tail if epistemic uncertainty enabled
        if(cfg.UC.EN_CLS_EPISTEMIC):
            cls_score_in = self._cls_tail(fc7)
        else:
            cls_score_in = fc7
        cls_score = self.cls_score_net(cls_score_in)
        if(cfg.UC.EN_BBOX_EPISTEMIC):
            bbox_pred_in = self._bbox_tail(fc7)
        else:
            bbox_pred_in = fc7

        bbox_s_pred  = self.bbox_pred_net(bbox_pred_in)
        #if(cfg.NET_TYPE == 'lidar'):
        #    heading_pred = self.heading_pred_net(bbox_pred_in)
        #    bbox_z_pred  = self.bbox_z_pred_net(bbox_pred_in)
        #    #Mix heading back into bbox pred
        #    bbox_pred    = torch.cat((bbox_s_pred.view(self._e_num_sample,-1,cfg.IMAGE.NUM_BBOX_ELEM),bbox_z_pred.view(self._e_num_sample,-1,2),heading_pred.view(self._e_num_sample,-1,1)),dim=2)
        #    bbox_pred    = bbox_pred.view(self._e_num_sample,-1,cfg.LIDAR.NUM_BBOX_ELEM*self._num_classes)
        #else:
        bbox_pred = bbox_s_pred
        cls_score_mean = torch.mean(cls_score,dim=0)
        cls_pred = torch.max(cls_score_mean, 1)[1]
        cls_prob = F.softmax(cls_score, dim=2)
        cls_prob_mean = torch.mean(cls_prob,dim=0)

        #Compute aleatoric unceratinty if computed
        if(cfg.UC.EN_BBOX_ALEATORIC):
            bbox_var  = self.bbox_al_var_net(bbox_pred_in)
            self._predictions['a_bbox_var']  = torch.mean(bbox_var,dim=0)
        if(cfg.UC.EN_CLS_ALEATORIC):
            a_cls_var   = self.cls_al_var_net(cls_score_in)
            a_cls_var = torch.exp(torch.mean(a_cls_var,dim=0))  #exp of mean or mean of exp?
            self._predictions['a_cls_var']   = a_cls_var

        self._mc_run_output['bbox_pred'] = bbox_pred
        self._mc_run_output['cls_score'] = cls_score
        self._mc_run_output['cls_prob']  = cls_prob
        self._predictions['cls_score'] = cls_score_mean
        self._predictions['cls_pred'] = cls_pred
        self._predictions['cls_prob'] = cls_prob_mean
        #TODO: Make domain shift here
        self._predictions['bbox_pred'] = torch.mean(bbox_pred,dim=0)

    def _cls_tail(self,fc7):
        fc7_reshape = fc7.view(-1,fc7.shape[2])
        fc_relu      = nn.ReLU(inplace=True)
        if(self._dropout_en):
            out  = self.cls_drop1(fc7_reshape)
        out  = self.cls_fc1(out)
        out  = self.cls_bn1(out)
        out  = fc_relu(out)
        if(self._dropout_en):
            out  = self.cls_drop1(out)
        out  = self.cls_fc2(out)
        out  = self.cls_bn2(out)
        out  = fc_relu(out)
        if(self._dropout_en):
            out  = self.cls_drop2(out)
        out = out.view(fc7.shape[0],fc7.shape[1],-1)
        return out

    def _bbox_tail(self,fc7):
        fc7_reshape = fc7.view(-1,fc7.shape[2])
        fc_relu      = nn.ReLU(inplace=True)
        out     = self.bbox_fc1(fc7_reshape)
        out   = self.bbox_bn1(out)
        out   = fc_relu(out)
        if(self._dropout_en):
            out   = self.bbox_drop1(out)
        out   = self.bbox_fc2(out)
        out   = self.bbox_bn2(out)
        out   = fc_relu(out)
        if(self._dropout_en):
            out   = self.bbox_drop2(out)
        out = out.view(fc7.shape[0],fc7.shape[1],-1)
        return out

    def _custom_tail(self,pool5):
        pool5 = pool5.view(pool5.shape[0],-1).unsqueeze(0).repeat(self._e_num_sample,1,1)
        #pool5 = pool5.mean(3).mean(2).unsqueeze(0).repeat(self._e_num_sample,1,1)
        if(self._dropout_en):
            conv_dropout_rate = 0.2
            fc_dropout_rate   = 0.4
        else:
            conv_dropout_rate = 0.0
            fc_dropout_rate   = 0.0
        pool_dropout = nn.Dropout(conv_dropout_rate)
        fc_dropout1   = nn.Dropout(fc_dropout_rate)
        fc_dropout2   = nn.Dropout(fc_dropout_rate)
        fc_dropout3   = nn.Dropout(fc_dropout_rate)
        fc_dropout4   = nn.Dropout(fc_dropout_rate)
        fc_relu      = nn.ReLU(inplace=True)
        x   = self.t_fc1(pool5)
        x   = self.t_relu(x)
        if(self._dropout_en):
            x   = fc_dropout1(x)
        x   = self.t_fc2(x)
        x   = self.t_relu(x)
        if(self._dropout_en):
            x   = fc_dropout2(x)
        x   = self.t_fc3(x)
        x   = self.t_relu(x)
        if(self._dropout_en):
            x   = fc_dropout3(x)
        return x

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
            #for key, var in self._score_summaries.items():
            #    summaries.append(self._add_score_summary(key, var))
            self._score_summaries = {}
            # Add act summaries
            #for key, var in self._act_summaries.items():
            #    summaries += self._add_act_summary(key, var)
            self._act_summaries = {}
            # Add train summaries
            #for k, var in dict(self.named_parameters()).items():
            #    if var.requires_grad:
            #        summaries.append(self._add_train_summary(k, var))

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
        #self.timers['anchor_gen'].tic()
        if(cfg.USE_FPN):
            rois = []
            roi_scores = []
            anchor_3d  = []
            feat_stride = self._feat_stride
            rois       = None
            roi_scores = None
            anchors_3d = None
            #Run RPN for each feature level
            for k, v in net_conv.items():
                self._anchor_component(v.size(2), v.size(3), feat_stride)
                s_rois, s_roi_scores, s_anchors_3d = self._region_proposal(v)
                feat_stride = feat_stride * 2
                if(rois is None):
                    rois = s_rois
                else:
                    rois = torch.cat((rois,s_rois),dim=0)
                if(roi_scores is None):
                    roi_scores = s_roi_scores
                else:
                    roi_scores = torch.cat((roi_scores,s_roi_scores),dim=0)
                if(anchors_3d is None):
                    anchors_3d = s_anchors_3d
                else:
                    anchors_3d = torch.cat((anchors_3d,s_anchors_3d),dim=0)
        else:
            self._anchor_component(net_conv.size(2), net_conv.size(3), self._feat_stride)
            rois, roi_scores, anchors_3d = self._region_proposal(net_conv)
        if(cfg.ENABLE_FULL_NET):
            if self._mode == 'TRAIN':
                rois, _, anchors_3d = self._proposal_target_layer(rois, roi_scores, anchors_3d)
            if cfg.POOLING_MODE == 'multiscale':
                pool5 = self._multiscale_roi_align_layer(net_conv, rois)
            elif cfg.POOLING_MODE == 'align':
                pool5 = self._roi_align_layer(net_conv, rois)
            else:
                pool5 = self._roi_pool_layer(net_conv, rois)
            #del net_conv
            if self._mode == 'TRAIN':
                torch.backends.cudnn.benchmark = True  # benchmark because now the input size are fixed
            if(cfg.ENABLE_CUSTOM_TAIL):
                fc7 = self._custom_tail(pool5)
            else:
                fc7 = self._head_to_tail(pool5)
                fc7 = fc7.unsqueeze(0).repeat(self._e_num_sample,1,1)

            self._region_classification(fc7)
            #self.timers['net'].toc()
        self._predictions['rois']       = rois
        self._predictions['roi_scores'] = roi_scores
        self._predictions['anchors_3d'] = anchors_3d
        for k in self._predictions.keys():
            self._score_summaries[k] = self._predictions[k]

    def forward(self, frame, info=None, gt_boxes=None, gt_boxes_dc=None, mode='TRAIN'):
        self._gt_summaries['frame'] = frame
        self._gt_summaries['gt_boxes'] = gt_boxes
        self._gt_summaries['gt_boxes_dc'] = gt_boxes_dc
        self._gt_summaries['info'] = info
        self._info = info  # No need to change; actually it can be an list
        self._frame_scale = info[6]
        self._frame = torch.from_numpy(frame.transpose([0, 3, 1,
                                                        2])).to(self._device)
        if(mode == 'TRAIN' or mode == 'VAL'):
            if(self._net_type == 'image'):
                true_gt_boxes = gt_boxes

            elif(self._net_type == 'lidar'):
                #TODO: Should info contain bev extants? Seems like the cleanest way
                gt_box_labels = gt_boxes[:, -1, np.newaxis]
                gt_bboxes     = gt_boxes[:, :-1]
                gt_boxes      = bbox_utils.bbaa_graphics_gems(gt_bboxes,info[1],info[3])
                gt_boxes      = np.concatenate((gt_boxes, gt_box_labels),axis=1)
                #Still in 3D format
                true_gt_boxes = np.concatenate((gt_bboxes, gt_box_labels),axis=1)
                #Dont care areas
                #gt_boxes_dc   = bbox_utils.bbox_3d_to_bev_axis_aligned(gt_boxes_dc)
                gt_boxes_dc = None
                #gt_boxes_dc   = bbox_utils.bbox_bev_to_voxel_grid(gt_boxes,self._bev_extants,info)
        else:
            true_gt_boxes = None
        

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
        if(cfg.DEBUG.DRAW_ANCHORS and cfg.DEBUG.EN):
            for i,(k,v) in enumerate(self._anchors_cache.items()):
                self._draw_and_save_anchors(frame,
                                            v,
                                            self._net_type,
                                            k)
        #ENABLE to draw all anchor targets
        if(cfg.DEBUG.DRAW_ANCHOR_T and cfg.DEBUG.EN):
            for i,(k,v) in enumerate(self._anchors_cache.items()):
                self._draw_and_save_targets(frame,
                                            self._anchor_targets['rpn_bbox_targets'][i],
                                            v,
                                            None,
                                            self._anchor_targets['rpn_labels'][i],
                                            self._anchor_targets['rpn_bbox_inside_weights'][i],
                                            'anchor',
                                            self._net_type,
                                            k)
        #ENABLE to draw all proposal targets
        if(cfg.DEBUG.DRAW_PROPOSAL_T and cfg.DEBUG.EN):
            self._draw_and_save_targets(frame,
                                        self._proposal_targets['bbox_targets'],
                                        self._proposal_targets['rois'],
                                        self._predictions['anchors_3d'],
                                        self._proposal_targets['labels'],
                                        self._proposal_targets['bbox_inside_weights'],
                                        'proposal',
                                        self._net_type,
                                        0)

        if(mode == 'VAL' or mode == 'TRAIN'):
            #self.timers['losses'].tic()
            self._add_losses()  # compute losses
            #self.timers['losses'].toc()
        if(mode == 'VAL' or mode == 'TEST'):
            if(cfg.ENABLE_FULL_NET):
                bbox_pred = self._predictions['bbox_pred']
                rois      = self._predictions['rois'][:,1:]
                #bbox_targets are pre-normalized for loss, so modifying here.
                #Denormalize bbox target predictions
                bbox_mean = bbox_pred.mul(self._bbox_stds.repeat(self._num_classes)).add(self._bbox_means.repeat(self._num_classes))
                #bbox_mean = bbox_pred.mul(self._bbox_stds).add(self._bbox_means)
                anchors_3d = self._predictions['anchors_3d']
                rois = self._predictions['rois'][:,1:5]
                #TODO: Clean up to speed up - how to extract variance without inversion
                if(cfg.UC.EN_BBOX_ALEATORIC):
                    #DEPRECATED - need jacobian matrices for this.
                    #a_bbox_var = torch.exp(self._predictions['a_bbox_var'])
                    #if(self._net_type == 'image'):
                    #    uncertainty_inv = uncertainty_transform_inv(rois,bbox_mean,a_bbox_var,self._frame_scale)
                    #elif(self._net_type == 'lidar'):
                    #    uncertainty_inv = lidar_3d_uncertainty_transform_inv(rois,anchors_3d,bbox_mean,a_bbox_var,self._frame_scale)
                    #self._predictions['a_bbox_var_inv'] = uncertainty_inv
                    
                    #Must do monte-carlo sampling due to 
                    bbox_gaussian = torch.distributions.Normal(0,torch.sqrt(torch.exp(self._predictions['a_bbox_var'])))
                    bbox_samples = bbox_gaussian.sample((cfg.UC.A_NUM_BBOX_SAMPLE,)) + bbox_mean
                    #Manually expand anchors to match the multiplier for monte-carlo samples
                    roi_coords = rois.unsqueeze(0).repeat(cfg.UC.A_NUM_BBOX_SAMPLE,1,1)
                    roi_coords = roi_coords.view(-1,roi_coords.shape[2])
                    bbox_samples = bbox_samples.view(-1,bbox_samples.shape[2])
                    if(self._net_type == 'image'):
                        bbox_inv_samples = bbox_transform_inv(roi_coords,bbox_samples,self._frame_scale)
                    elif(self._net_type == 'lidar'):
                        #Manually expand anchors just as ROI's to match the bbox_sample multiplier
                        anchor_3d_coords = anchors_3d.unsqueeze(0).repeat(cfg.UC.A_NUM_BBOX_SAMPLE,1,1)
                        anchor_3d_coords = anchor_3d_coords.view(-1,anchor_3d_coords.shape[2])
                        bbox_inv_samples = lidar_3d_bbox_transform_inv(roi_coords,anchor_3d_coords,bbox_samples,self._frame_scale)
                        #Convert into true point cloud scale before computing variance
                        area_extents = [cfg.LIDAR.X_RANGE[0],cfg.LIDAR.Y_RANGE[0],cfg.LIDAR.Z_RANGE[0],cfg.LIDAR.X_RANGE[1],cfg.LIDAR.Y_RANGE[1],cfg.LIDAR.Z_RANGE[1]]
                        bbox_inv_samples = bbox_utils.bbox_voxel_grid_to_pc(bbox_inv_samples,area_extents,info)
                    bbox_inv_samples = bbox_inv_samples.view(cfg.UC.A_NUM_BBOX_SAMPLE,-1,bbox_inv_samples.shape[1])
                    bbox_inv_var = loss_utils.compute_bbox_var(bbox_inv_samples)
                    self._predictions['a_bbox_var_inv'] = bbox_inv_var

                #Traditional bbox output
                if(self._net_type == 'image'):
                    mean_bbox_inv = bbox_transform_inv(rois,bbox_mean,self._frame_scale)
                elif(self._net_type == 'lidar'):
                    mean_bbox_inv = lidar_3d_bbox_transform_inv(self._predictions['rois'][:,1:5],anchors_3d,bbox_mean,self._frame_scale)
                
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
        cls_score, cls_prob, bbox_pred, rois, anchors_3d = self._predictions["cls_score"].data.cpu().detach(), \
                                                         self._predictions['cls_prob'].data.detach(), \
                                                         self._predictions['bbox_inv_pred'].data.detach(), \
                                                         self._predictions['rois'].data.detach(), \
                                                         self._predictions['anchors_3d'].data.detach()
        #a_bbox_var, e_bbox_var, a_cls_entropy, a_cls_var, e_cls_mutual_info
        uncertainties = self._uncertainty_postprocess(bbox_pred,cls_prob,rois,anchors_3d,scale)
        return cls_score, cls_prob, bbox_pred, rois, uncertainties

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
        scale = blobs['info'][6]
        if(cfg.ENABLE_FULL_NET):
            summary           = None
            bbox_pred         = self._predictions['bbox_inv_pred'].data.detach() #(self._fc7_channels, self._num_classes * 4)
            cls_prob          = self._predictions['cls_prob'].data.detach() #(self._fc7_channels, self._num_classes)
            rois              = self._predictions['rois'].data.detach()/scale
            roi_labels        = self._proposal_targets['labels'].data.detach()
            anchors_3d        = self._predictions['anchors_3d'].data.detach()
            uncertainties = self._uncertainty_postprocess(bbox_pred,cls_prob,rois,anchors_3d,blobs['info'][6])
        else:
            summary    = None
            bbox_pred  = self._predictions['rois'][:,1:5]/scale
            cls_prob   = self._predictions['roi_scores']
            rois       = self._gt_boxes[:, :4]/scale
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

    def _uncertainty_postprocess(self,bbox_pred,cls_prob,rois,anchors_3d,frame_scale):
        uncertainties = {}
        if(cfg.UC.EN_CLS_ALEATORIC):
            cls_prob  = self._predictions['cls_prob']
            cls_score = self._predictions['cls_score']
            cls_var   = self._predictions['a_cls_var']
            #TODO: This should not be taking the mean yet, we need to filter by top indices
            #a_cls_entropy = -torch.mean(a_cls_entropy)*torch.log(torch.mean(a_cls_entropy)) - (1-torch.mean(a_cls_entropy))*torch.log(1-torch.mean(a_cls_entropy))
            a_cls_entropy                      = loss_utils.categorical_entropy(cls_prob)
            distorted_cls_score                = loss_utils.logit_distort(cls_score,cls_var,cfg.UC.A_NUM_CE_SAMPLE)
            a_cls_mutual_info                  = loss_utils.categorical_mutual_information(distorted_cls_score)
            uncertainties['a_entropy']         = a_cls_entropy.data.detach()
            uncertainties['a_mutual_info']     = a_cls_mutual_info
            uncertainties['a_cls_var']         = cls_var

        #For tensorboard
        if(cfg.UC.EN_CLS_EPISTEMIC):
            e_cls_score     = self._mc_run_output['cls_score'].detach()
            e_cls_prob      = self._mc_run_output['cls_prob'].detach()
            e_cls_prob_mean = torch.mean(e_cls_prob,dim=0)
            #Compute average entropy via mutual information
            e_cls_entropy                             = loss_utils.categorical_entropy(e_cls_prob_mean)
            e_cls_mutual_info                         = loss_utils.categorical_mutual_information(e_cls_score)
            self._mc_run_results['e_mutual_info']     = torch.mean(e_cls_mutual_info)
            self._mc_run_results['e_entropy']         = torch.mean(e_cls_entropy)
            uncertainties['e_entropy']                = e_cls_entropy
            uncertainties['e_mutual_info']            = e_cls_mutual_info

        if(cfg.UC.EN_BBOX_ALEATORIC):
            #Grab after bbox are transformed into pc space and MC sampling occurs
            uncertainties['a_bbox_var'] = self._predictions['a_bbox_var_inv'].data.detach()
            #Grab directly from output of deltas
            #uncertainties['a_bbox_var'] = torch.exp(self._predictions['a_bbox_var']).data.detach()

        if(cfg.UC.EN_BBOX_EPISTEMIC):
            #All of this to simply get the predictions from [M,N,C] to [M*N,C] interleaved.
            #This is to not change bbox_transform_inv
            #TODO: add mean and std deviation
            mc_bbox_pred = self._mc_run_output['bbox_pred']
            mc_bbox_pred = mc_bbox_pred.view(-1,mc_bbox_pred.shape[2])
            mc_bbox_pred = mc_bbox_pred.mul(self._bbox_stds.repeat(self._num_classes)).add(self._bbox_means.repeat(self._num_classes))
            roi_sampled  = rois[:,1:]
            roi_sampled  = roi_sampled.unsqueeze(0).repeat(self._e_num_sample,1,1)
            roi_sampled  = roi_sampled.view(-1,roi_sampled.shape[2])


            if(cfg.NET_TYPE == 'image'):
                mc_bbox_pred = bbox_transform_inv(roi_sampled,mc_bbox_pred,frame_scale)
            elif(cfg.NET_TYPE == 'lidar'):
                anchor_3d_sampled = anchors_3d.unsqueeze(0).repeat(self._e_num_sample,1,1)
                anchor_3d_sampled = anchor_3d_sampled.view(-1,anchor_3d_sampled.shape[2])
                mc_bbox_pred = lidar_3d_bbox_transform_inv(roi_sampled,anchor_3d_sampled,mc_bbox_pred,frame_scale)
            mc_bbox_pred = mc_bbox_pred.view(self._e_num_sample,-1,mc_bbox_pred.shape[1])
            #Way #1 to compute bbox var
            #mc_bbox_covar = loss_utils.compute_bbox_cov(mc_bbox_pred)
            #Way #2 to compute bbox var
            e_bbox_var   = loss_utils.compute_bbox_var(mc_bbox_pred)
            #Way #3 to compute bbox var (Doesnt work??)
            #e_bbox_var = torch.var(mc_bbox_pred,dim=0)
            uncertainties['e_bbox_var']        = e_bbox_var
            self._mc_run_output['e_bbox_var']  = e_bbox_var
            self._mc_run_results['e_bbox_var'] = torch.mean(e_bbox_var)
            #Compute average variance
        #else:
        #    uncertainties['e_bbox_var'] = torch.tensor([0])
        if(cfg.UC.EN_BBOX_EPISTEMIC or cfg.UC.EN_CLS_EPISTEMIC):
            for k in self._mc_run_results.keys():
                if(k in self._val_event_summaries):
                    self._val_event_summaries[k] += self._mc_run_results[k].item()
                else:
                    self._val_event_summaries[k]  = self._mc_run_results[k].item()
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

        self._batch_gt_entries        += len(blobs['gt_boxes'])
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
    def _draw_and_save_targets(self,frame,targets,rois,anchors_3d,labels,mask,target_type,net_type,fpn_cnt):
        datapath = os.path.join(cfg.ROOT_DIR,'debug')
        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        if(target_type == 'anchor'):
           cnt = self._anchor_cnt
           self._anchor_cnt += 1 
           out_file = os.path.join(datapath,'{}_{}_target_{}_stride_{}.png'.format(cnt,target_type,net_type,fpn_cnt))
        elif(target_type == 'proposal'):
           cnt = self._proposal_cnt
           self._proposal_cnt += 1
           out_file = os.path.join(datapath,'{}_{}_target_{}.png'.format(cnt,target_type,net_type))
        else:
            print('Error in draw_and_save_targets (network.py)')
        if(net_type == 'lidar'):
            self._draw_and_save_lidar_targets(frame,targets,rois,anchors_3d,labels,mask,target_type,out_file)
        elif(net_type == 'image'):
            self._draw_and_save_image_targets(frame,targets,rois,labels,mask,target_type,out_file)
        print('Saving target file at location {}'.format(out_file))  

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
            #Extract XC,YC and L,W
            targets = sel_targets
            targets = targets.mul(self._bbox_stds).add(self._bbox_means)
            targets = targets.view(-1,7)
            anchors = lidar_3d_bbox_transform_inv(rois,anchors_3d,targets)
            anchors = anchors.data.cpu().numpy()
            anchors = bbox_utils.bbaa_graphics_gems(anchors,voxel_grid_rgb.shape[1],voxel_grid_rgb.shape[0])
            rois    = rois.data.cpu().numpy()
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
        frame = frame[:,:,cfg.PIXEL_ARRANGE_BGR]
        frame = frame.astype(dtype=np.uint8)
        img = Image.fromarray(frame,'RGB')
        draw = ImageDraw.Draw(img)
        if(target_type == 'anchor'):
            mask   = mask.view(-1,4)
            labels = labels.permute(0,2,3,1)
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
            targets = targets.mul(self._bbox_stds).add(self._bbox_means)
        rois = rois.view(-1,4)
        labels = labels.reshape(-1)
        targets = targets.view(-1,4)
        anchors = bbox_transform_inv(rois,targets)
        label_mask = labels + 1
        label_idx  = label_mask.nonzero().squeeze(1)
        anchors_filtered = anchors[label_idx,:]
        for idx in label_idx:
            bbox       = anchors[idx]
            bbox_mask  = mask[idx]
            bbox_label = int(labels[idx])
            roi        = rois[idx]
            np_bbox = None
            if(bbox_label >= 1):
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
    def _draw_and_save_anchors(self, frame, anchors, net_type, fpn_cnt):
        datapath = os.path.join(cfg.ROOT_DIR,'debug')
        if not os.path.isdir(datapath):
            os.mkdir(datapath)
        out_file = os.path.join(datapath,'{}_anchors_{}_{}.png'.format(self._cnt,net_type,fpn_cnt))
        if(net_type == 'lidar'):
            img = self._draw_and_save_lidar_anchors(frame,anchors)
        elif(net_type == 'image'):
            img = self._draw_and_save_image_anchors(frame,anchors)
        else:
            print('Cannot draw and save anchors for net type: {}'.format(net_type))
            img = None
        draw = ImageDraw.Draw(img)
        for i, bbox in enumerate(anchors.data.cpu().numpy()):
            if(i%100 < 9):
                c = (255,255,255)
                if(i%3 == 0):
                    c = (255,0,0)
                if(i%3 == 1):
                    c = (0,255,0)
                if(i%3 == 2):
                    c = (0,0,255)
                draw.rectangle(bbox,width=1,outline=c)
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
