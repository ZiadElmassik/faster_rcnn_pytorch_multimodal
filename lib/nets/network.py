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

import utils.timer

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from torchvision.ops import RoIAlign, RoIPool

from model.config import cfg

import tensorboardX as tb

from scipy.misc import imresize


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._predictions = {}
        self._losses = {}
        self._cum_losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = {}
        self._score_summaries = {}
        self._val_event_summaries = {}
        self._event_summaries = {}
        self._image_gt_summaries = {}
        self._variables_to_fix = {}
        self._device = 'cuda'
        self._cum_losses['total_loss']        = 0
        self._cum_losses['rpn_cross_entropy'] = 0
        self._cum_losses['rpn_loss_box']      = 0
        self._cum_losses['cross_entropy']     = 0
        self._cum_losses['loss_box']          = 0
        self._cum_gt_entries                  = 0
        self._batch_gt_entries                = 0

    def _add_gt_image(self):
        # add back mean
        image = self._image_gt_summaries['image'] + cfg.PIXEL_MEANS
        image = imresize(image[0], self._im_info[:2] / self._im_info[2])
        # BGR to RGB (opencv uses BGR)
        self._gt_image = image[np.newaxis, :, :, ::-1].copy(order='C')

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        self._add_gt_image()
        image = draw_bounding_boxes(\
                          self._gt_image, self._image_gt_summaries['gt_boxes'], self._image_gt_summaries['im_info'])

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
        rois, rpn_scores = proposal_top_layer(\
                                        rpn_cls_prob, rpn_bbox_pred, self._im_info,
                                         self._feat_stride, self._anchors, self._num_anchors)
        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred):
        rois, rpn_scores = proposal_layer(\
                                        rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                         self._feat_stride, self._anchors, self._num_anchors)
        return rois, rpn_scores

    def _roi_pool_layer(self, bottom, rois):
        return RoIPool((cfg.POOLING_SIZE, cfg.POOLING_SIZE),
                       1.0 / 16.0)(bottom, rois)

    def _roi_align_layer(self, bottom, rois):
        return RoIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0 / 16.0,
                        0)(bottom, rois)

    def _anchor_target_layer(self, rpn_cls_score):
        #.data is used to pull a tensor from a pytorch variable. Deprecated, but it grabs a copy of the data that will not be tracked by gradients
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer(rpn_cls_score.data, self._gt_boxes.data.cpu().numpy(), self._gt_boxes_dc.data.cpu().numpy(), self._im_info, self._feat_stride, self._anchors.data.cpu().numpy(), self._num_anchors)

        rpn_labels = torch.from_numpy(rpn_labels).float().to(
            self._device)  #.set_shape([1, 1, None, None])
        rpn_bbox_targets = torch.from_numpy(rpn_bbox_targets).float().to(
            self._device)  #.set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_inside_weights = torch.from_numpy(
            rpn_bbox_inside_weights).float().to(
                self.
                _device)  #.set_shape([1, None, None, self._num_anchors * 4])
        rpn_bbox_outside_weights = torch.from_numpy(
            rpn_bbox_outside_weights).float().to(
                self.
                _device)  #.set_shape([1, None, None, self._num_anchors * 4])

        rpn_labels = rpn_labels.long()
        self._anchor_targets['rpn_labels'] = rpn_labels
        self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
        self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
        self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

        for k in self._anchor_targets.keys():
            self._score_summaries[k] = self._anchor_targets[k]

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores):
        rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer(rois, roi_scores, self._gt_boxes, self._gt_boxes_dc, self._num_classes)

        self._proposal_targets['rois'] = rois
        self._proposal_targets['labels'] = labels.long()
        self._proposal_targets['bbox_targets'] = bbox_targets
        self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
        self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

        for k in self._proposal_targets.keys():
            self._score_summaries[k] = self._proposal_targets[k]

        return rois, roi_scores

    def _anchor_component(self, height, width):
        # just to get the shape right
        #height = int(math.ceil(self._im_info.data[0, 0] / self._feat_stride[0]))
        #width = int(math.ceil(self._im_info.data[0, 1] / self._feat_stride[0]))
        anchors, anchor_length = generate_anchors_pre(\
                                              height, width,
                                               self._feat_stride, self._anchor_scales, self._anchor_ratios)
        self._anchors = torch.from_numpy(anchors).to(self._device)
        self._anchor_length = anchor_length

    def _smooth_l1_loss(self,
                        bbox_pred,
                        bbox_targets,
                        bbox_inside_weights,
                        bbox_outside_weights,
                        sigma=1.0,
                        dim=[1]):
        sigma_2 = sigma**2
        box_diff = bbox_pred - bbox_targets
        #print(bbox_targets.size())
        #print(bbox_pred.size())
        #Ignore diff when target is not a foreground target
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = torch.abs(in_box_diff)
        smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
        in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        #Used to normalize the predictions?
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = out_loss_box
        #Condense down to 1D array, each entry is the box_loss for an individual box, array is batch size of all predicted boxes
        #[loss,y,x,num_anchor]
        for i in sorted(dim, reverse=True):
            loss_box = loss_box.sum(i)
        #print(loss_box.size())
        loss_box = loss_box.mean()
        return loss_box
    #Determine losses for single batch image
    def _add_losses(self, sigma_rpn=3.0):
        # RPN, class loss
        #View rearranges the matrix to match specified dimension -1 is inferred from other dims, probably OBJ/Not OBJ
        rpn_cls_score = self._predictions['rpn_cls_score_reshape'].view(-1, 2)
        #What is the target label out of the RPN
        rpn_label = self._anchor_targets['rpn_labels'].view(-1)
        #Remove all non zeros to get an index list of target objects, not non-objects
        rpn_select = (rpn_label.data != -1).nonzero().view(-1)
        rpn_cls_score = rpn_cls_score.index_select(
            0, rpn_select).contiguous().view(-1, 2)
        #Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
        rpn_label = rpn_label.index_select(0, rpn_select).contiguous().view(-1)
        #torch.nn.functional
        #Compare labels from anchor_target_layer and rpn_layer
        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, reduction='mean')

        # RPN, bbox loss

        rpn_bbox_pred = self._predictions['rpn_bbox_pred']
        rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
        rpn_bbox_inside_weights = self._anchor_targets[
            'rpn_bbox_inside_weights']
        rpn_bbox_outside_weights = self._anchor_targets[
            'rpn_bbox_outside_weights']
        rpn_loss_box = self._smooth_l1_loss(
            rpn_bbox_pred,
            rpn_bbox_targets,
            rpn_bbox_inside_weights,
            rpn_bbox_outside_weights,
            sigma=sigma_rpn,
            dim=[1, 2, 3])

        # RCNN, class loss
        cls_score = self._predictions["cls_score"]
        label = self._proposal_targets["labels"].view(-1)
        cross_entropy = F.cross_entropy(
            cls_score.view(-1, self._num_classes), label)

        # RCNN, bbox loss
        bbox_pred = self._predictions['bbox_pred']
        bbox_targets = self._proposal_targets['bbox_targets']
        bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
        bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
        loss_box = self._smooth_l1_loss(
            bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

        self._losses['cross_entropy'] = cross_entropy
        self._losses['loss_box'] = loss_box
        self._losses['rpn_cross_entropy'] = rpn_cross_entropy
        self._losses['rpn_loss_box'] = rpn_loss_box
        #Computed losses
        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
        self._losses['total_loss'] = loss
        #print('individual image loss:{:f}'.format(loss))
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
        rpn_cls_pred = torch.max(rpn_cls_score_reshape.view(-1, 2), 1)[1]

        rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
        rpn_bbox_pred = rpn_bbox_pred.permute(
            0, 2, 3, 1).contiguous()  # batch * h * w * (num_anchors*4)

        if self._mode == 'TRAIN':
            rois, roi_scores = self._proposal_layer(
                rpn_cls_prob, rpn_bbox_pred)  # rois, roi_scores are varible
            #target labels and roi's to supply later half of network with golden results
            rpn_labels = self._anchor_target_layer(rpn_cls_score)
            rois, _ = self._proposal_target_layer(rois, roi_scores)
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred)
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred)
            else:
                raise NotImplementedError

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois

        return rois

    def _region_classification(self, fc7):
        cls_score = self.cls_score_net(fc7)
        cls_pred = torch.max(cls_score, 1)[1]
        cls_prob = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_pred_net(fc7)

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def _image_to_head(self):
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

    def _init_modules(self):
        self._init_head_tail()

        # rpn
        self.rpn_net = nn.Conv2d(
            self._net_conv_channels, cfg.RPN_CHANNELS, [3, 3], padding=1)

        self.rpn_cls_score_net = nn.Conv2d(cfg.RPN_CHANNELS,
                                           self._num_anchors * 2, [1, 1])

        self.rpn_bbox_pred_net = nn.Conv2d(cfg.RPN_CHANNELS,
                                           self._num_anchors * 4, [1, 1])

        self.cls_score_net = nn.Linear(self._fc7_channels, self._num_classes)
        self.bbox_pred_net = nn.Linear(self._fc7_channels, self._num_classes * 4)

        self.init_weights()

    def _run_summary_op(self, val=False, summary_size=1):
        """
            Run the summary operator: feed the placeholders with corresponding network outputs(activations)
        """
        summaries = []
        # Add image gt
        summaries.append(self._add_gt_image_summary())
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

            self._image_gt_summaries = {}
        else:
            for key, var in self._val_event_summaries.items():
                #print("adding validation summary for key {:s} with value {:f} and summary size divisor {:d}".format(key,var,summary_size))
                summaries.append(tb.summary.scalar(key, var/float(summary_size)))
            #Reset summary val
            self._val_event_summaries = {}

        return summaries

    def _predict(self):
        # This is just _build_network in tf-faster-rcnn
        torch.backends.cudnn.benchmark = False
        net_conv = self._image_to_head()
        #print(net_conv)
        # build the anchors for the image
        self._anchor_component(net_conv.size(2), net_conv.size(3))
        #print('run region proposal network')
        #numpy_out = net_conv.cpu().detach().numpy()[0, :, :, :]
        #print(numpy_out.shape)
        #for i in range(0,1000):
        #    numpy.savetxt('/home/mat/Thesis/train_net_conv_out_feature_{:d}_.txt'.format(i), numpy_out[i,:,:], delimiter=',')
        rois = self._region_proposal(net_conv)
        #print('_predict ROIs')
        #print(rois)
        if cfg.POOLING_MODE == 'align':
            pool5 = self._roi_align_layer(net_conv, rois)
        else:
            pool5 = self._roi_pool_layer(net_conv, rois)

        if self._mode == 'TRAIN':
            #Find best algo
            torch.backends.cudnn.benchmark = True  # benchmark because now the input size are fixed
        fc7 = self._head_to_tail(pool5)

        cls_prob, bbox_pred = self._region_classification(fc7)

        for k in self._predictions.keys():
            self._score_summaries[k] = self._predictions[k]

        return rois, cls_prob, bbox_pred

    def forward(self, image, im_info, gt_boxes=None, gt_boxes_dc=None, mode='TRAIN'):
        self._image_gt_summaries['image'] = image
        self._image_gt_summaries['gt_boxes'] = gt_boxes
        self._image_gt_summaries['gt_boxes_dc'] = gt_boxes_dc
        self._image_gt_summaries['im_info'] = im_info
        self._image = torch.from_numpy(image.transpose([0, 3, 1,
                                                        2])).to(self._device)
        self._im_info = im_info  # No need to change; actually it can be an list
        self._gt_boxes = torch.from_numpy(gt_boxes).to(
            self._device) if gt_boxes is not None else None
        self._gt_boxes_dc = torch.from_numpy(gt_boxes_dc).to(
            self._device) if gt_boxes is not None else None
        self._mode = mode
        if(mode == 'VAL'):
            self._mode = 'TRAIN'

        rois, cls_prob, bbox_pred = self._predict()
        if mode == 'TEST':
            #These are the deltas and they come out of the NN normalized, need to undo this
            stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(
                self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(
                self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            #Batch Norm?
            self._predictions["bbox_pred"] = bbox_pred.mul(stds).add(means)
        elif(mode == 'VAL'):
            self._add_losses()
            #????
            #Expand as -> broadcast 
            stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(
                self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(
                self._num_classes).unsqueeze(0).expand_as(bbox_pred)
            #Batch Norm?
            self._predictions["bbox_pred"] = bbox_pred.mul(stds).add(means)
        else:
            self._add_losses()  # compute losses

    def init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
      weight initalizer: truncated normal and random normal.
      """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.rpn_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn_bbox_pred_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.cls_score_net, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.bbox_pred_net, 0, 0.001, cfg.TRAIN.TRUNCATED)

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, image):
        feat = self._layers["head"](torch.from_numpy(
            image.transpose([0, 3, 1, 2])).to(self._device))
        return feat

    # only useful during testing mode
    def test_image(self, image, im_info):
        self.eval()
        with torch.no_grad():
            self.forward(image, im_info, None, None, mode='TEST')
        cls_score, cls_prob, bbox_pred, rois = self._predictions["cls_score"].data.cpu().numpy(), \
                                                         self._predictions['cls_prob'].data.cpu().numpy(), \
                                                         self._predictions['bbox_pred'].data.cpu().numpy(), \
                                                         self._predictions['rois'].data.cpu().numpy()
        return cls_score, cls_prob, bbox_pred, rois

    def delete_intermediate_states(self):
        # Delete intermediate result to save memory
        for d in [
                self._losses, self._predictions, self._anchor_targets,
                self._proposal_targets
        ]:
            for k in list(d):
                del d[k]
                
    #Eval summary required
    def run_eval(self, blobs, sum_size):
        #Flip between eval and train mode -> gradient doesnt accumulate?
        self.eval() # model.eval() will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
        self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_boxes_dc'], mode='VAL')
        self.train()
        bbox_predictions = self._predictions['bbox_pred'].data.cpu().numpy() #(self._fc7_channels, self._num_classes * 4)
        cls_prob        = self._predictions['cls_prob'].data.cpu().numpy() #(self._fc7_channels, self._num_classes)
        rois             = self._predictions['rois'].data.cpu().numpy()
        #For tensorboard
        for k in self._losses.keys():
            if(k in self._val_event_summaries):
                self._val_event_summaries[k] += self._losses[k]
            else:
                self._val_event_summaries[k] = self._losses[k]
        summary = self._run_summary_op(True,len(blobs['gt_boxes']))
        self.delete_intermediate_states()
        return summary, rois, bbox_predictions, cls_prob

    def train_step(self, blobs, train_op, update_weights=False):
        #Computes losses for single image
        self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_boxes_dc'])
        #.item() converts single element of type pytorch.tensor to a primitive float/int
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self._losses["rpn_cross_entropy"].item(), \
                                                                            self._losses['rpn_loss_box'].item(), \
                                                                            self._losses['cross_entropy'].item(), \
                                                                            self._losses['loss_box'].item(), \
                                                                            self._losses['total_loss'].item()
        #utils.timer.timer.tic('backward')
        self._losses['total_loss'].backward()
        #utils.timer.timer.toc('backward')
        self._cum_losses['total_loss']        += loss
        self._cum_losses['rpn_cross_entropy'] += rpn_loss_cls
        self._cum_losses['rpn_loss_box']      += rpn_loss_box
        self._cum_losses['cross_entropy']     += loss_cls
        self._cum_losses['loss_box']          += loss_box
        self._batch_gt_entries                += len(blobs['gt_boxes'])
        if(update_weights):
            #print('updating weights to end batch')
            #print(self._cum_losses['total_loss'])
            train_op.step()
            train_op.zero_grad()
            for k in self._cum_losses.keys():
                if(k in self._event_summaries):
                    self._event_summaries[k] += self._cum_losses[k]
                else:
                    self._event_summaries[k] = self._cum_losses[k]
            self._cum_gt_entries                  += self._batch_gt_entries
            #print('num GT boxes')
            #print(self._cum_gt_entries)
            self._cum_losses['total_loss']        = 0
            self._cum_losses['rpn_cross_entropy'] = 0
            self._cum_losses['rpn_loss_box']      = 0
            self._cum_losses['cross_entropy']     = 0
            self._cum_losses['loss_box']          = 0
            self._batch_gt_entries                = 0
        self.delete_intermediate_states()

        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

    def train_step_with_summary(self, blobs, train_op, sum_size, update_weights=False):
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss = self.train_step(blobs, train_op, update_weights)
        summary = self._run_summary_op(False, self._cum_gt_entries)
        self._cum_gt_entries = 0
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, summary

    def train_step_no_return(self, blobs, train_op):
        self.forward(blobs['data'], blobs['im_info'], blobs['gt_boxes'], blobs['gt_boxes_dc'])
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
