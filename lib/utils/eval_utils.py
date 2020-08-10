# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
from model.config import cfg
from shapely.geometry import Polygon
import pickle
import numpy as np
import utils.bbox as bbox_utils
from scipy.interpolate import InterpolatedUnivariateSpline
import sys
import operator
import json
import re
from scipy.spatial import ConvexHull

#Values    Name      Description
#----------------------------------------------------------------------------
#   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
#                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
#                     'Misc' or 'DontCare'
#   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
#                     truncated refers to the object leaving frame boundaries
#   1    occluded     Integer (0,1,2,3) indicating occlusion state:
#                     0 = fully visible, 1 = partly occluded
#                     2 = largely occluded, 3 = unknown
#   1    alpha        Observation angle of object, ranging [-pi..pi]
#   4    bbox         2D bounding box of object in the frame (0-based index):
#                     contains left, top, right, bottom pixel coordinates
#   3    dimensions   3D object dimensions: height, width, length (in meters)
#   3    location     3D object location x,y,z in camera coordinates (in meters)
#   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
#   1    score        Only for results: Float, indicating confidence in
#                     detection, needed for p/r curves, higher is better.


def parse_rec(filename):
    """ Parse the labels for the specific frame """
    label_lines = open(filename, 'r').readlines()
    objects = []
    for line in label_lines:
        label_arr = line.split(' ')
        obj_struct = {}
        obj_struct['name'] = label_arr[0]
        obj_struct['truncated'] = label_arr[1]
        obj_struct['occluded'] = label_arr[2]
        obj_struct['alpha'] = label_arr[3]
        obj_struct['bbox'] = [
            float(label_arr[4]),
            float(label_arr[5]),
            float(label_arr[6]),
            float(label_arr[7])
        ]
        obj_struct['3D_dim'] = [
            float(label_arr[8]),
            float(label_arr[9]),
            float(label_arr[10])
        ]
        obj_struct['3D_loc'] = [
            float(label_arr[11]),
            float(label_arr[12]),
            float(label_arr[13])
        ]
        obj_struct['rot_y'] = float(label_arr[14])
        objects.append(obj_struct)

    return objects


def get_labels_filename(db, eval_type):
    if(eval_type == 'bev' or eval_type == '3d' or eval_type == 'bev_aa'):
        labels_filename = 'lidar_labels.json'
    elif(eval_type == '2d'):
        labels_filename = 'image_labels.json'
    return labels_filename


def load_cached_annotations(cachedir, filename):
    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, '{}_{}_annots.pkl'.format(mode,classname))
    cachefile  = os.path.join(cachedir,filename)
    if not os.path.isfile(cachefile):
        return None
        # load annotations
        # save
        #print('Saving cached annotations to {:s}'.format(cachefile))
        #with open(cachefile, 'wb') as f:
        #    pickle.dump(class_recs, f)
    else:
        # load
        print('loading cached annotations from {:s}'.format(cachefile))
        with open(cachefile, 'rb') as f:
            try:
                class_recs = pickle.load(f)
            except:
                class_recs = pickle.load(f, encoding='bytes')
        return class_recs

def save_annotations_to_cache(cachedir, filename, class_recs):
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, '{}_{}_annots.pkl'.format(mode,classname))
    cachefile  = os.path.join(cachedir,filename)
    if not os.path.isfile(cachefile):
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(class_recs, f)
    else:
        print('annotations cache already exists')

def ap(rec, prec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
  Compute VOC AP given precision and recall.
  If use_07_metric is true, uses the
  VOC 07 11 point method (default:False).
  """
    #if use_07_metric:
        # 11 point metric
    #    ap = 0.
    #    for t in np.arange(0., 1.1, 0.1):
    #        if np.sum(rec >= t) == 0:
    #            p = 0
    #        else:
    #            p = np.max(prec[rec >= t])
    #        ap = ap + p / 11.
    #else:
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope (going backwards, precision will always increase as sorted by -confidence)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def extract_uncertainties(bbox_elem, splitlines):
    uc_avg        = {}
    uncertainties = {}
    u_start = bbox_elem + 3
    if(cfg.UC.EN_CLS_ALEATORIC):
        uc_avg['a_entropy']  = np.zeros((cfg.NUM_SCENES,1))
        uc_avg['a_mutual_info'] = np.zeros((cfg.NUM_SCENES,1))
        uc_avg['a_cls_var'] = np.zeros((cfg.NUM_SCENES,2))
        #uncertainties['a_cls_var'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        #u_start += 1
        uncertainties['a_entropy'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
        uncertainties['a_mutual_info'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
        uncertainties['a_cls_var'] = np.array([[float(z) for z in x[u_start:u_start+2]] for x in splitlines])
        u_start += 2
    if(cfg.UC.EN_CLS_EPISTEMIC):
        uc_avg['e_entropy'] = np.zeros((cfg.NUM_SCENES,1))
        uc_avg['e_mutual_info'] = np.zeros((cfg.NUM_SCENES,1))
        uc_avg['e_cls_var']     = np.zeros((cfg.NUM_SCENES,2))
        uncertainties['e_entropy'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
        uncertainties['e_mutual_info'] = np.array([[float(z) for z in x[u_start:u_start+1]] for x in splitlines])
        u_start += 1
        uncertainties['e_cls_var'] = np.array([[float(z) for z in x[u_start:u_start+2]] for x in splitlines])
        u_start += 2
    if(cfg.UC.EN_BBOX_ALEATORIC):
        uc_avg['a_bbox_var'] = np.zeros((cfg.NUM_SCENES,bbox_elem))
        uncertainties['a_bbox_var'] = np.array([[float(z) for z in x[u_start:u_start+bbox_elem]] for x in splitlines])
        u_start += bbox_elem
    if(cfg.UC.EN_BBOX_EPISTEMIC):
        uc_avg['e_bbox_var'] = np.zeros((cfg.NUM_SCENES,bbox_elem))
        uncertainties['e_bbox_var'] = np.array([[float(z) for z in x[u_start:u_start+bbox_elem]] for x in splitlines])
        u_start += bbox_elem
    assert (u_start == len(splitlines[0]))
    return uc_avg, uncertainties

def display_frame_counts(tp_frame,fp_frame,npos_frame):
    tp_frame = tp_frame[tp_frame != 0]
    fp_frame = fp_frame[fp_frame != 0]
    npos_frame = npos_frame[npos_frame != 0]
    tp_idx = tp_frame.nonzero()
    fp_idx = fp_frame.nonzero()
    npos_idx = npos_frame.nonzero()
    print('tp')
    print(tp_frame)
    print(tp_idx)
    print('fp')
    print(fp_frame)
    print(fp_idx)
    print('npos')
    print(npos_frame)
    print(npos_idx)

def write_frame_uncertainty(uc_avg,frame_dets,idx):
    print_str = ''
    print_start = 'Frame: {} \n num_dets: {}'.format(idx,frame_dets)
    if(frame_dets != 0):
        if(cfg.UC.EN_CLS_ALEATORIC):
            a_entropy     = uc_avg['a_entropy'][idx]/frame_dets
            a_mutual_info = uc_avg['a_mutual_info'][idx]/frame_dets
            a_cls_var     = uc_avg['a_cls_var'][idx]/frame_dets
            print_str    += ' a_entropy: {:4.3f} a_mutual_info: {:4.3f} a_cls_var: {:4.3f} '.format(a_entropy[0],a_mutual_info[0],a_cls_var[0])
        if(cfg.UC.EN_CLS_EPISTEMIC):
            e_entropy     = uc_avg['e_entropy'][idx]/frame_dets
            e_mutual_info = uc_avg['e_mutual_info'][idx]/frame_dets
            print_str    += ' e_entropy: {:4.3f} e_mutual_info: {:4.3f} '.format(e_entropy[0],e_mutual_info[0])
        if(cfg.UC.EN_BBOX_ALEATORIC):
            print_str += ' a_bbox: '
            for var_elem in uc_avg['a_bbox_var'][idx]/frame_dets:
                print_str += '{:4.3f} '.format(var_elem)
        if(cfg.UC.EN_BBOX_EPISTEMIC):
            print_str += ' e_bbox: '
            for var_elem in uc_avg['e_bbox_var'][idx]/frame_dets:
                print_str += '{:4.3f} '.format(var_elem)
    if(print_str != ''):
        print_start += '\n'
        print_start += print_str
        return print_start
    else:
        return print_str


def write_scene_uncertainty(uc_avg,scene_dets,idx):
    print_str = ''
    print_start = 'Scene: {} \n num_dets: {}'.format(idx,scene_dets)
    if(scene_dets != 0):
        if(cfg.UC.EN_CLS_ALEATORIC):
            a_entropy     = uc_avg['a_entropy'][idx]/scene_dets
            a_mutual_info = uc_avg['a_mutual_info'][idx]/scene_dets
            a_cls_var     = uc_avg['a_cls_var'][idx]/scene_dets
            print_str    += ' a_entropy: {:4.3f} a_mutual_info: {:4.3f} a_cls_var: {:4.3f} '.format(a_entropy[0],a_mutual_info[0],a_cls_var[0])
        if(cfg.UC.EN_CLS_EPISTEMIC):
            e_entropy     = uc_avg['e_entropy'][idx]/scene_dets
            e_mutual_info = uc_avg['e_mutual_info'][idx]/scene_dets
            print_str    += ' e_entropy: {:4.3f} e_mutual_info: {:4.3f} '.format(e_entropy[0],e_mutual_info[0])
        if(cfg.UC.EN_BBOX_ALEATORIC):
            print_str += ' a_bbox: '
            for var_elem in uc_avg['a_bbox_var'][idx]/scene_dets:
                print_str += '{:4.3f} '.format(var_elem)
        if(cfg.UC.EN_BBOX_EPISTEMIC):
            print_str += ' e_bbox: '
            for var_elem in uc_avg['e_bbox_var'][idx]/scene_dets:
                print_str += '{:4.3f} '.format(var_elem)
    if(print_str != ''):
        print_start += '\n'
        print_start += print_str
        return print_start
    else:
        return print_str

def find_rec(class_recs, token):
    for rec in class_recs:
        if(rec['idx'] == re.sub('[^0-9]','',token)):
            if(rec['ignore_frame'] is False):
                return rec
            else:
                return None
    return None

def save_detection_results(results,pathdir,filename):
    if not os.path.isdir(pathdir):
        print('Making detection output path {}'.format(pathdir))
        os.makedirs(pathdir)
    fp = os.path.join(pathdir,filename)
    file_ptr = open(fp,'w+')
    for result in results:
        if(result is not None and result != ''):
            file_ptr.write(result)
            file_ptr.write('\n')
    file_ptr.close()

def iou(bbgt,bbdet,eval_type):
    if(eval_type == '2d' or eval_type == 'bev_aa'):
        overlaps = eval_2d_iou(bbgt,bbdet)
    elif(eval_type == 'bev'):
        overlaps = eval_bev_iou(bbgt,bbdet)
    elif(eval_type == '3d'):
        overlaps = eval_3d_iou(bbgt,bbdet)
    else:
        overlaps = None
    return overlaps

"""
function: eval_2d_iou
Inputs:
bbgt: (N,4) ground truth boxes
bbdet: (4,) detection box
output:
overlaps: (N) total overlap for each bbgt with the bbdet
"""
def eval_2d_iou(bbgt,bbdet):
    # compute overlaps
    # intersection
    #bbgt = [xmin,ymin,xmax,ymax]
    ixmin = np.maximum(bbgt[:, 0], bbdet[0])
    iymin = np.maximum(bbgt[:, 1], bbdet[1])
    ixmax = np.minimum(bbgt[:, 2], bbdet[2])
    iymax = np.minimum(bbgt[:, 3], bbdet[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    #This is the intersection of both BB's
    inters = iw * ih

    # union
    uni = ((bbdet[2] - bbdet[0] + 1.) * (bbdet[3] - bbdet[1] + 1.) +
            (bbgt[:, 2] - bbgt[:, 0] + 1.) *
            (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)
    #IoU - intersection over union
    overlaps = inters / uni
    return overlaps


"""
function: eval_bev_iou
Inputs:
bbgt: (N,7) ground truth boxes
bbdet: (7,) detection box
output:
overlaps: (N) total overlap for each bbgt with the bbdet
"""
def eval_bev_iou(bbgt, bbdet):
    det_4pt = bbox_utils.bbox_3d_to_bev_4pt(bbdet[np.newaxis,:])[0]
    gts_4pt  = bbox_utils.bbox_3d_to_bev_4pt(bbgt)
    overlaps = np.zeros((bbgt.shape[0]))
    for i, gt_4pt in enumerate(gts_4pt):
        gt_poly = bbox_to_polygon_2d(gt_4pt)
        det_poly = bbox_to_polygon_2d(det_4pt)
        inter = gt_poly.intersection(det_poly).area
        union = gt_poly.union(det_poly).area
        iou_2d = inter/union
        overlaps[i] = iou_2d
    return overlaps

"""
function: eval_bev_iou
Inputs:
bbgt: (N,7) ground truth boxes
bbdet: (7,) detection box
output:
overlaps: (N) total overlap for each bbgt with the bbdet
"""
def eval_3d_iou(bbgt, bbdet):
    overlaps = np.zeros((bbgt.shape[0]))
    det_4pt = bbox_utils.bbox_3d_to_bev_4pt(bbdet[np.newaxis,:])[0]
    gts_4pt  = bbox_utils.bbox_3d_to_bev_4pt(bbgt)
    det_z    = [bbdet[2]-bbdet[5]/2,bbdet[2]+bbdet[5]/2]
    gt_z     = [bbgt[:,2]-bbgt[:,5]/2,bbgt[:,2]+bbgt[:,5]/2]
    det_height = bbdet[5]
    for i, gt_4pt in enumerate(gts_4pt):
        gt_height = bbgt[i,5]
        inter_max = min(gt_z[1][i],det_z[1]) 
        inter_min = max(gt_z[0][i],det_z[0])
        inter_height = max(0.0,inter_max - inter_min)
        gt_poly = bbox_to_polygon_2d(gt_4pt)
        det_poly = bbox_to_polygon_2d(det_4pt)
        inter = gt_poly.intersection(det_poly).area
        union = gt_poly.union(det_poly).area
        inter_vol = inter*inter_height
        if(inter_vol < 0):
            inter_vol = 0
        #Compute iou 3d by including heights, as height is axis aligned
        iou_3d = inter_vol/(gt_poly.area*gt_height + det_poly.area*det_height - inter_vol)
        overlaps[i] = iou_3d
    return overlaps

def bbox_to_polygon_2d(bbox):
    return Polygon([(bbox[0,0], bbox[0,1]), (bbox[1,0], bbox[1,1]), (bbox[2,0], bbox[2,1]), (bbox[3,0], bbox[3,1])])
