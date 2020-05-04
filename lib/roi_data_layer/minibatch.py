# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np
import numpy.random as npr
import cv2
import imgaug as ia
from copy import deepcopy
import imgaug.augmenters as iaa
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob, prep_bev_map_for_blob, bev_map_list_to_blob
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw, ImageEnhance
import os
from pyntcloud import PyntCloud
from scipy.ndimage.filters import gaussian_filter
import spconv
import utils.bbox as bbox_utils
import shutil
import re
from datasets.waymo_lidb import waymo_lidb

def draw_and_save_image_minibatch(blobs,cnt):

    datapath = os.path.join(cfg.ROOT_DIR, 'debug')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    #datapath = os.path.join(cfg.DATA_DIR, 'waymo','tmp_drawn')
    out_file = os.path.basename(blobs['filename'])
    out_file = os.path.join(datapath,out_file)
    img = blobs['data'][0]*cfg.PIXEL_STDDEVS + cfg.PIXEL_MEANS
    img = img.astype(dtype='uint8')
    source_img = Image.fromarray(img)
    draw = ImageDraw.Draw(source_img)
    for det in blobs['gt_boxes']:
        color = int(255*det[4])
        draw.rectangle([(det[0],det[1]),(det[2],det[3])],outline=(color,color,color))
    print('Saving file at location {}'.format(out_file))
    source_img.save(out_file,'PNG')  

def draw_and_save_lidar_minibatch(blob,cnt):
    filename = blob['filename']
    info = blob['info']
    scale = info[6]
    lidb = waymo_lidb()
    #Extract voxel grid size
    #width   = int(info[1] - info[0])
    #Y is along height axis in image domain
    #height  = int(info[3] - info[2])
    #lidb._imheight = height
    #lidb._imwidth  = width
    datapath = os.path.join(cfg.ROOT_DIR, 'debug')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    #out_file = filename.replace('/point_clouds/','/minibatch_drawn/').replace('.{}'.format('npy'),'.{}'.format('png'))
    filenum = re.sub('[^0-9]','',(os.path.basename(filename)))
    out_file  = os.path.join(datapath,'{}_{}_minibatch.png'.format(cnt,filenum))
    #source_bin = np.load(filename)

    #draw_file  = Image.new('RGB', (width, height), (255,255,255))
    #draw = ImageDraw.Draw(draw_file)
    #lidb.draw_bev(source_bin,draw)
    #for bbox in blob['gt_boxes']:
    #    lidb.draw_bev_bbox(draw,bbox,transform=False)
    #draw_file.save(out_file.replace('.png','_bev.png'),'png')
    voxel_grid = blob['data'][0]
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
    if(blob['flipped'] is True):
        draw.text((0,0),'flipped')
    else:
        draw.text((0,0),'normal')
    for bbox in blob['gt_boxes']:
        #bbox[0:2] = bbox[0:2]*scale
        #bbox[3:5] = bbox[3:5]*scale
        lidb.draw_bev_bbox(draw,bbox,transform=False)
    #for bbox_dc in enumerate(blob['gt_boxes_dc']):
    #    lidb.draw_bev_bbox(draw,bbox_dc)
    print('Saving BEV map file at location {}'.format(out_file))
    img.save(out_file,'png')

def get_minibatch(roidb, num_classes, augment_en,cnt):
    num_frames = len(roidb)
    assert num_frames == 1, "Single batch only"
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_frames)
    assert(cfg.TRAIN.BATCH_SIZE % num_frames == 0), \
      'num_frames ({}) must divide BATCH_SIZE ({})'. \
      format(num_frames, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    scale = cfg.TRAIN.SCALES[random_scale_inds[0]]
    if(cfg.NET_TYPE == 'image'):
        return get_image_minibatch(roidb,num_classes,augment_en,scale,cnt)
    elif(cfg.NET_TYPE == 'lidar'):
        return get_lidar_minibatch(roidb,num_classes,augment_en,scale,cnt)
    else:
        print('getting minibatch failed. Invalid NET TYPE in cfg')
        return None

def get_lidar_minibatch(roidb, num_classes, augment_en, scale, cnt):
    """Given a roidb, construct a minibatch sampled from it."""

    gt_box_size = cfg.LIDAR.NUM_BBOX_ELEM + 1  #BBox + Cls
    #X1,Y1,Z1,X2,Y2,Z2
    area_extents = [cfg.LIDAR.X_RANGE[0],cfg.LIDAR.Y_RANGE[0],cfg.LIDAR.Z_RANGE[0],cfg.LIDAR.X_RANGE[1],cfg.LIDAR.Y_RANGE[1],cfg.LIDAR.Z_RANGE[1]]

    # Get the input lidar blob
    infos, pc_blob, local_roidb = _get_lidar_blob(roidb, area_extents, scale, augment_en)
    info = infos[0]
    roi_entry = local_roidb[0]
    #Create numpy array storage for bounding boxes (enforce type)
    gt_len  = roi_entry['boxes'].shape[0]
    dc_len  = roi_entry['boxes_dc'].shape[0]
    gt_boxes = np.empty((gt_len, gt_box_size), dtype=np.float32)
    gt_boxes_dc = np.empty((dc_len, gt_box_size), dtype=np.float32)

    #Contains point cloud tensor
    blobs = {'data': pc_blob}
    blobs['flipped'] = roi_entry['flipped']
    #assert len(pc_scales) == 1, "Single batch only"
    assert len(local_roidb) == 1, "Single batch only"

    # gt boxes: (xc, yc, zc, xd, yd, zd, theta, cls)
    gt_inds = np.where(roi_entry['ignore'] == 0)[0]
    blobs['filename'] = roi_entry['filename']
    #print(blobs['filename'])
    #TODO: Ground plane estimation and subtraction
    #Transform into voxel_grid form (flip y-axis, scale to image size (e.g. 800,700))
    gt_boxes[:, 0:-1] = bbox_utils.bbox_pc_to_voxel_grid(roi_entry['boxes'][gt_inds, :],area_extents,info)
    gt_boxes[:, 0:2] = gt_boxes[:, 0:2] * scale
    gt_boxes[:, 3:5] = gt_boxes[:, 3:5] * scale
    #shift gt_boxes to voxel domain
    bbox_labels = roi_entry['gt_classes'][gt_inds]
    gt_boxes[:, -1] = bbox_labels
    blobs['gt_boxes'] = gt_boxes
    #Do we include don't care areas, so we ignore certain ground truth boxes (Might be kitti only, even tho waymo has NLZ)
    if cfg.TRAIN.IGNORE_DC:
        gt_ind_dc = np.arange(dc_len)
        gt_boxes_dc[:, 0:-1] = roi_entry['boxes_dc'][gt_ind_dc, :]
        gt_boxes_dc[:, -1] = np.zeros(dc_len)
    #TODO: FIX
    #vg_boxes_dc = bbox_utils.bbox_pc_to_voxel_grid(gt_boxes_dc,area_extents,info)
    vg_boxes_dc = np.empty(0)
    blobs['gt_boxes_dc'] = vg_boxes_dc * scale
    blobs['info'] = np.array(np.hstack((info,scale)), dtype=np.float32)
    #blobs['info'] = np.array([pc_blob.shape[0], pc_blob.shape[1], pc_blob.shape[2]], dtype=np.float32)
    if(cfg.DEBUG.DRAW_MINIBATCH):
        draw_and_save_lidar_minibatch(blobs,cnt)
    if(len(blobs['gt_boxes']) == 0):
        #print('No GT boxes for augmented image. Skipping')
        return None

    return blobs

def get_image_minibatch(roidb, num_classes, augment_en, scale, cnt):
    """Given a roidb, construct a minibatch sampled from it."""

    infos, im_blob, local_roidb = _get_image_blob(roidb, scale, augment_en)

    #Only one frame per minibatch allowed
    info = infos[0]
    im_scale = info[6]
    roi_entry = local_roidb[0]

    blobs = {'data': im_blob}
    blobs['info'] = info


    # gt boxes: (x1, y1, x2, y2, cls)
    #gt_inds = np.where(local_roidb[0]['gt_classes'] != 0)[0]
    #print(local_roidb[0]['ignore'])
    gt_inds = np.where(roi_entry['ignore'] == 0)[0]
    dc_len  = roi_entry['boxes_dc'].shape[0]
    blobs['filename'] = roi_entry['filename']
    #print('from get_image_minibatch')
    #print(blobs['filename'])
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    #print('scaling gt boxes by {}'.format(im_scales[0]))
    gt_boxes[:, 0:4] = roi_entry['boxes'][gt_inds, :] * im_scale
    gt_boxes[:, 4] = roi_entry['gt_classes'][gt_inds]
    blobs['gt_boxes'] = gt_boxes
    gt_boxes_dc = np.empty((dc_len, 5), dtype=np.float32)
    if cfg.TRAIN.IGNORE_DC:
        gt_ind_dc = np.arange(dc_len)
        gt_boxes_dc[:, 0:4] = roi_entry['boxes_dc'][gt_ind_dc, :] * im_scale
        gt_boxes_dc[:, 4] = np.zeros(dc_len)
    blobs['gt_boxes_dc'] = gt_boxes_dc

    if(cfg.DEBUG.DRAW_MINIBATCH):
        draw_and_save_image_minibatch(blobs,cnt)
    #print('gt boxes')
    #assert(len(blobs['gt_boxes']) != 0), 'gt_boxes is empty for image {:s}'.format(roi_entry['filename'])
    if(len(blobs['gt_boxes']) == 0):
        #print('No GT boxes for augmented image. Skipping')
        return None
    #assert((item is False for item in roidb[0]['ignore']).any(), 'All GT boxes are set to ignore.')
    return blobs


def _get_lidar_blob(roidb, pc_extents, scale, augment_en=False,mode='train'):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    processed_frames = []
    num_frame = len(roidb)
    infos = []
    for i in range(num_frame):
        if(mode == 'test'):
            assert augment_en is False
            source_bin = roidb[i]
            local_roidb = None
        else:
            source_bin = np.load(roidb[i]['filename'])
            local_roidb = deepcopy(roidb)
        #np.random.shuffle(source_bin)
        if(augment_en):
            #print('augmenting image {}'.format(roidb[i]['filename']))
            #shape 0 -> height
            #shape 1 -> width
            flip_num = np.random.normal(1.0, 2.0)
            if(local_roidb[i]['flipped'] is True):
                print('something wrong has happened')
            if(flip_num > 1.0):
                #Flip source binary across Y plane
                source_bin[:,1]       = -source_bin[:,1]
                local_roidb[i]['flipped'] = True
                oldy_c = local_roidb[i]['boxes'][:, 1].copy()
                old_ry = local_roidb[i]['boxes'][:, 6].copy()
                y_mean = (cfg.LIDAR.Y_RANGE[0]+cfg.LIDAR.Y_RANGE[1])/2
                local_roidb[i]['boxes'][:, 1] = -(oldy_c-y_mean) + y_mean
                local_roidb[i]['boxes'][:, 6] = -old_ry
            else:
                local_roidb[i]['flipped'] = False
        #print(roidb[i]['filename'])
        #print('min z value: {}'.format(np.amin(source_bin[:,2])))
        voxel_len = cfg.LIDAR.VOXEL_LEN/scale
        num_x_voxel = int((cfg.LIDAR.X_RANGE[1] - cfg.LIDAR.X_RANGE[0])*(1/voxel_len))
        num_y_voxel = int((cfg.LIDAR.Y_RANGE[1] - cfg.LIDAR.Y_RANGE[0])*(1/voxel_len))
        num_z_voxel = int(cfg.LIDAR.NUM_SLICES)
        infos.append([0,num_x_voxel,0,num_y_voxel,0,num_z_voxel,scale])
        vertical_voxel_size = (cfg.LIDAR.Z_RANGE[1] - cfg.LIDAR.Z_RANGE[0])/(cfg.LIDAR.NUM_SLICES+0.0)

        #Shift up to have voxel grid be at bottom of pc_extents
        pc_extents[5] -= pc_extents[2]
        pc_extents[2] = 0
        assert vertical_voxel_size == cfg.LIDAR.VOXEL_HEIGHT
        voxel_generator = spconv.utils.VoxelGeneratorV2(
            voxel_size=[voxel_len, voxel_len, cfg.LIDAR.VOXEL_HEIGHT],
            point_cloud_range=pc_extents,
            max_num_points=cfg.LIDAR.MAX_PTS_PER_VOXEL,
            max_voxels=cfg.LIDAR.MAX_NUM_VOXEL
        )
        #Coords returns zyx format
        #Subtract min height, so (0m,6m) instead of (-3m,3m)
        source_bin[:,2] -= cfg.LIDAR.Z_RANGE[0]
        res = voxel_generator.generate(source_bin)
        voxels = res['voxels']
        coords = res['coordinates']
        num_points_per_voxel = res['num_points_per_voxel']
        #Generate empty numpy arra to be populated
        bev_map = np.zeros((int(num_x_voxel),int(num_y_voxel),(cfg.LIDAR.NUM_CHANNEL)),dtype=np.float32)
        #zyx to xyz
        coords[:,[2,1,0]] = coords[:,[0,1,2]]
        xy_coords = coords[:,0:2]
        #Voxel contains (x,y,z,intensity,elongation)   
        voxel_min_heights = (coords[:,2]/cfg.LIDAR.VOXEL_HEIGHT)
        voxel_min_heights = np.repeat(voxel_min_heights[:,np.newaxis], voxels.shape[1],axis=1)   
        voxel_heights = voxels[:,:,2]
        voxel_max_height = np.amax(voxel_heights, axis=1) - coords[:,2]*cfg.LIDAR.VOXEL_HEIGHT
        voxel_mh_mean    = np.mean(voxel_max_height)
        #voxel_min_height = np.amin(voxel_heights, axis=1)
        #print('min height of frame: {}'.format(voxel_min_height))
        voxel_intensity = np.sum(voxels[:,:,3], axis=1)/num_points_per_voxel
        voxel_elongation = np.sum(voxels[:,:,4], axis=1)/num_points_per_voxel
        voxel_density    = num_points_per_voxel/cfg.LIDAR.MAX_PTS_PER_VOXEL
        voxel_d_mean     = np.mean(voxel_density)
        #Scatter height slices into bev_map
        maxheight_tuple = tuple(zip(*coords))
        bev_map[maxheight_tuple] = voxel_max_height
        #Scatter intensity into bev_map
        intensity_loc = np.full((xy_coords.shape[0],1),cfg.LIDAR.NUM_SLICES)
        intensity_coords = np.hstack((xy_coords,intensity_loc))
        intensity_tuple = tuple(zip(*intensity_coords))
        tanh_intensity = np.tanh(voxel_intensity)
        tanh_i_mean    = np.mean(tanh_intensity)
        bev_map[intensity_tuple] = tanh_intensity

        #Scatter elongation into bev_map
        elongation_loc = np.full((xy_coords.shape[0],1),cfg.LIDAR.NUM_SLICES+1)
        elongation_coords = np.hstack((xy_coords,elongation_loc))
        elongation_tuple = tuple(zip(*elongation_coords))
        bev_map[elongation_tuple] = np.tanh(voxel_elongation)

        #Scatter density into bev_map
        density_loc       = np.full((xy_coords.shape[0],1),cfg.LIDAR.NUM_SLICES+2)
        density_coords    = np.hstack((xy_coords,density_loc))
        density_tuple     = tuple(zip(*density_coords))
        bev_map[density_tuple] = voxel_density
        #Transpose so Y(left-right)/X(front-back) is X(left-right)/Y(front-back)
        bev_map        = np.transpose(bev_map,axes=[1,0,2])
        #proc_bev_map = prep_bev_map_for_blob(bev_map, cfg.LIDAR.MEANS, cfg.LIDAR.STDDEVS, scale)
        processed_frames.append(bev_map)
    # Create a blob to hold the input images
    blob = bev_map_list_to_blob(processed_frames)

    return infos, blob, local_roidb

def _get_image_blob(roidb, im_scale, augment_en=False, mode='train'):
    """Builds an input blob from the images in the roidb at the specified
  scales.
  """
    num_images = len(roidb)
    processed_ims = []
    im_infos = []
    for i in range(num_images):
        if(mode == 'test'):
            assert augment_en is False
            im          = roidb[i]
            local_roidb = None
        else:
            im          = cv2.imread(roidb[i]['filename'])
            local_roidb = deepcopy(roidb)
        img_arr  = im
        mean     = 0
        sigma    = 2
        #scale


        if(augment_en):
            #print('augmenting image {}'.format(roidb[i]['filename']))
            #shape 0 -> height
            #shape 1 -> width
            flip_num = np.random.normal(1.0, 2.0)
            if(local_roidb[i]['flipped'] is True):
                print('something wrong has happened')
            if(flip_num > 1.0):
                img_arr = img_arr[:, ::-1, :]
                local_roidb[i]['flipped'] = True
                oldx1 = local_roidb[i]['boxes'][:, 0].copy()
                oldx2 = local_roidb[i]['boxes'][:, 2].copy()
                local_roidb[i]['boxes'][:, 0] = im.shape[1] - oldx2 - 1
                local_roidb[i]['boxes'][:, 2] = im.shape[1] - oldx1 - 1
            else:
                local_roidb[i]['flipped'] = False
            #iaa.Sometimes(0.5,(iaa.CropAndPad(
            #    percent=(0, 0.1),
            #    pad_mode='constant',
            #    pad_cval=(0, 255),
            #    keep_size=True
            #))),
            seq = iaa.Sequential(
                [
                    iaa.Sometimes(0.6,(iaa.Affine(
                        scale={"x": (1, 1.5), "y": (1, 1.5)},  # scale images to 80-120% of their size, individually per axis
                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},  # translate by -20 to +20 percent (per axis)
                        order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                        shear=(-0.1, 0.1),
                        mode='constant'  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    ))),
                    #iaa.Sometimes(0.5,iaa.Dropout((0.01, 0.1), per_channel=0.5)),
                    #iaa.SomeOf((0,1),[
                    #iaa.Sometimes(0.5,iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                    #    iaa.Invert(0.05, per_channel=True)
                    #]),
                    #iaa.OneOf([
                    #iaa.Sometimes(0.5,iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                    #    iaa.PiecewiseAffine(scale=(0.01, 0.05))
                    #]),
                    iaa.SomeOf((0, 2),[
                        iaa.SomeOf((0,3),([
                            iaa.GaussianBlur((0.1, 2.0)),  # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(1, 3)),  # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(1, 3)),  # blur image using local medians with kernel sizes between 2 and 7
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
                        ])),
                        iaa.AdditiveGaussianNoise(
                            loc=0,
                            scale=(0.0, 0.08*255),
                            per_channel=0.5),
                        iaa.AddToHueAndSaturation((-10, 10)),  # change hue and saturation
                    ], random_order=True)
                ], random_order=False
            )
            images_aug, bboxes_aug = seq(images=[img_arr],bounding_boxes=[local_roidb[i]['boxes']])
            img_arr = images_aug[0]
            local_roidb[i]['boxes'] = bboxes_aug[0]
            orig_height = img_arr.shape[0]
            orig_width  = img_arr.shape[1]
            img_arr     = np.minimum(img_arr,255)
            img_arr     = np.maximum(img_arr,0)
            img_arr     = img_arr.astype('uint8')
            #if(down_shift < 0):
            #    img_arr = np.pad(img_arr,((0,abs(down_shift)), (0,0), (0,0)), mode='constant',constant_values=(127))[abs(down_shift):,:,:]
            #elif(down_shift > 0):
            #    img_arr = np.pad(img_arr,((abs(down_shift),0), (0,0), (0,0)), mode='constant',constant_values=(127))[:-down_shift,:,:]
            #if(right_shift < 0):
            #    img_arr = np.pad(img_arr,((0,0), (0,abs(right_shift)), (0,0)), mode='constant',constant_values=(127))[:,abs(right_shift):,:]
            #elif(right_shift > 0):
            #    img_arr = np.pad(img_arr,((0,0), (abs(right_shift),0), (0,0)), mode='constant',constant_values=(127))[:,:-right_shift,:]
            for j, roi in enumerate(local_roidb[i]['boxes']):
                #boxes[ix, :] = [x1, y1, x2, y2]
                orig = roi
                roi[0] = np.minimum(np.maximum(roi[0],0),orig_width-1)
                roi[2] = np.minimum(np.maximum(roi[2],0),orig_width-1)
                roi[1] = np.minimum(np.maximum(roi[1],0),orig_height-1)
                roi[3] = np.minimum(np.maximum(roi[3],0),orig_height-1)
                #TODO: magic number
                if(roi[3] - roi[1] < 12 and (roi[3] >= img_arr.shape[0]-1 or roi[1] <= 0)):
                    #print('removing box y0 {} y1 {}'.format(roi[1],roi[3]))
                    local_roidb[i]['ignore'][j] = True
                #TODO: magic number
                if(roi[2] - roi[0] < 12 and (roi[2] >= img_arr.shape[1]-1 or roi[0] <= 0)):
                    #print('removing box  x0 {} x1 {}'.format(roi[0],roi[2]))
                    local_roidb[i]['ignore'][j] = True

                w = roi[2] - roi[0]
                h = roi[3] - roi[1]
                if(h < 0.1):
                    local_roidb[i]['ignore'][j] = True
                elif(w < 0.1):
                    local_roidb[i]['ignore'][j] = True
                elif(h/w > 3.5 or w/h > 5.0):
                    local_roidb[i]['ignore'][j] = True

                if(local_roidb[i]['ignore'][j] is False and roi[2] < roi[0]):
                    print('x2 is smaller than x1')
                if(local_roidb[i]['ignore'][j] is False and roi[3] < roi[1]):
                    print('y2 is smaller than y1')
                if(local_roidb[i]['ignore'][j] is False and roi[2] - roi[0] > orig[2] - orig[0]):
                    print('new x is larger than old x diff')
                if(local_roidb[i]['ignore'][j] is False and roi[3] - roi[1] > orig[3] - orig[1]):
                    print('new y diff is larger than old y diff')
            im = img_arr
        #draw_and_save_minibatch(im,local_roidb[0])
        #draw_and_save_minibatch(im[:,:,cfg.PIXEL_ARRANGE_BGR],roidb[i])
        #TODO: Move scaling to be before imgaug, to save time
        im = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.PIXEL_STDDEVS, cfg.PIXEL_ARRANGE, im_scale)
            #x_min, x_max, y_min, y_max, scale
        info = np.array([0, im.shape[1], 0, im.shape[0], 0, 0, im_scale], dtype=np.float32)
        im_infos.append(info)
        processed_ims.append(im)
    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return im_infos, blob, local_roidb
