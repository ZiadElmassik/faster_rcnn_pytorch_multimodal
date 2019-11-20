# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
import PIL


def prepare_roidb(mode,imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """
    if(mode == 'train'):
        roidb = imdb.roidb
        idx = len(imdb.roidb)
    elif(mode == 'test'):
        roidb = imdb.test_roidb
        idx = imdb._num_test_images
    elif(mode == 'val'):
        roidb = imdb.val_roidb
        idx = len(imdb.val_roidb)
    if(imdb.name == 'kitti'):
        sizes = [
            PIL.Image.open(imdb.image_path_at(i)).size
            for i in range(imdb.num_images)
        ]
    elif(imdb.name == 'nuscenes'):
        #Bad practice, oh well.
        #sizes = np.empty([img_idx,2])
        sizes = np.full([idx,2],[imdb._imwidth,imdb._imheight])
        #sizes[:][1] = np.full(img_idx,imdb._imheight)
    elif(imdb.name == 'waymo'):
        sizes = np.full([idx,2],[imdb._imwidth,imdb._imheight])
    #Loop thru all images
    print('index size {:d}'.format(idx))
    for i in range(idx):
        #Store image path
        if(imdb.name == 'kitti'):
            roidb[i]['imagefile'] = imdb.image_path_at(i)

        #print('Preparing ROI\'s for image {:s} '.format(roidb[i]['imagefile']))
        #store weidth and height of entire image (why?)
        roidb[i]['width'] = sizes[i][0]
        roidb[i]['height'] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        #print(roidb[i]['max_overlaps'])
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)
