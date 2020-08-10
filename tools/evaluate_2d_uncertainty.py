import os
import math
import numpy as np
import itertools
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from enum import Enum
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import scipy.stats as scipy_stats
mypath = '/home/mat/thesis/data2/waymo'
date = 'aug02'
detection_file = os.path.join(mypath,'uncertainty_output',date,'image.txt')
gt_file        = os.path.join(mypath,'val','labels','image_labels.json')
#column_names = ['assoc_frame','scene_idx','frame_idx','bbdet','a_cls_var','a_cls_entropy','a_cls_mutual_info','e_cls_entropy','e_cls_mutual_info','a_bbox_var','e_bbox_var','track_idx','difficulty','pts','cls_idx','bbgt']
num_scenes = 210

def parse_dets(det_file):
    data_rows     = []
    column_names  = ['ID']
    track_en      = False
    int_en        = False
    skip_cols     = 0
    for i, line in enumerate(det_file):
        line = line.replace('bbdet', ' bbdet')
        line = line.replace('\n','').split(' ')
        row = []
        row.append(i)
        for j,elmnt in enumerate(line):
            if(skip_cols == 0):
                if(isfloat(elmnt)):
                    if(int_en):
                        row.append(int(elmnt))
                    else:
                        row.append(float(elmnt))
                else:
                    col = elmnt.replace(':','')
                    #Override for track idx as it has characters, override to save as integer when needed
                    int_en   = False
                    if('track' in col):
                        row.append(line[j+1])
                        skip_cols = 1
                    elif('idx' in col or 'pts' in col or 'difficulty' in col):
                        int_en = True
                    elif('cls_var' in col):
                        row.append([float(line[j+1]),float(line[j+2])])
                        skip_cols = 2
                    elif('bb' in col and '3d' not in col):
                        row.append([float(line[j+1]),float(line[j+2]),float(line[j+3]),float(line[j+4])])
                        skip_cols = 4
                    elif('bb' in col and '3d' in col):
                        row.append([float(line[j+1]),float(line[j+2]),float(line[j+3]),float(line[j+4]),float(line[j+5]),float(line[j+6]),float(line[j+7])])
                        skip_cols = 7
                    if(col not in column_names and col != ''):
                        column_names.append(col)
            else:
                skip_cols = skip_cols - 1
        data_rows.append(row)
    df = pd.DataFrame(data_rows,columns=column_names)
    df.set_index('ID')
    return df

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False


def parse_labels(dets_df, gt_file):
    with open(gt_file,'r') as labels_file:
        labels   = json.loads(labels_file.read())
    scene_name_dict = {}
    weather_dict    = {}
    tod_dict        = {}
    for label in labels:
        scene_type = label['scene_type'][0]
        tod = scene_type['tod']
        weather = scene_type['weather']
        scene_name = label['scene_name']
        assoc_frame  = label['assoc_frame']
        scene_idx  = int(int(label['assoc_frame'])/1000)

        if(scene_idx not in scene_name_dict.keys()):
            scene_name_dict[scene_idx] = scene_name
            weather_dict[scene_idx]    = weather
            tod_dict[scene_idx]        = tod
    full_df = dets_df
    full_df['scene_name'] = full_df['scene_idx'].map(scene_name_dict)
    full_df['weather'] = full_df['scene_idx'].map(weather_dict)
    full_df['tod'] = full_df['scene_idx'].map(tod_dict)
    return full_df

'''
monte carlo sampler 3D to 2D xyxy
takes a set of 3D bounding boxes and uncertainty input, samples the distribution 10000 times and then transforms into a new domain.
uncertainty is re-extracted after domain shift
'''

'''
plot scatter uncertainty 2D
Should be able to plot uncertainty (bbox or classification) defined in argument list as a function of a tuple pair of ordinal names.
'''

'''
plot scatter uncertainty
Should be able to plot uncertainty (bbox or classification) defined in argument list as a function of one ordinal, but multiple plots are allowed on one graph if multiple ordinal_names are specified
'''


'''
draw_combined_labels
Should read in an image based on the assoc_frame value of the labels, transform the lidar bbox to 8 pt image domain and draw, along with 2d image labels
'''

'''
associate_detections
combine both image and lidar result set to create one large panda db.
Detections without a corresponding ground truth are attempted to be matched with remaining bboxes via IoU
Detections that cannot be matched are removed
Additional columns to be added will be the transformed xyxy uncertainty values as well as the transformed 2D bbox.
'''

'''
plot scatter uncertainty per track
Same as above, but connect dots for individual track ID's. Ideally use a rainbow heatmap based on avg uncertainty
'''

#def plot_scatter_uc(dets,uc_type,scene_type,ordinal_names,max_val)


def plot_histo_cls_uc(dets,scene,min_val,max_val):
    #ax = dets.plot.hist(column='a_bbox_var',bins=12,alpha=0.5)
    bboxes = dets.columns
    for column in bboxes:
        if('a_entropy' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()
            conf_list = dets['confidence'].to_list()
            hist_data = []
            for i,var_line in enumerate(data):
                variance = var_line
                #conf = dets['confidence'][i]
                conf = conf_list[i]
                #variance = variance/conf
                hist_data.append(variance)
            #max_val = max(hist_data)
            #min_val = min(hist_data)
            #mean    = np.mean(hist_data)
            #hist_data = (hist_data-min_val)/(max_val-min_val)
            plt.hist(hist_data,bins=50,range=[min_val,max_val],alpha=0.3,label=labelname,density=True,stacked=True)
    #bboxes = bboxes.to_dict(orient="list")
    return None

def plot_histo_bbox_uc(dets,scene,min_val,max_val):
    #ax = dets.plot.hist(column='a_bbox_var',bins=12,alpha=0.5)
    bboxes = dets.filter(like='bb').columns
    for column in bboxes:
        if('e_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()
            bbgt = dets['bbgt'].to_list()
            bbdet = dets['bbdet'].to_list()
            hist_data = []
            data = np.asarray(data)
            data = np.sum(data,axis=1)
            bbdet = np.asarray(bbdet)
            bbox_area = (bbdet[:,2]-bbdet[:,0])*(bbdet[:,3]-bbdet[:,1]) + 1
            #print(len(bbox_area))
            #data = data/bbox_area
            #data = data/bbox_area
            #for i, bbox_var in enumerate(data):
            #
            #    bbox_area = (bbdet[i][2]-bbdet[i][0])*(bbdet[i][3]-bbdet[i][1]) + 1
            #    variance = sum(bbox_var)
            #    #variance = sum(bbox_var)/4
            #    hist_data.append(variance)
            #max_val = max(hist_data)
            #min_val = min(hist_data)
            #mean    = np.mean(hist_data)
            #hist_data = (hist_data-min_val)/(max_val-min_val)
            plt.hist(data,bins=200,range=[min_val,max_val],alpha=0.5,label=labelname,density=True,stacked=True)
    #bboxes = bboxes.to_dict(orient="list")
    return data

if __name__ == '__main__':
    with open(detection_file) as det_file:
        dets_df  = parse_dets(det_file.readlines())
    df  = parse_labels(dets_df, gt_file)
    print(df)
    #df  = df.loc[df['difficulty'] != -1]
    #df   = df.loc[df['confidence'] > 0.9]
    night_dets = df.loc[df['tod'] == 'Night']
    day_dets = df.loc[df['tod'] == 'Day']
    rain_dets = df.loc[df['weather'] == 'rain']
    sun_dets = df.loc[df['weather'] == 'sunny']
    scene_dets = df.loc[df['scene_idx'] == 168]
    diff1_dets = df.loc[df['difficulty'] != 2]
    diff2_dets = df.loc[df['difficulty'] == 2]
    minm = -10
    maxm = 10
    #scene_data = plot_histo_bbox_uc(scene_dets,'scene',minm,maxm)
    night_data = plot_histo_bbox_uc(night_dets,'night',minm,maxm)
    day_data   = plot_histo_bbox_uc(day_dets,'day',minm,maxm)
    #day_mean = np.mean(day_dets)
    #day_mean = np.mean(day_data)
    #print(len(night_data))
    #print(len(day_data))
    #r = scipy_stats.poisson.rvs(day_mean)
    #result = scipy_stats.ks_2samp(day_data,scene_data)
    #print(result)
    #plot_histo_bbox_uc(rain_dets,'rain',minm,maxm)
    #plot_histo_cls_uc(night_dets,'night',minm,maxm)
    #plot_histo_cls_uc(day_dets,'day',minm,maxm)
    #plot_histo_cls_uc(rain_dets,'rain',minm,maxm)
    #plot_histo_cls_uc(sun_dets,'sunny',minm,maxm)
    #plot_histo_bbox_uc(diff2_dets,'lvl2',minm,maxm)
    #plot_histo_bbox_uc(diff1_dets,'lvl1',minm,maxm)
    plt.legend()
    plt.show()
    #print(day_dets)
    #print(night_dets)
    #print(rain_dets)
