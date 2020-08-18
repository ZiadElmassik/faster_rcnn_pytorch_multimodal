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
import scipy.stats as scipy_stats
mypath = '/home/mat/thesis/data2/waymo'
date = 'aug11'
detection_file = os.path.join(mypath,'uncertainty_output',date,'image_pow2.txt')
#detection_file_2 = os.path.join(mypath,'uncertainty_output',date,'image_dropout_p_0_2.txt')
#detection_file_3 = os.path.join(mypath,'uncertainty_output',date,'image_dropout_p_0_4.txt')
gt_file        = os.path.join(mypath,'val','labels','image_labels.json')
#column_names = ['assoc_frame','scene_idx','frame_idx','bbdet','a_cls_var','a_cls_entropy','a_cls_mutual_info','e_cls_entropy','e_cls_mutual_info','a_bbox_var','e_bbox_var','track_idx','difficulty','pts','cls_idx','bbgt']
num_scenes = 210
mode_2d = True
def parse_dets(det_file):
    data_rows     = []
    column_names  = ['ID']
    track_en      = False
    int_en        = False
    skip_cols     = 0
    for i, line in enumerate(det_file):
        #line = line.replace('bbdet', ' bbdet')
        #line = line.replace('bbgt: -1 -1 -1 -1 -1 -1 -1', '')
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
                    elif('bb' in col and '3d' in col):
                        row.append([float(line[j+1]),float(line[j+2]),float(line[j+3]),float(line[j+4]),float(line[j+5]),float(line[j+6]),float(line[j+7])])
                        skip_cols = 7
                    elif('bb' in col):
                        if(mode_2d):
                            row.append([float(line[j+1]),float(line[j+2]),float(line[j+3]),float(line[j+4])])
                            skip_cols = 4
                        else:
                            row.append([float(line[j+1]),float(line[j+2]),float(line[j+3]),float(line[j+4]),float(line[j+5]),float(line[j+6]),float(line[j+7])])
                            skip_cols = 7  
                    if(col not in column_names and col != ''):
                        #print(col)
                        column_names.append(col)
            else:
                skip_cols = skip_cols - 1
        data_rows.append(row)
        #print(column_names)
        #print(row)
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


def plot_histo_cls_uc(dets,scene,min_val,max_val,uc_type='a_cls_var',cls_sel=None,fit=False):
    #ax = dets.plot.hist(column='a_bbox_var',bins=12,alpha=0.5)
    bboxes = dets.columns
    for column in bboxes:
        if(uc_type in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()
            e_entropy = dets['e_entropy'].to_list()
            conf_list = dets['confidence'].to_list()
            hist_data = []
            data_arr = np.asarray(data)
            #if('a_cls_var' in column):
                #data_arr = np.power(data_arr,2)
                #data_arr  = np.exp(data_arr)
            #data_mean = np.mean(data_arr[:,1])
            #data_var  = np.var(data_arr[:,0])
            data_max  = data_arr.max(axis=0)
            #data_arr  = data_arr/data_max
            data_arr  = data_arr
            #print(data_mean)
            variance = data_arr
            if('cls_var' in column):
                if(cls_sel is None):
                    variance = np.sum(variance,axis=1)
                else:
                    variance = variance[:,cls_sel]
            #max_val = max(hist_data)
            #min_val = min(hist_data)
            #mean    = np.mean(hist_data)
            #hist_data = (hist_data-min_val)/(max_val-min_val)
            #variance = conf_list/variance
            if(fit):
                x = np.arange(min_val,max_val,.0001)
                shape,loc,scale = scipy_stats.invgamma.fit(variance)
                g1 = scipy_stats.invgamma.pdf(x=x, a=shape, loc=loc, scale=scale)
                plt.plot(x,g1,label='{} fitted_gamma: {:.3f} {:.3f} {:.3f}'.format(scene,shape,loc,scale))
            plt.hist(variance,bins=200,range=[min_val,max_val],alpha=0.3,label=labelname,density=True,stacked=True)
            #(a,b,scale,loc) = scipy_stats.beta.fit(hist_data)
            #x = np.linspace(scipy_stats.beta.ppf(0.01, a, b), scipy_stats.beta.ppf(0.99, a, b), 100)
            #plt.scatter(x,scipy_stats.beta.pdf(x, a,b))
    #bboxes = bboxes.to_dict(orient="list")
    return None

def plot_histo_bbox_uc(dets,scene,min_val,max_val,fit=False):
    #ax = dets.plot.hist(column='a_bbox_var',bins=12,alpha=0.5)
    bboxes = dets.filter(like='bb').columns
    for column in bboxes:
        if('e_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()
            #bbgt = dets['bbgt'].to_list()
            #bbdet = dets['bbdet'].to_list()
            hist_data = []
            data_arr = np.asarray(data)
            data_mean = np.mean(data_arr,axis=0)
            data_var  = np.var(data_arr,axis=0)
            data_max  = data_arr.max(axis=0)
            #print(data_mean)
            data_arr = data_arr/data_max
            #data_sum = np.sum(data,axis=1)
            if(fit):
                x = np.arange(min_val,max_val,.0001)
                shape,loc,scale = scipy_stats.invgamma.fit(data_arr)
                g1 = scipy_stats.invgamma.pdf(x=x, a=shape, loc=loc, scale=scale)
                plt.plot(x,g1,label='{} fitted_gamma: {:.3f} {:.3f} {:.3f}'.format(scene,shape,loc,scale))
            #for i, bbox_var in enumerate(data):

                #bbox_area = (bbdet[i][2]-bbdet[i][0])*(bbdet[i][3]-bbdet[i][1]) + 1
                #bbox_var = (bbox_var - data_mean)
                #variance  = np.log(np.prod(bbox_var))
                #variance  = np.exp(bbox_var)
                #variance = sum(bbox_var)
                #variance = sum(bbox_var)/4
                #hist_data.append(variance)
            #max_val = max(hist_data)
            #min_val = min(hist_data)
            #mean    = np.mean(hist_data)
            #hist_data = (hist_data-min_val)/(max_val-min_val)
            plt.hist(data_arr,bins=300,range=[min_val,max_val],alpha=0.5,label=labelname,density=True,stacked=True)
    #bboxes = bboxes.to_dict(orient="list")
    return data_arr

if __name__ == '__main__':
    with open(detection_file) as det_file:
        dets_df  = parse_dets(det_file.readlines())
    df  = parse_labels(dets_df, gt_file)

    #with open(detection_file_2) as det_file:
    #    dets_df_2  = parse_dets(det_file.readlines())
    #df_2  = parse_labels(dets_df_2, gt_file)

    #with open(detection_file_3) as det_file:
    #    dets_df_3  = parse_dets(det_file.readlines())
    #df_3  = parse_labels(dets_df_3, gt_file)
    df  = df.loc[df['confidence'] > 0.5]
    df_n = df.loc[df['tod'] == 'Night']
    df_d = df.loc[df['tod'] == 'Day']
    df_s = df.loc[df['weather'] == 'sunny']
    df_r = df.loc[df['weather'] == 'rain']
    #df   = df.loc[df['tod'] == 'Day']
    df2  = df.loc[df['difficulty'] != -1]
    df3  = df.loc[df['difficulty'] == -1]
    df4  = df_n.loc[df_n['difficulty'] != -1]
    df5  = df_n.loc[df_n['difficulty'] == -1]
    df_s_1  = df_s.loc[df_s['difficulty'] != -1]
    df_s_2  = df_s.loc[df_s['difficulty'] == -1]
    df_r_1  = df_r.loc[df_r['difficulty'] != -1]
    df_r_2  = df_r.loc[df_r['difficulty'] == -1]
    df_d_1  = df_d.loc[df_d['difficulty'] != -1]
    df_d_2  = df_d.loc[df_d['difficulty'] == -1]
    df_n_1  = df_n.loc[df_n['difficulty'] != -1]
    df_n_2  = df_n.loc[df_n['difficulty'] == -1]
    #df2_2 = df_2.loc[df_2['difficulty'] != -1]
    #df2   = df.loc[df['confidence'] > 0.5]
    #df3   = df.loc[df['confidence'] <= 0.5]
    #day_dets = df.loc[df['tod'] == 'Day']
    #scene_dets = df.loc[df['scene_idx'] == 80]
    #rain_dets = df.loc[df['weather'] == 'rain']
    #sun_dets = df.loc[df['weather'] == 'sunny']
    #far_dets = df.loc[df['bbdet'][0] > 30]
    #near_dets = df.loc[df['bbdet'][0] <= 30]
    #diff1_dets = df.loc[df['difficulty'] != 2]
    #diff2_dets = df.loc[df['difficulty'] == 2]
    minm = 0.0
    maxm = 0.05
    #plot_histo_bbox_uc(night_dets,'night',minm,maxm)
    #plot_histo_bbox_uc(df2,'TP',minm,maxm,fit=False)
    #plot_histo_bbox_uc(df3,'FP',minm,maxm,fit=False)
    #plot_histo_bbox_uc(df_d_1,'day-TP',minm,maxm,fit=True)
    #plot_histo_bbox_uc(df_d_2,'day-FP',minm,maxm,fit=False)
    #plot_histo_bbox_uc(df_n_1,'night-TP',minm,maxm,fit=True)
    #plot_histo_bbox_uc(df_n_2,'night-FP',minm,maxm,fit=False)
    #plot_histo_bbox_uc(df_r_1,'Rain-TP',minm,maxm,fit=False)
    #plot_histo_bbox_uc(df_s_2,'Sun-FP',minm,maxm,fit=True)
    #plot_histo_bbox_uc(df_r_2,'Rain-FP',minm,maxm,fit=False)
    #plot_histo_bbox_uc(df_2,'dropout p=0.2',minm,maxm,fit=False)
    #plot_histo_bbox_uc(df_3,'dropout p=0.4',minm,maxm,fit=False)
    #day_data   = plot_histo_bbox_uc(df4,'tp-N',minm,maxm)
    #night_data = plot_histo_bbox_uc(df5,'fp-N',minm,maxm)
    #day_data = plot_histo_bbox_uc(day_dets,'day',minm,maxm)
    #print(len(night_data))
    #print(len(day_data))
    #result = scipy_stats.ks_2samp(day_data,night_data)
    #print(result)
    plot_histo_cls_uc(df_d_1,'tp-sun',minm,maxm,uc_type='a_cls_var',fit=False,cls_sel=1)
    plot_histo_cls_uc(df_d_2,'fp-sun',minm,maxm,uc_type='a_cls_var',fit=False,cls_sel=1)
    plot_histo_cls_uc(df_n_1,'tp-night',minm,maxm,uc_type='a_cls_var',fit=False,cls_sel=1)
    plot_histo_cls_uc(df_n_2,'fp-night',minm,maxm,uc_type='a_cls_var',fit=False,cls_sel=1)
    #plot_histo_cls_uc(df_2,'all-corrupted',minm,maxm,uc_type='e_entropy',cls_sel=1)
    #plot_histo_cls_uc(df3,'fp',minm,maxm,uc_type='a_cls_var',cls_sel=1)
    #plot_histo_cls_uc(rain_dets,'rain',minm,maxm,0)
    #plot_histo_cls_uc(sun_dets,'sun',minm,maxm,0)
    #plot_histo_cls_uc(df2,'tp-1',minm,maxm,1)
    #plot_histo_cls_uc(df3,'fp-1',minm,maxm,1)
    #plot_histo_cls_uc(scene_dets,'rain-1',minm,maxm,1)
    #plot_histo_cls_uc(day_dets,'day-1',minm,maxm)
    #plot_histo_cls_uc(df2,'tp',minm,maxm)
    #plot_histo_cls_uc(df3,'fp',minm,maxm)
    #plot_histo_bbox_uc(diff2_dets,'lvl2',minm,maxm)
    #plot_histo_bbox_uc(diff1_dets,'lvl1',minm,maxm)
    #plot_histo_bbox_uc(sun_dets,'total',minm,maxm)
    plt.legend()
    plt.show()
    #print(day_dets)
    #print(night_dets)
    #print(rain_dets)
