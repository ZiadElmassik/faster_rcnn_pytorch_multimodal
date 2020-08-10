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
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import halfnorm
mypath = '/home/mat/thesis/data2'
imgpath = os.path.join(mypath,'waymo/val/images')  # can join these later for completion
savepath = os.path.join(mypath,'2d_uncertainty_drawn')
#detection_file = os.path.join(mypath,'faster_rcnn_pytorch_multimodal','tools','results','vehicle.car_detection_results_simple.txt')
detection_file = os.path.join(mypath,'faster_rcnn_pytorch_multimodal','tools','results','lidar_3d_iou_uncertainty_results.txt')
gt_file        = os.path.join(mypath,'labels_full','combined_labels_new.json')
#column_names = ['assoc_frame','scene_idx','frame_idx','bbdet','a_cls_var','a_cls_entropy','a_cls_mutual_info','e_cls_entropy','e_cls_mutual_info','a_bbox_var','e_bbox_var','track_idx','difficulty','pts','cls_idx','bbgt']
num_scenes = 210
top_crop = 300
bot_crop = 30

def parse_dets(det_file):
    if ("3d" in detection_file):  # is the file from lidar or img domain
        lidar_flag = True
    else:
        lidar_flag = False
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
                    if (lidar_flag and col == 'bbgt3d'):
                        col = 'bbgt'
                    #Override for track idx as it has characters, override to save as integer when needed
                    int_en   = False
                    if('track' in col):
                        row.append(line[j+1])
                        skip_cols = 1
                    elif('idx' in col or 'pts' in col or 'difficulty' in col):
                        int_en = True
                    elif(lidar_flag and 'bb' in col):
                        row.append([float(line[j+1]),float(line[j+2]),float(line[j+3]),float(line[j+4]),float(line[j+5]),float(line[j+6]),float(line[j+7])])
                        skip_cols = 7
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
    calibration_dict = {}
    meta_dict = {}
    for label in labels:
        scene_type = label['scene_type'][0]
        tod = scene_type['tod']
        weather = scene_type['weather']
        scene_name = label['scene_name']
        assoc_frame  = label['assoc_frame']
        scene_idx  = int(int(label['assoc_frame'])/1000)
        calibration = label['calibration']  # for transforming 3d to 2d
        meta = label['meta']  # for transforming 3d to 2d

        if(scene_idx not in scene_name_dict.keys()):
            scene_name_dict[scene_idx] = scene_name
            weather_dict[scene_idx]    = weather
            tod_dict[scene_idx]        = tod
            calibration_dict[scene_idx] = calibration
            meta_dict[scene_idx] = meta
    full_df = dets_df
    full_df['scene_name'] = full_df['scene_idx'].map(scene_name_dict)
    full_df['weather'] = full_df['scene_idx'].map(weather_dict)
    full_df['tod'] = full_df['scene_idx'].map(tod_dict)
    full_df['calibration'] = full_df['scene_idx'].map(calibration_dict)
    full_df['meta'] = full_df['scene_idx'].map(meta_dict)
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
        if('a_mutual_info' in column):
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

def plot_histo_gaussian(df,dets,scene,min_val,max_val):
    bboxes = dets.filter(like='bb').columns
    a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
    a_bbox_var = np.sort(np.sum(a_bbox_var,axis=1))  # sum and sort
    a_bbox_var_std_dev = np.std(a_bbox_var)

    e_bbox_var = np.asarray(df['e_bbox_var'].to_list())
    e_bbox_var = np.sort(np.sum(e_bbox_var,axis=1))  # sum and sort
    e_bbox_var_std_dev = np.std(e_bbox_var)

    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
            #bbgt = dets['bbgt'].to_list()
            #bbdet = dets['bbdet'].to_list()
            #hist_data = []
            data = np.asarray(data)
            data = np.sum(data,axis=1)  # sum and sort
            range = np.max(data) - np.min(data)
            data_truncated = data[data < range*0.20]
            data_truncated = (data_truncated)/a_bbox_var_std_dev  # convert to z scores            
            data_truncated = np.log(data_truncated)  # to fit normal distribution, log data 
            (mu, sigma) = norm.fit(data_truncated) # fit data to normal distribution
            #l = plt.plot(200, y, 'r--', linewidth=2)

            x_range = np.arange(np.min(data_truncated),np.max(data_truncated),0.001)
            pdf = norm.pdf(x_range, mu, sigma)
            estimated_mu = float("{:.3f}".format(mu))
            estimated_sigma = float("{:.3f}".format(sigma))

            plt.plot(x_range,pdf,label='fitted_norm_pdf')
            plt.hist(data_truncated,bins=200,range=[np.min(data_truncated),np.max(data_truncated)+1],alpha=0.5,label=labelname,density=True,stacked=True)
            plt.text(1.5,1.2,r'$\mu=$''%s\n'r'$\sigma=$''%s'%(np.mean(data_truncated),estimated_sigma)) 
    return data, estimated_mu, estimated_sigma


def plot_histo_chi_squared(dets,scene,min_val,max_val):
    bboxes = dets.filter(like='bb').columns
    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
            bbgt = dets['bbgt'].to_list()
            bbdet = dets['bbdet'].to_list()
            hist_data = []
            data = np.asarray(data)
            data = np.sort(np.sum(data,axis=1))  # sum and sort
            range = np.max(data) - np.min(data)
            data_truncated = data[data < range*0.20]
            data_std = np.std(data_truncated)
            data_truncated = (data_truncated)/data_std  # convert to z scores
            df = 4  # degrees of freedom
            x_range = np.arange(np.min(data_truncated),np.max(data_truncated),0.001)

            s = .2

            # chi_square = np.random.chisquare(1000,len(data)*10000)
            # chi_norm = np.linalg.norm(chi_square)
            # chi_square = chi_square/chi_norm
            # bbdet = np.asarray(bbdet)
            # bbox_area = (bbdet[:,2]-bbdet[:,0])*(bbdet[:,3]-bbdet[:,1]) + 1
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
            #min_vdraw   = np.mean(hist_data)
            #hist_data = (hist_data-min_val)/(max_val-min_val)
    
            plt.hist(data_truncated,bins=200,range=[np.min(data_truncated),np.max(data_truncated)],alpha=0.5,label=labelname,density=True,stacked=True)
            plt.plot(x_range, lognorm.pdf(x_range, s,-0.8,1), alpha=0.6, label='lognorm pdf')
            #plt.plot(x_range, chi2.pdf(x_range, df), alpha=0.6, label='chi2 pdf')
            #plt.text(1.50,1.0,'mean= %s \n std_dev=%s '%(data_mean,data_std))
            #sns.distplot(data,label='fitting curve')
            #plt.hist(r,bins=200,range=[min_val,max_val],alpha=0.5,label='r',density=True,stacked=True)
    #bboxes = bboxes.to_dict(orient="list")
    return data

def plot_histo_lognorm(df,dets,scene,min_val,max_val):
    bboxes = dets.filter(like='bb').columns
    a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
    a_bbox_var = np.sort(np.sum(a_bbox_var,axis=1))  # sum and sort
    a_bbox_var_std_dev = np.std(a_bbox_var)

    e_bbox_var = np.asarray(df['e_bbox_var'].to_list())
    e_bbox_var = np.sort(np.sum(e_bbox_var,axis=1))  # sum and sort
    e_bbox_var_std_dev = np.std(e_bbox_var)

    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
            bbgt = dets['bbgt'].to_list()
            bbdet = dets['bbdet'].to_list()
            hist_data = []
            data = np.asarray(data)
            data = np.sort(np.sum(data,axis=1))  # sum and sort
            range = np.max(data) - np.min(data)
            data_truncated = data[data < range*0.20]
            data_truncated = (data_truncated)/e_bbox_var_std_dev  # convert to z scores            
            df = 4  # degrees of freedom
            x_range = np.arange(np.min(data_truncated),np.max(data_truncated),0.001)
            data_mean = np.mean(data_truncated)
            s,loc,scale = lognorm.fit(data_truncated,floc=0)
            estimated_mu = np.log(scale)
            estimated_sigma = s  # shape is std dev
            estimated_mu = float("{:.3f}".format(estimated_mu))
            estimated_sigma = float("{:.3f}".format(estimated_sigma))
            pdf = lognorm.pdf(x_range,s,scale=scale)
            plt.plot(x_range,pdf,label='fitted_lognorm_pdf')
            plt.hist(data_truncated,bins=200,range=[np.min(data_truncated),np.max(data_truncated)+1],alpha=0.5,label=labelname,density=True,stacked=True)
            plt.text(1.5,1.2,r'$\mu=$''%s\n'r'$\sigma=$''%s'%(np.mean(data_truncated),estimated_sigma)) 
    return data, estimated_mu, estimated_sigma


def plot_histo_half_gaussian(df,dets,scene,min_val,max_val):
    bboxes = dets.filter(like='bb').columns
    a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
    a_bbox_var = np.sort(np.sum(a_bbox_var,axis=1))  # sum and sort
    a_bbox_var_std_dev = np.std(a_bbox_var)
    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
            #bbgt = dets['bbgt'].to_list()
            #bbdet = dets['bbdet'].to_list()
            #hist_data = []
            data = np.asarray(data)
            data = np.sort(np.sum(data,axis=1))  # sum and sort
            range = np.max(data) - np.min(data)
            data_truncated = data[data < range*0.20]
            data_truncated = (data_truncated)/a_bbox_var_std_dev  # convert to z scores            
            x_range = np.arange(np.min(data_truncated),np.max(data_truncated),0.001)
            loc,scale = halfnorm.fit(data_truncated,floc=0)
            estimated_mu = float("{:.3f}".format(np.mean(data_truncated)))
            estimated_sigma = float("{:.3f}".format(np.std(data_truncated)))
            pdf = halfnorm.pdf(x_range,scale=scale)

            plt.plot(x_range,pdf,label='fitted_halfnorm_pdf')
            #plt.hist(data_truncated,bins=200,range=[np.min(data_truncated),np.max(data_truncated)+1],alpha=0.5,label=labelname,density=True,stacked=True)
            #plt.text(1.5,1.2,r'$\mu=$''%s\n'r'$\sigma=$''%s'%(estimated_mu,estimated_sigma)) 
    return data


def plot_histo_bbox_uc(dets,scene,min_val,max_val):
    #ax = dets.plot.hist(column='a_bbox_var',bins=12,alpha=0.5)
    bboxes = dets.filter(like='bb').columns
    for column in bboxes:
        if('a_bbox_var' in column):
            labelname = scene + '_' + column
            data = dets[column].to_list()  # uncertainty
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
            #min_vdraw   = np.mean(hist_data)
            #hist_data = (hist_data-min_val)/(max_val-min_val)
    return data

def extract_bad_predictions(df,confidence):
    """
    Filter low confidence predictions (filter out good predictions) and return 
    a new df. To be used with drawing and scatter plots.
    args: df, confidence(0-1,3.f)
    return: filtered df
    """
    confidence_idx = df['confidence']>confidence  # everything below the confidence threshold
    filtered_df = df[confidence_idx]
    return filtered_df

def extract_good_predictions(df,confidence):
    """
    Filter high confidence predictions (filter out bad predictions) and return 
    a new df. To be used with drawing and scatter plots.
    args: df, confidence(0-1,3.f)
    return: filtered df
    """
    confidence_idx = df['confidence']<confidence  # everything above the confidence threshold
    filtered_df = df[confidence_idx]
    return filtered_df

def extract_twosigma(df,mode=None):
    """
    Filter low confidence predictions beyond 2sigma log(var) in positive and negative direction
    return the filtered df, to be used with drawing and scatter plots.
    args: df, mode(0:high var, 1:low var)
    return: filtered twosigma df
    """
    a_bbox_var = np.asarray(df['a_bbox_var'].to_list())
    a_bbox_var = np.sum(a_bbox_var,axis=1)  # sum 
    a_bbox_var_log = np.log(a_bbox_var)
    df.insert(len(df.columns),'a_bbox_var_sum',a_bbox_var)
    df.insert(len(df.columns),'a_bbox_var_log',a_bbox_var_log)
    std_dev = np.std(a_bbox_var_log)
    mean = np.mean(a_bbox_var_log)
    twosigma = std_dev*2
    threshold = mean + twosigma
    if (not mode):
        filtered_df = df.loc[df['a_bbox_var_log'] >= threshold]  # high variance
    else:
        filtered_df = df.loc[df['a_bbox_var_log'] <= threshold]  # low variance
    return filtered_df
    
def draw_filtered_detections(df):
    """
    Draw filtered detections from dataframe
    open image, draw bbdets (for all dets in frame), repeat
    """
    frame_idx = np.asarray(df['frame_idx'])
    scene_idx = np.asarray(df['scene_idx'])
    pic_idx = (frame_idx + scene_idx) * 1000
    df.insert(len(df.columns),'pic_idx',pic_idx)  # insert picture values into df
    df = df.sort_values(by=['pic_idx'])  # sorting to group detections on SAME imgs (efficient iterating)
    
    # locate column indexs needed based on name 
    pic_column = df.columns.get_loc("pic_idx")
    bbdets_column = df.columns.get_loc("bbdet3d_2d")
    # convert dataframe to numpy arr for interating 
    data = df.values
    #data[:,pic_column] = np.char.zfill(data[:,pic_column],7)
    idx = data[0,pic_column]
    # open image
    img_data = Image.open(imgpath + (str(idx)).zfill(7) + '.png')
    draw = ImageDraw.Draw(img_data)
    for row in data:  # iterate through each detection
        current_idx = row[pic_column]  # either current pic or move to next pic
        if (current_idx == idx):  # string comparison, but is ok
            if (row[bbdets_column] == [-1,-1,-1,-1]):  # no detection
                continue
            draw.rectangle(row[bbdets_column])  # draw rectangle over jpeg
        else:
            out_file = os.path.join(savepath,(str(idx)).zfill(7))  # all detections drawn for pic, save and next
            img_data.save(out_file,'PNG')
            img_data = Image.open(imgpath + (str(current_idx)).zfill(7) + '.png')
            draw = ImageDraw.Draw(img_data)
            if (row[bbdets_column] == [-1,-1,-1,-1]):
                idx = current_idx  # update idx
                continue
            draw.rectangle(row[bbdets_column])  # must draw as for loop will iterate off detection
            idx = current_idx  # update idx

def plot_scatter_var(df,x,y):
    '''
    Scatter plot function x against y, y is usually variance (uncertainty)
    args: df, x, y (x and y must be same length)
    returns: None
    '''
    data_x = df[x].to_list()
    data_x = np.asarray(data_x)
    data_y = df[y].to_list()
    data_y = np.asarray(data_y)
    if  x == 'bbdet':  # x0,y0,x1,y1
        width = data_x[:,2]-data_x[:,0]
        length = data_x[:,3]-data_x[:,1]
        data_x = width * length 
        x = 'bbox_area'
    elif data_x[1].shape:  # for both data, check if they must be summed
        data_x = np.sum(data_x,axis=1) 
    if y == 'bbdet':  # x0,y0,x1,y1
        width = data_y[:,2]-data_y[:,0]
        length = data_y[:,3]-data_y[:,1]
        data_y = width * length 
        y = 'bbox_area'
    elif data_y[1].shape:
        data_y = np.sum(data_y,axis=1) 
        
    label = x + ' vs ' + y
    covariance = np.cov(data_x,data_y)
    plt.scatter(data_x,data_y,label=label,color='r',marker='*',s=1)
    #plt.text(.7,1,'cov = %s %s  %s %s' %(covariance[0,0],covariance[0,1],covariance[1,0],covariance[1,1])) 
    #plt.text(1.2,0,'cov = %s %s  %s %s' %(covariance[0,0],covariance[0,1],covariance[1,0],covariance[1,1]))
    print('covariance=\n', covariance[0,0],covariance[0,1],'\n',covariance[1,0],covariance[1,1]) 
    plt.xlabel(x)
    plt.ylabel(y)

def label_3D_to_image(json_calib, metadata, bbox):
    bbox_transform_matrix = get_box_transformation_matrix(bbox)  
    instrinsic = json_calib[0]['cam_intrinsic']
    extrinsic = np.array(json_calib[0]['cam_extrinsic_transform']).reshape(4,4)
    vehicle_to_image = get_image_transform(instrinsic, extrinsic)  # magic array 4,4 to multiply and get image domain
    box_to_image = np.matmul(vehicle_to_image, bbox_transform_matrix)


    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    # 1: 000, 2: 001, 3: 010:, 4: 100
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)

    return vertices

def get_box_transformation_matrix(box):
    """Create a transformation matrix for a given label box pose."""

    #tx,ty,tz = box.center_x,box.center_y,box.center_z
    tx = box[0]
    ty = box[1]
    tz = box[2]
    c = math.cos(box[3])
    s = math.sin(box[3])

    #sl, sh, sw = box.length, box.height, box.width
    sl = box[4]
    sh = box[5]
    sw = box[6]

    return np.array([
        [ sl*c,-sw*s,  0,tx],
        [ sl*s, sw*c,  0,ty],
        [    0,    0, sh,tz],
        [    0,    0,  0, 1]])

def get_image_transform(intrinsic, extrinsic):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """
    # Camera model:
    # | fx  0 cx 0 |
    # |  0 fy cy 0 |
    # |  0 0  1 0 |
    camera_model = np.array([
        [intrinsic[0], 0, intrinsic[2], 0],
        [0, intrinsic[1], intrinsic[3], 0],
        [0, 0,                       1, 0]])

    # Swap the axes around
    axes_transformation = np.array([
        [0,-1,0,0],
        [0,0,-1,0],
        [1,0,0,0],
        [0,0,0,1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic)))
    return vehicle_to_image

def compute_2d_bounding_box(points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    return (x1,y1,x2,y2)

def bbdet3d_to_bbdet2d(df):
    """
    Function to convert 3d bounding boxes to 2d. These transformed coordinates
    are appended to the df to be drawn. 
    args: df 
    returns: modified df
    """
    bbox2D_col = []
    # for each bbdet entry, transform it to image domain and append
    for index, row in df.iterrows():
        bbox2D = label_3D_to_image(row['calibration'],0,row['bbdet3d'])  # investigate meta 
        if (bbox2D is None):
            bbox2D = [-1,-1,-1,-1]
            bbox2D_col.append(bbox2D)
            continue
        bbox2D = compute_2d_bounding_box(bbox2D)
        bbox2D = [bbox2D[0], bbox2D[1]-top_crop, bbox2D[2], bbox2D[3]-top_crop]
        bbox2D_col.append(bbox2D)
    df['bbdet3d_2d'] = bbox2D_col
    return df

if __name__ == '__main__': 
    with open(detection_file) as det_file:
        dets_df  = parse_dets(det_file.readlines())
    df  = parse_labels(dets_df, gt_file)
    df = bbdet3d_to_bbdet2d(df)
    print(df)
    df = df.loc[df['difficulty'] != -1]
    #df   = df.loc[df['confidence'] > 0.9]
    night_dets = df.loc[df['tod'] == 'Night']
    day_dets = df.loc[df['tod'] == 'Day']
    rain_dets = df.loc[df['weather'] == 'rain']
    sun_dets = df.loc[df['weather'] == 'sunny']
    scene_dets = df.loc[df['scene_idx'] == 168]
    diff1_dets = df.loc[df['difficulty'] != 2]
    diff2_dets = df.loc[df['difficulty'] == 2]
    minm = 0
    maxm = .02
    # scene_data = plot_histo_bbox_uc(scene_dets,'scene',minm,maxm)
    #night_data = plot_histo_bbox_uc(night_dets,'night',minm,maxm)
    # day_data   = plot_histo_bbox_uc(day_dets,'day',minm,maxm)
    #day_mean = np.mean(day_dets)
    # day_mean = np.mean(day_data)
    #something = plot_histo_poisson(night_dets,'night',minm,maxm)
    #something = plot_histo_chi_squared(night_dets,'night',minm,maxm)
    #something = plot_histo_lognorm(df,df,'all_detections',minm,maxm)  # 
    #something = plot_histo_half_gaussian(df,df,'valid_predictions',minm,maxm)
    data, mu, sigma = plot_histo_gaussian(df,df,'valid_predictions',minm,maxm)
    #data, mu, sigma = plot_histo_lognorm(df,df,'valid_predictions',minm,maxm)  # 
    #something = plot_histo_log(df,'log_valid_predictions',minm,maxm)  # need mu and sigma
    #plot_scatter_var(df,'bbdet','a_bbox_var')
    #confidence = 0.9
    #filtered_df = extract_bad_predictions(df,confidence)
    filtered_df = extract_twosigma(df,1)  # none is high var, 1 is low var
    draw_filtered_detections(filtered_df)
    #print(len(night_data))
    # print(len(day_data))
    #r = scipy_stats.poisson.rvs(day_mean)
    # result = scipy_stats.ks_2samp(day_data,scene_data)
    # print(result)
    #plot_histo_bbox_uc(rain_dets,'rain',minm,maxm)
    #plot_histo_cls_uc(night_dets,'night',minm,maxm)
    #plot_histo_cls_uc(day_dets,'day',minm,maxm)
    #plot_histo_cls_uc(rain_dets,'rain',minm,maxm)
    #plot_histo_cls_uc(sun_dets,'sunny',minm,maxm)
    #plot_histo_bbox_uc(diff2_dets,'lvl2',minm,maxm)
    #plot_histo_bbox_uc(diff1_dets,'lvl1',minm,maxm)
    #print('mu = ',mu,'sigma = ',sigma)
    plt.legend()
    plt.show()
    #print(day_dets)
    #print(night_dets)
    #print(rain_dets)
