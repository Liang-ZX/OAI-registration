import pydicom
import os,re
import numpy as np
import shutil
import cv2
from pydicom.dataset import Dataset, FileDataset
import tempfile
import SimpleITK as sitk
import math
from ipywidgets import interact, fixed
from skimage.transform import resize
from IPython.display import clear_output
from scipy.ndimage import zoom
import scipy.misc
from sys import platform
import shutil
import sys

sys.path.append("..")
from MyProcessDicom import *
from RegistrationMethod import *
from ServerDicomIO import *
from icafeAPI import generate_centerline

import importlib
import sys
sys.path.append(r'..\FRAPPE')
import DB
importlib.reload(DB)
from DB import DB
import pandas as pd


def get_centerline(pid, TPid, side='L'):
    dir_path = r'../centerline/P'+pid+side
    file_path = dir_path+'/tracing_raw_ves_TH_'+str(TPid)+'_P'+pid+side+'_U.swc'
    df = pd.read_csv(file_path, header=None, sep=' ',index_col=0, names=['vessel_id','x','y','z','undefined','last_id'])
    line = np.array([df['x'], df['y'], df['z']]).T
    return df, line


def generate_regis_with_centerline(line1, line2, transform):
    points = []
    for i in range(line1.shape[0]):
        tmp_point = transform.TransformPoint((line1[i][0], line1[i][1], i))
        points.append(tmp_point)
    ans = np.array(points)
    ans[:,2] = np.round(ans[:,2])
    dx, dy = [], []
    for i in range(ans.shape[0]):
        if int(ans[i][2]) >= line2.shape[0]:
            break
        dx.append(ans[i][0] - line2[int(ans[i][2])][0])
        dy.append(ans[i][1] - line2[int(ans[i][2])][1])
    dx = np.array(dx)
    dy = np.array(dy)
    displacement_image = sitk.Image([384,384,dx.shape[0]], sitk.sitkVectorFloat64)
    # The only point that has any displacement is (0,0)
    for i in range(dx.shape[0]):
        displacement = (-dx[i],-dy[i],0)
        for j in range(384):
            for k in range(384):
                displacement_image[j,k,i] = displacement
    return sitk.DisplacementFieldTransform(displacement_image)

def registration_with_centerline(case):
    pid = case['pid']
    side = case['side']
    regtp = case['TP']
    if len(regtp) < 2:
        print(pid, 'not enough TPs')
        return
    tp1 = regtp[0]
    print("Fixed Image %d" % tp1)
    _, line1 = get_centerline(pid, tp1, side)
    
    tp1_casepath = precasepath + 'TP0/'
    fixed_image = get_dicom_series(tp1_casepath)
    icafesavepath = icafepath+'0_P'+pi+side+'_U'+'/'
    
    # isotropic resolution
    img1 = resample_image(fixed_image)
    sitk.WriteImage(sitk.Cast(img1, sitk.sitkInt16), icafesavepath + 'TH_0_P' + pi + side + '_U.tif')
    print('save to icafe path', icafesavepath + 'TH_0_P' + pi + side + '_U.tif')
    sitk.WriteImage(sitk.Cast(img1, sitk.sitkInt16), icafesavepath + 'TH_0_P'+  pi + side + '_US100.tif')
    print('save to icafe path', icafesavepath + 'TH_0_P' + pi + side + '_US100.tif')
    
    for tp2 in regtp[1:]:
        tp2_casepath = getdcmpath(pid,tp2,side)
        if tp2_casepath is None:
            print('cannot find dcm path for TPid', tp2)
            continue
        _, line2 = get_centerline(pid, tp2, side)
        if tp2==10:
            SEQ = 'S109'
        else:
            SEQ = 'S10'+str(tp2)
        affine = sitk.ReadTransform(transform_path + 'affine_P'+  case['pid'] + case['side'] + 'U' + SEQ + '.tfm')
        print("Please wait...(TP"+str(tp2)+" is being processed)")
        displacement_field_transform = generate_regis_with_centerline(line1, line2, affine)
        sitk.WriteTransform(displacement_field_transform, transform_path + 'displacement_P'+  pid + side + 'U' + SEQ + '.tfm')
    return


def generate_result(case):
    pid = case['pid']
    side = case['side']
    regtp = case['TP']
    icafesavepath = icafepath+'0_P'+pid+side+'_U'+'/'
    precasepath = split_seq_path + pid + side + '/'
    if len(regtp) < 2:
        print(pid, 'not enough TPs')
        return
    tp1 = regtp[0]
    tp1_casepath = precasepath + 'TP0/'
    fixed_image = get_dicom_series(tp1_casepath)
    fixed_image = empty_preprocess(fixed_image)
    print("Fixed Image %d" % tp1)
    for tp2 in regtp[1:]:
        tp2_casepath = getdcmpath(pid,tp2,side)
        if tp2_casepath is None:
            print('cannot find dcm path for TPid', tp2)
            continue
        tp2_casepath = precasepath + 'TP' + str(tp2) + '/'
        moving_image = get_dicom_series(tp2_casepath)
        space2 = moving_image.GetSpacing()
        if tp2==10:
            SEQ = 'S109'
        else:
            SEQ = 'S10'+str(tp2)
        moving_image = empty_preprocess(moving_image)
        print("Please wait...(TP"+str(tp2)+" is being processed)")
        affine = sitk.ReadTransform(transform_path + 'affine_P'+  case['pid'] + case['side'] + 'U' + SEQ + '.tfm')
        displacement_field_transform = sitk.ReadTransform(transform_path + 'displacement_P'+  pid + side + 'U' + SEQ + '.tfm')
        resampled = transform_img(fixed_image, moving_image, affine)
        resampled2 = transform_img(fixed_image, resampled, displacement_field_transform)
        resampled2 = resample_image(resampled2, space2)
        sitk.WriteImage(sitk.Cast(resampled2, sitk.sitkInt16), icafesavepath + 'TH_0_P' + pid + side + '_U' + SEQ + '.tif')
    return


file_name = "../slicelocation.csv"
icafepath = r'../iCafe/result/with_translation/'
sample_num = 25
offset = 9
data = pd.read_csv(file_name)

caselist = []
for i in range(sample_num):  # self-defined
    caselist.append({'pid':str(data.loc[i+offset,"PID"]), 'TP':[0,1,2,3,4,5,6,8,10], 'side':str(data.loc[i+offset,"EID"])[-1]})

if not os.path.exists(icafepath):
    os.mkdir(icafepath)

for case in caselist:
    pid = case['pid']
    side = case['side']
    for j in range(len(case['TP'])):
        generate_centerline(pid, case['TP'][j], side)
    file_path = r'../centerline/P'+pid+side+'/tracing_raw_ves_TH_'+str(case['TP'][0])+'_P'+pid+side+'_U.swc'
    icafesavepath = icafepath+'0_P'+pid+side+'_U'+'/'
    if not os.path.exists(icafesavepath):
        os.mkdir(icafesavepath)
    tmp_path = r'../iCafe/result/OAIMTP/without_xy/0_P'+pid+side+'_U'+'/'
    shutil.copy(file_path, icafesavepath)
    shutil.copy(file_path, tmp_path)
    registration_with_centerline(case)
    generate_result(case)
