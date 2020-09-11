import pydicom
import os,re
import numpy as np
import shutil
import matplotlib.pyplot as plt
%matplotlib inline
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

file_name = "../slicelocation.csv"
icafepath = r'../iCafe/result/OAIMTP/without_xy/'
sample_num = 5
data = pd.read_csv(file_name)

caselist = []
for i in range(sample_num):  # self-defined
    caselist.append({'pid':str(data.loc[i,"PID"]), 'TP':[0,1,2,3,4,5,6,8,10], 'side':str(data.loc[i,"EID"])[-1]})

# #split seqs for a case
getFileFromDB(caselist)

if not os.path.exists(icafepath):
    os.mkdir(icafepath)
    
for casei in caselist:
    do_registration(casei, affine_registration, isVis=False)
    generate_result(casei)
    

def do_registration(casei, regis_fun, isVis=False):
    pi = casei['pid']
    side = casei['side']
    precasepath = split_seq_path + pi + side + '/'
    regtp = casei['TP']
    if len(regtp) < 2:
        print(pi, 'not enough TPs')
        return
    tp1 = regtp[0]
    print("Fixed Image %d" % tp1)
    tp1_casepath = precasepath + 'TP0/'
    fixed_image = get_dicom_series(tp1_casepath)
    
    icafesavepath = icafepath+'0_P'+pi+side+'_U'+'/'
    if not os.path.exists(icafesavepath):
        print('Init icafe dir')
        os.mkdir(icafesavepath)
    
    # isotropic resolution
    img1 = resample_image(fixed_image)
    sitk.WriteImage(sitk.Cast(img1, sitk.sitkInt16), icafesavepath + 'TH_0_P' + pi + side + '_U.tif')
    print('save to icafe path', icafesavepath + 'TH_0_P' + pi + side + '_U.tif')
    sitk.WriteImage(sitk.Cast(img1, sitk.sitkInt16), icafesavepath + 'TH_0_P'+  pi + side + '_US100.tif')
    print('save to icafe path', icafesavepath + 'TH_0_P' + pi + side + '_US100.tif')
    
    fixed_image = preprocess(fixed_image)
    for tp2 in regtp[1:]:
        #if tp2 != 1: continue
        print("************************************************")
        print("Processing %d......" % tp2)
        tp2_casepath = getdcmpath(pi,tp2,side)
        if tp2_casepath is None:
            print('cannot find dcm path for TPid', tp2)
            continue
        
        tp2_casepath = precasepath + 'TP' + str(tp2) + '/'
        print("reading image......", pi + 'TP' + str(tp2))
        moving_image = get_dicom_series(tp2_casepath)
        moving_image = preprocess(moving_image)
        
        transform = regis_fun(fixed_image, moving_image, isVis=isVis)
        if tp2==10:
            SEQ = 'S109'
        else:
            SEQ = 'S10'+str(tp2)
        sitk.WriteTransform(transform, transform_path + 'affine_P'+ pi + side + 'U' + SEQ + '.tfm')    
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
        resampled = transform_img(fixed_image, moving_image, affine)
        resampled = resample_image(resampled, space2)
        sitk.WriteImage(sitk.Cast(resampled, sitk.sitkInt16), icafesavepath + 'TH_0_P' + pid + side + '_U' + SEQ + '.tif')
    return