import pydicom
import os,re
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import SimpleITK as sitk
from IPython.display import clear_output
import math
import sys

from ipywidgets import interact, fixed


def get_dicom_series(file_path):
    file_path = file_path + '/'
    if not os.path.exists(file_path+'../coronal'):
        os.mkdir(file_path+'../coronal')
    TP1path = []
    flist = os.listdir(file_path)
    for f in flist:
        dicomfilename = file_path + f
        dcmimgcr = pydicom.dcmread(dicomfilename)
        if abs(float(dcmimgcr.ImageOrientationPatient[0]))<abs(float(dcmimgcr.ImageOrientationPatient[1])):
            #coronal slice showing scan areas
            tmp_path = file_path[-5:-1]
            if tmp_path[0] != '/':
                tmp_path = '/' + tmp_path
            shutil.move(dicomfilename,file_path+'../coronal'+tmp_path+'_'+f)
            print('coronal',dicomfilename)
            print('move '+ dicomfilename +' to '+ file_path+'../coronal'+tmp_path+'_'+f)
        else:
            TP1path.append(dicomfilename)
    TP1path.sort()
    fixed_image = sitk.ReadImage(TP1path)
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    fixed_image = resacleFilter.Execute(fixed_image)
    return fixed_image


# return numpy images && sitk images
def read_dicom_image(file_path):
    reader = sitk.ImageSeriesReader()
    series_Names = reader.GetGDCMSeriesFileNames(file_path)
    print(len(series_Names))
    reader.SetFileNames(series_Names)
    image3D = reader.Execute()
    print('size =', image3D.GetSize()) #width(x), height(y), depth(z) 
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image3D = resacleFilter.Execute(image3D)
#     origin = image3D.GetOrigin() # x, y, z
#     spacing = image3D.GetSpacing() # x, y, z
#     sitk.WriteImage(image3D, 'img3D.dcm')
    img_array = sitk.GetArrayFromImage(image3D) # z, y, x
    img_array = np.transpose(img_array,(2,1,0)) 
    return img_array, image3D


# sitk.Cast(fixed_image, sitk.sitkInt16)
# save sitk images
def write_dicom_series(img_series, save_path, file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sitk.WriteImage(img_series,save_path+"/"+file_name+".dcm")
#     for z in range(img_series.GetDepth()):
#         img = img_series[:,:,z]
#         sitk.WriteImage(img,save_path+"/%03d.dcm" % (z+1))
