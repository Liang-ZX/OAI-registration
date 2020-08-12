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
import AssistVisualization as vis

from ipywidgets import interact, fixed

import cv2


def resample_image(fixed_image):
    isoresample = sitk.ResampleImageFilter()
    isoresample.SetInterpolator(sitk.sitkBSpline)
    isoresample.SetOutputDirection(fixed_image.GetDirection())
    isoresample.SetOutputOrigin(fixed_image.GetOrigin())
    orig_spacing = fixed_image.GetSpacing()
    new_spacing = (orig_spacing[0],orig_spacing[0],orig_spacing[0])
    isoresample.SetOutputSpacing(new_spacing)
    orig_size = np.array(fixed_image.GetSize(), dtype=np.int)    
    new_size = orig_size.copy()
    new_size[2] = int(orig_size[2]*(orig_spacing[2]/orig_spacing[0])+0.5)
    new_size = [int(s) for s in new_size]
    print(orig_size, new_size)
    isoresample.SetSize(new_size)
    return isoresample.Execute(fixed_image)


def adaptive_thresh(images):
    new_images = []
    for z in range(images.GetDepth()):
        img = sitk.GetArrayFromImage(images[:,:,z])
        img = img.astype(np.uint8)
#         th1, binary1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#         img = cv2.medianBlur(img,5)
#         binary1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        img = cv2.GaussianBlur(img,(5,5),0)
        canny = cv2.Canny(img, 50, 80, apertureSize = 3) # threshold2越大，提取的边缘越少
        dst = canny
        size = 2
        kernel = np.ones((size, size), dtype=np.uint8)
        img_open = cv2.erode(cv2.dilate(dst, kernel), kernel)
        new_images.append(img_open)
    img_array = np.array(new_images)
    return sitk.GetImageFromArray(img_array)


def do_thresh(images):
    new_images = []
    for z in range(images.GetDepth()):
        img = sitk.GetArrayFromImage(images[:,:,z])
        img = img.astype(np.uint8)
        th1, binary1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#         img = cv2.medianBlur(img,5)
#         binary1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        new_images.append(binary1)
    img_array = np.array(new_images)
    return sitk.GetImageFromArray(img_array)


def empty_preprocess(images):
    new_images = []
    for z in range(images.GetDepth()):
        img = sitk.GetArrayFromImage(images[:,:,z])
        img = img.astype(np.uint8)
        new_images.append(img)
    return sitk.GetImageFromArray(np.array(new_images))


def postprocess(images):
    new_images = []
    for z in range(images.GetDepth()):
        img = sitk.GetArrayFromImage(images[:,:,z])
        img = img.astype(np.uint8)
#         th1, binary1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.GaussianBlur(img,(5,5),0)
#         th1, binary1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        dst = img
        kernel = np.ones((5, 5), dtype=np.uint8)
        img_open = cv2.erode(cv2.dilate(dst, kernel), kernel)
        new_images.append(img_open)
    img_array = np.array(new_images)
    return sitk.GetImageFromArray(img_array)


def preprocess(images):
    tmp_img = do_thresh(images)
    return tmp_img


# def preprocess(images):
#     new_images = []
#     for z in range(images.GetDepth()):
#         img = sitk.GetArrayFromImage(images[:,:,z])
#         img = img.astype(np.uint8)
#         dst = cv2.equalizeHist(img)
#         size = 3
#         kernel = np.ones((size, size), dtype=np.uint8)
#         img_erosion = cv2.erode(dst, kernel, iterations=1)
#         img_dilation = cv2.dilate(dst, kernel, iterations=1)
#         img_open = cv2.dilate(cv2.erode(dst, kernel), kernel)
#         new_images.append(dst)
#     img_array = np.array(new_images)

#     return sitk.GetImageFromArray(img_array)


# def edge_detect(images, threshold2=100):
#     new_images = []
#     for z in range(images.GetDepth()):
#         img = sitk.GetArrayFromImage(images[:,:,z])
#         img = img.astype(np.uint8)
        
#         img = cv2.GaussianBlur(img,(5,5),0)
#         canny = cv2.Canny(img, 50, threshold2, apertureSize = 3) # threshold2越大，提取的边缘越少
        
#         new_images.append(canny)
#     img_array = np.array(new_images)

#     return sitk.GetImageFromArray(img_array)


def transform_img(fixed_image, moving_image, transform):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)    
    resample.SetInterpolator(sitk.sitkBSpline)  
    resample.SetTransform(transform)
    return resample.Execute(moving_image)


def show_images(fixed_image, moving_image):
    interact(vis.display_images, fixed_image_z=(0,fixed_image.GetSize()[2]-1), moving_image_z=(0,moving_image.GetSize()[2]-1), 
         fixed_npa = fixed(sitk.GetArrayViewFromImage(fixed_image)), moving_npa=fixed(sitk.GetArrayViewFromImage(moving_image)));
    

def show_mixed_images(fixed_image, moving_image):
    interact(vis.display_images_with_alpha, image_z=(0,fixed_image.GetSize()[2] - 1), alpha=(0.0,1.0,0.05), fixed = fixed(fixed_image), moving=fixed(moving_image));