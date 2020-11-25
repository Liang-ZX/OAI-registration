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
from skimage.measure import label

from ipywidgets import interact, fixed

import cv2


def resample_image(fixed_image, space=None):
    if space is not None:
        isoresample = sitk.ResampleImageFilter()
        isoresample.SetInterpolator(sitk.sitkBSpline) # NearestNeighbor
        isoresample.SetOutputDirection(fixed_image.GetDirection())
        isoresample.SetOutputOrigin(fixed_image.GetOrigin())
        orig_size = np.array(fixed_image.GetSize(), dtype=np.int)    
        new_size = orig_size.copy()
        new_size[2] = int(orig_size[2]*(space[2]/space[0])+0.5)
        new_size = [int(s) for s in new_size]
        print(orig_size, new_size)
        isoresample.SetSize(new_size)
        orig_spacing = fixed_image.GetSpacing()
        new_spacing = (orig_spacing[0],orig_spacing[0],orig_spacing[0]*orig_size[2]/new_size[2])
        isoresample.SetOutputSpacing(new_spacing)
        return isoresample.Execute(fixed_image)
    else:
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

    
def resample_bwimage(fixed_image):
    isoresample = sitk.ResampleImageFilter()
    isoresample.SetInterpolator(sitk.sitkNearestNeighbor)
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
        binary1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#         img = cv2.GaussianBlur(img,(5,5),0)
#         canny = cv2.Canny(img, 50, 80, apertureSize = 3) # threshold2越大，提取的边缘越少
        dst = binary1
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
        
        img = cv2.medianBlur(img,3)
        img = cv2.GaussianBlur(img,(3,3),0)
        th1, binary1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
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

def get_component_process(images):
    new_images = []
    for z in range(images.GetDepth()):
        img = sitk.GetArrayFromImage(images[:,:,z])
        img = img.astype(np.uint8)
        kernel = np.ones((2, 2), dtype=np.uint8)
        tmp_img = cv2.erode(cv2.dilate(img, kernel), kernel)
#         contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        lcc = largestConnectComponent(tmp_img)
#         tmp = np.zeros((384,384))
#         cv2.drawContours(tmp,contours,-1,(255,255,255),-1) 
        new_images.append(lcc)
    img_array = np.array(new_images)
    return sitk.GetImageFromArray(img_array)
    
    
def preprocess(images):
    tmp_img = do_thresh(images)
    tmp_img = adaptive_thresh(tmp_img)
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


def largestConnectComponent(bw_img):
    labeled_img, num = label(bw_img, connectivity=1, background=0, return_num=True)    
    max_label = 0
    max_num = 0
    for i in range(1, num+1): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc.astype(np.uint8)

def show_sagittal_images(fixed_image, moving_image):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))
    
    fixed_npa = sitk.GetArrayViewFromImage(fixed_image)
    moving_npa = sitk.GetArrayViewFromImage(moving_image)
    size = np.size(fixed_npa, 2)
    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[:,:,size//2],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')
    
    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[:,:,size//2],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')
    
    plt.show()  

def show_masked_images(image, mask):
    interact(vis.display_images_with_mask, image_z=(0,image.GetSize()[2] - 1), 
             fixed = fixed(image), moving=fixed(mask));
    
def show_images(fixed_image, moving_image):
    interact(vis.display_images, fixed_image_z=(0,fixed_image.GetSize()[2]-1), moving_image_z=(0,moving_image.GetSize()[2]-1), 
         fixed_npa = fixed(sitk.GetArrayViewFromImage(fixed_image)), moving_npa=fixed(sitk.GetArrayViewFromImage(moving_image)));
    

def show_mixed_images(fixed_image, moving_image):
    interact(vis.display_images_with_alpha, image_z=(0,fixed_image.GetSize()[2] - 1), alpha=(0.0,1.0,0.05), fixed = fixed(fixed_image), moving=fixed(moving_image));