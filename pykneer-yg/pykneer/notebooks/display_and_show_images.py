import matplotlib.pyplot as plt
import sys
import os
%matplotlib inline
import numpy as np
from ipywidgets import interact, fixed
sys.path.append("../../..")
from MyProcessDicom import *
import SimpleITK as sitk

def show_slice_images(all_image_data):
    last_name = '0000000L'
    # for each image
    for i in range(0, len(all_image_data)):

        # get paths and file names of the current image
        image_data                    = all_image_data[i]
#         if image_data['moving_name'][:12] != last_name:
#             last_name = image_data['moving_name'][:8]
        print(image_data['moving_name'][:12])
            
        moving_file_name              = image_data['registered_sub_folder']    + image_data['fspline_name']
        reference_file_name              = image_data['reference_folder']    + image_data['reference_name']
        reference = sitk.ReadImage(reference_file_name)
        moving = sitk.ReadImage(moving_file_name)
        show_mixed_images(reference, moving)
        
        
def display_images_with_mask(image_z, fixed, moving):
    # img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z]
    fixed = sitk.GetArrayFromImage(fixed)
    moving = sitk.GetArrayFromImage(moving) # mask
    dst = fixed[image_z,:,:]*0.5+moving[image_z,:,:]*0.5*255
    plt.figure(figsize=(9,9))
    plt.imshow(dst, cmap=plt.cm.Greys_r)
    # plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()

def show_masked_images(image, mask):
    interact(display_images_with_mask, image_z=(0,image.GetSize()[2] - 1), 
             fixed = fixed(image), moving=fixed(mask));
    

def display_images_with_mask_sagittal(image_z, fixed, moving):
    # img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z]
    fixed = sitk.GetArrayFromImage(fixed)
    moving = sitk.GetArrayFromImage(moving) # mask
    dst = fixed[:,:,image_z]*0.5+moving[:,:,image_z]*0.5*255
    plt.figure(figsize=(7,7))
    plt.imshow(dst, cmap=plt.cm.Greys_r)
    # plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()

def show_masked_images_sagittal(image, mask):
    interact(display_images_with_mask_sagittal, image_z=(0,image.GetSize()[0] - 1), 
             fixed = fixed(image), moving=fixed(mask));

    
def display_images_with_all_masks(image_z, alpha, fixed, moving, tibia, femur_show = 1):
    # img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z]
    fixed = sitk.GetArrayFromImage(fixed)
    moving = sitk.GetArrayFromImage(moving) # mask
    tibia = sitk.GetArrayFromImage(tibia) # mask
    dst = fixed[:,:,image_z]*alpha+moving[:,:,image_z]*(1-alpha)*255*femur_show+tibia[:,:,image_z]*(1-alpha)*255
    dst = np.flipud(dst)
    plt.figure(figsize=(7,7))
    plt.imshow(dst, cmap=plt.cm.Greys_r)
    # plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()

    
def show_all_masked_images(image, mask, tibia, femur_show=1):
    interact(display_images_with_all_masks, image_z=(0,image.GetSize()[0] - 1), alpha=(0.0,1.0,0.05),
             fixed = fixed(image), moving=fixed(mask), tibia=fixed(tibia), femur_show=fixed(femur_show));
    
    
def show_sagital_slice_images(all_image_data, is_rigid=False):
    last_name = '0000000L'
    # for each image
    for i in range(0, len(all_image_data)):
#     for i in range(0, 4):

        # get paths and file names of the current image
        image_data                    = all_image_data[i]
#         if image_data['moving_name'][:8] != last_name:
#             last_name = image_data['moving_name'][:8]
        print(image_data['moving_name'][:12])       
   
        # mask_file_name = image_data[0]['reference_folder'] + '9941231L_TP0_prep_f.mha'
        if is_rigid:
            mask_file_name = image_data['segmented_folder'] + image_data['fmask'][:-5] + 'f_rigid.mha'
            tibia_file_name = image_data['segmented_folder'] + image_data['fmask'][:-5] + 't_rigid.mha'
        else:
            mask_file_name = image_data['segmented_folder'] + image_data['fmask']
            tibia_file_name = image_data['segmented_folder'] + image_data['fmask'][:-5] + 't.mha'
        # mask_file_name = 'C:\\Zhixuan\\OAI-registration\\pykneer-yg\\reference\\longitudinal\\9941446L_TP0_prep_t.mha'
        moving_file_name   = image_data['moving_folder']    + image_data['moving_name']
        mask = sitk.ReadImage(mask_file_name)
        tibia = sitk.ReadImage(tibia_file_name)
        moving = sitk.ReadImage(moving_file_name)
        key = 1
        if key == 1:
            mask = resample_bwimage(mask)
            tibia = resample_bwimage(tibia)
            moving = resample_bwimage(moving)
        #     show_masked_images_sagittal(moving, mask)
            show_all_masked_images(moving, mask,tibia)
        else:
            show_masked_images(moving, mask)    

            
def show_sagital_reference(all_image_data, femur_show = 0):
    last_name = '0000000L'
    # for each image
    for i in range(0, len(all_image_data)):
#     for i in range(0, 4):

        # get paths and file names of the current image
        image_data                    = all_image_data[i]
#         if image_data['moving_name'][:8] != last_name:
#             last_name = image_data['moving_name'][:8]
        print(image_data['moving_name'][:12])       
   
        # mask_file_name = image_data[0]['reference_folder'] + '9941231L_TP0_prep_f.mha'
        mask_file_name = image_data['segmented_folder'] + image_data['fmask']
        tibia_file_name = "C:\\Zhixuan\\OAI-registration\\pykneer-yg\\reference\\longitudinal\\" + image_data['fmask'][:-5] + 't.mha'
        # mask_file_name = 'C:\\Zhixuan\\OAI-registration\\pykneer-yg\\reference\\longitudinal\\9941446L_TP0_prep_t.mha'
        moving_file_name   = image_data['moving_folder']    + image_data['moving_name']
        mask = sitk.ReadImage(mask_file_name)
        tibia = sitk.ReadImage(tibia_file_name)
        moving = sitk.ReadImage(moving_file_name)
        key = 1
        if key == 1:
            mask = resample_bwimage(mask)
            tibia = resample_bwimage(tibia)
            moving = resample_bwimage(moving)
        #     show_masked_images_sagittal(moving, mask)
            show_all_masked_images(moving, mask,tibia, femur_show)
        else:
            show_masked_images(moving, mask)    
            