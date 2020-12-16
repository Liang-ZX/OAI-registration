# Serena Bonaretti, 2018

"""
Module with the class registration. There are two instances: bone and cartilage. 
For each instance, the common methods are :
    - rigid, similarity, and spline register the moving to the reference
    - i_rigid, i_similarity, and i_spline invert the transformations 
    - t_rigid, t_similarity, and t_spline warp the reference mask to the moving image using the inverted tranformation
Other functions in the abstract class are: 
    - prepare_reference 
    - modify_transformation 
The instance bone also has the function: 
    - vf_spline used to find the reference bone (see find_reference.py)
    
Functions at the bottom are to test when elastix does not work - the output messages are in the function "rigid" of the class "bone" (the first function used in the pipeline)
"""

from abc import ABC, abstractmethod
import os
import subprocess
import SimpleITK as sitk
from preparing_images import modify_output_line

import pkg_resources
import platform
import subprocess

# pyKNEER imports 
# ugly way to use relative vs. absolute imports when developing vs. when using package - cannot find a better way
if __package__ is None or __package__ == '':
    # uses current directory visibility
    import sitk_functions  as sitkf
else:
    # uses current package visibility
    from . import sitk_functions  as sitkf


# ---------------------------------------------------------------------------------------------------------------------------
# ABSTRACT CLASS FOR REGISTRATION -------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
class registration(ABC):

    @abstractmethod
    def rigid(self):
        pass

    @abstractmethod
    def similarity(self):
        pass

    @abstractmethod
    def spline(self):
        pass

    @abstractmethod
    def i_rigid(self):
        pass

    @abstractmethod
    def i_similarity(self):
        pass

    @abstractmethod
    def i_spline(self):
        pass

    @abstractmethod
    def t_rigid(self):
        pass

    @abstractmethod
    def t_similarity(self):
        pass

    @abstractmethod
    def t_spline(self):
        pass

    @abstractmethod
    def vf_spline(self):
        pass

    def rm_reference(self, image_data):
        anatomy                      = image_data["current_anatomy"]
        reference_mask_name          = image_data["reference_folder"] + image_data[anatomy + "mask_file_name"]
        reference_mask_dil_name      = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        reference_mask_levelset_name = image_data["reference_folder"] + image_data[anatomy + "levelset_mask_file_name"]
        radius                       = image_data["dilate_radius"]

        # dilate mask
        if os.path.exists(reference_mask_dil_name):
            os.remove(reference_mask_dil_name)

        # convert mask from binary to levelset for warping
        if os.path.exists(reference_mask_levelset_name):
            os.remove(reference_mask_levelset_name)
    
    def rm_tibia_reference(self, image_data):
        anatomy                      = image_data["bone"]
        reference_mask_name          = image_data[anatomy + "mask_file_name"]
        reference_mask_name = reference_mask_name[:-5] + 't' + reference_mask_name[-4:]
        reference_mask_name          = image_data["reference_folder"] + reference_mask_name
        reference_mask_dil_name      = image_data[anatomy + "dil_mask_file_name"]
        reference_mask_dil_name = reference_mask_dil_name[:-8] + 't' + reference_mask_dil_name[-7:]
        reference_mask_dil_name      = image_data["reference_folder"] + reference_mask_dil_name
        reference_mask_levelset_name = image_data[anatomy + "levelset_mask_file_name"]
        reference_mask_levelset_name = reference_mask_levelset_name[:-14] + 't' + reference_mask_levelset_name[-13:]
        reference_mask_levelset_name = image_data["reference_folder"] + reference_mask_levelset_name
        radius                       = image_data["dilate_radius"]

        # dilate mask
        if os.path.exists(reference_mask_dil_name):
            os.remove(reference_mask_dil_name)

        # convert mask from binary to levelset for warping
        if os.path.exists(reference_mask_levelset_name):
            os.remove(reference_mask_levelset_name)
    
    def prepare_reference(self, image_data):

        anatomy                      = image_data["current_anatomy"]
        reference_mask_name          = image_data["reference_folder"] + image_data[anatomy + "mask_file_name"]
        reference_mask_dil_name      = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        reference_mask_levelset_name = image_data["reference_folder"] + image_data[anatomy + "levelset_mask_file_name"]
        radius                       = image_data["dilate_radius"]

        # dilate mask
        if not os.path.isfile(reference_mask_dil_name):
            mask    = sitk.ReadImage(reference_mask_name)
            maskDil = sitkf.dilate_mask(mask, radius)
            sitk.WriteImage(maskDil, reference_mask_dil_name)

        # convert mask from binary to levelset for warping
        if not os.path.isfile(reference_mask_levelset_name):
            mask   = sitk.ReadImage(reference_mask_name)
            maskLS = sitkf.binary2levelset(mask)
            sitk.WriteImage(maskLS, reference_mask_levelset_name)

            
    def prepare_tibia_reference(self, image_data):

        anatomy                      = image_data["bone"]
        reference_mask_name          = image_data[anatomy + "mask_file_name"]
        reference_mask_name = reference_mask_name[:-5] + 't' + reference_mask_name[-4:]
        reference_mask_name          = image_data["reference_folder"] + reference_mask_name
        reference_mask_dil_name      = image_data[anatomy + "dil_mask_file_name"]
        reference_mask_dil_name = reference_mask_dil_name[:-8] + 't' + reference_mask_dil_name[-7:]
        reference_mask_dil_name      = image_data["reference_folder"] + reference_mask_dil_name
        reference_mask_levelset_name = image_data[anatomy + "levelset_mask_file_name"]
        reference_mask_levelset_name = reference_mask_levelset_name[:-14] + 't' + reference_mask_levelset_name[-13:]
        reference_mask_levelset_name = image_data["reference_folder"] + reference_mask_levelset_name
        radius                       = image_data["dilate_radius"]

        # dilate mask
        if not os.path.isfile(reference_mask_dil_name):
            mask    = sitk.ReadImage(reference_mask_name)
            maskDil = sitkf.dilate_mask(mask, radius)
            sitk.WriteImage(maskDil, reference_mask_dil_name)

        # convert mask from binary to levelset for warping
        if not os.path.isfile(reference_mask_levelset_name):
            mask   = sitk.ReadImage(reference_mask_name)
            maskLS = sitkf.binary2levelset(mask)
            sitk.WriteImage(maskLS, reference_mask_levelset_name)


    def modify_transformation(self, image_data, transformation): 
        """
        It creates a new parameter file to calculate the inverted transformation
        Input: 
            parameter file used for registration of moving to reference
        Output: 
            modifided parameter files used to calculate the inverted transformation
        """

        anatomy = image_data["current_anatomy"]

        # transformation file name
        if transformation   == "rigid":
            input_file_name  = image_data["i_registered_sub_folder"] + image_data[anatomy + "i_rigid_transf_name"]
            output_file_name = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_transf_name"]
        elif transformation == "similarity":
            input_file_name  = image_data["i_registered_sub_folder"] + image_data[anatomy + "i_similarity_transf_name"]
            output_file_name = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_similarity_transf_name"]
        elif transformation == "spline":
            input_file_name  = image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"]
            output_file_name = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_spline_transf_name"]
        else:
            print("----------------------------------------------------------------------------------------")
            print("ERROR: This transformation is not supported. Use 'rigid', 'similarity', or 'spline'")
            print("----------------------------------------------------------------------------------------")
            return

        # check if transformation file exists
        if not os.path.exists(input_file_name):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: The file  %s does not exist" % (input_file_name) )
            print("----------------------------------------------------------------------------------------")
            return

        # read the file and modify the needed lines
        file_content=[]
        i = 0
        for line in open(input_file_name):
            file_content.append(line)
            if "InitialTransformParametersFileName" in file_content[i]:
                file_content[i] = "(InitialTransformParametersFileName \"NoInitialTransform\")\n"
            if "DefaultPixelValue" in file_content[i]:
                file_content[i] = "(DefaultPixelValue -4)\n"
            if "Size" in file_content[i] and transformation == "rigid":
                file_content[i] = "(Size %s %s %s)\n" % (image_data["image_size"][0], image_data["image_size"][1], image_data["image_size"][2])
            if "Spacing" in file_content[i] and transformation == "rigid":
                file_content[i] = "(Spacing %s %s %s)\n" % (image_data["image_spacing"][0], image_data["image_spacing"][1], image_data["image_spacing"][2])
            i = i+1

        # write the file
        f = open(output_file_name,"w")
        for i in range(0,len(file_content)):
            f.write(file_content[i])
        f.close()
        
        
    def modify_tibia_transformation(self, image_data, transformation): 
        """
        It creates a new parameter file to calculate the inverted transformation
        Input: 
            parameter file used for registration of moving to reference
        Output: 
            modifided parameter files used to calculate the inverted transformation
        """

        anatomy = image_data["current_anatomy"]

        # transformation file name
        if transformation   == "rigid":
            input_file_name  = image_data["i_registered_sub_folder"] + image_data[anatomy + "i_rigid_transf_name"]
            output_file_name = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_transf_name"]
            input_file_name = input_file_name[:-11] + 't' + input_file_name[-10:]
            output_file_name = output_file_name[:-11] + 't' + output_file_name[-10:]
        elif transformation == "similarity":
            input_file_name  = image_data["i_registered_sub_folder"] + image_data[anatomy + "i_similarity_transf_name"]
            output_file_name = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_similarity_transf_name"]
            input_file_name = input_file_name[:-16] + 't' + input_file_name[-15:]
            output_file_name = output_file_name[:-16] + 't' + output_file_name[-15:]
        elif transformation == "spline":
            input_file_name  = image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"]
            output_file_name = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_spline_transf_name"]
            input_file_name = input_file_name[:-12] + 't' + input_file_name[-11:]
            output_file_name = output_file_name[:-12] + 't' + output_file_name[-11:]
        else:
            print("----------------------------------------------------------------------------------------")
            print("ERROR: This transformation is not supported. Use 'rigid', 'similarity', or 'spline'")
            print("----------------------------------------------------------------------------------------")
            return
        
        # check if transformation file exists
        if not os.path.exists(input_file_name):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: The file  %s does not exist" % (input_file_name) )
            print("----------------------------------------------------------------------------------------")
            return

        # read the file and modify the needed lines
        file_content=[]
        i = 0
        for line in open(input_file_name):
            file_content.append(line)
            if "InitialTransformParametersFileName" in file_content[i]:
                file_content[i] = "(InitialTransformParametersFileName \"NoInitialTransform\")\n"
            if "DefaultPixelValue" in file_content[i]:
                file_content[i] = "(DefaultPixelValue -4)\n"
            if "Size" in file_content[i] and transformation == "rigid":
                file_content[i] = "(Size %s %s %s)\n" % (image_data["image_size"][0], image_data["image_size"][1], image_data["image_size"][2])
            if "Spacing" in file_content[i] and transformation == "rigid":
                file_content[i] = "(Spacing %s %s %s)\n" % (image_data["image_spacing"][0], image_data["image_spacing"][1], image_data["image_spacing"][2])
            i = i+1

        # write the file
        f = open(output_file_name,"w")
        for i in range(0,len(file_content)):
            f.write(file_content[i])
        f.close()



# ---------------------------------------------------------------------------------------------------------------------------
# BONE INSTANCE -------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

class bone (registration):

    # This function is specific for each bone
    def rigid(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        complete_moving_name             = image_data["moving_folder"]    + image_data["moving_name"]
        # parameters
        params                           = image_data["param_file_rigid"]
        # output folder
        output_folder                    = image_data["registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        # print ("     Rigid registration")
        # os.path.abspath because working directory is where elastix is in pykneer, so images need whole path, not only relative
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_moving_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder)]

        subprocess.run(cmd, cwd=elastix_path)

        # check if the registration worked
        # if the registration did not work
        if not os.path.exists(image_data["registered_sub_folder"] + "result.0.mha"):
            print("-------------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.rigid()")
            print("-------------------------------------------------------------------------------------------")
            
            # print out possible errors
            # elastix not in system
            output = test_elastix()
            if output == 0:
                print ("-> elastix correctly installed")
                print ("complete_elastix_path           : " + complete_elastix_path)
                print ("complete_reference_name         : " + complete_reference_name)
                print ("complete_reference_mask_dil_name: " + complete_reference_mask_dil_name)
                print ("complete_moving_name            : " + complete_moving_name)
                print ("params                          : " + params)
                print ("output_folder                   : " + output_folder)
            else: 
                print ("-> elastix not installed. Error message is: " + output)
                print ("   You might need to set Elastix environmental variables separately. To do so, go to pyKNEEr documentation here: https://sbonaretti.github.io/pyKNEEr/faq.html#elastix")
            
            
            
            
            return
        # change output names
        else:
            if os.path.exists(image_data["registered_sub_folder"] + image_data[anatomy + "rigid_name"]):
                os.remove(image_data["registered_sub_folder"] + image_data[anatomy + "rigid_name"])
                os.remove(image_data["registered_sub_folder"] + image_data[anatomy + "rigid_transf_name"])
            os.rename(image_data["registered_sub_folder"] + "result.0.mha",
                      image_data["registered_sub_folder"] + image_data[anatomy + "rigid_name"])
            os.rename(image_data["registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["registered_sub_folder"] + image_data[anatomy + "rigid_transf_name"])


    def similarity(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"]      + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"]      + image_data[anatomy + "dil_mask_file_name"]
        complete_moving_name             = image_data["registered_sub_folder"] + image_data[anatomy + "rigid_name"]
        # parameters
        params                           = image_data["param_file_similarity"]
        # output folder
        output_folder                    = image_data["registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Similarity registration")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_moving_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["registered_sub_folder"] + "result.0.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.similarity()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["registered_sub_folder"] + image_data[anatomy + "similarity_name"]):
                os.remove(image_data["registered_sub_folder"] + image_data[anatomy + "similarity_name"])
                os.remove(image_data["registered_sub_folder"] + image_data[anatomy + "similarity_transf_name"])
            os.rename(image_data["registered_sub_folder"] + "result.0.mha",
                      image_data["registered_sub_folder"] + image_data[anatomy + "similarity_name"])
            os.rename(image_data["registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["registered_sub_folder"] + image_data[anatomy + "similarity_transf_name"])


    def spline(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        if image_data["registration_type"] == "newsubject" or image_data["registration_type"] == "multimodal":
            complete_moving_name         = image_data["registered_sub_folder"] + image_data[anatomy + "similarity_name"]
        elif image_data["registration_type"] == "longitudinal":
            complete_moving_name         = image_data["registered_sub_folder"] + image_data[anatomy + "rigid_name"]
        # parameters
        params                           = image_data["param_file_spline"]
        # output folder
        output_folder                    = image_data["registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Spline registration")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_moving_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["registered_sub_folder"] + "result.0.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["registered_sub_folder"] + image_data[anatomy + "spline_name"]):
                os.remove(image_data["registered_sub_folder"] + image_data[anatomy + "spline_name"])
                os.remove(image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"])
            os.rename(image_data["registered_sub_folder"] + "result.0.mha",
                      image_data["registered_sub_folder"] + image_data[anatomy + "spline_name"])
            os.rename(image_data["registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"])


    def i_rigid(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        # parameters
        params                           = image_data["i_param_file_rigid"]
        # tranformation
        transformation                   = image_data["registered_sub_folder"] + image_data[anatomy + "rigid_transf_name"]
        # output folder   
        output_folder                    = image_data["i_registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Inverting rigid transformation")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_reference_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder),
                                      "-t0",    os.path.abspath(transformation)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.i_rigid()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["i_registered_sub_folder"] + image_data[anatomy + "i_rigid_transf_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy + "i_rigid_transf_name"])
            os.rename(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["i_registered_sub_folder"] + image_data[anatomy + "i_rigid_transf_name"])


    def i_similarity(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        # parameters
        params                           = image_data["i_param_file_similarity"]
         # tranformation
        transformation                   = image_data["registered_sub_folder"] + image_data[anatomy + "similarity_transf_name"]
        # output folder
        output_folder                    = image_data["i_registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Inverting similarity transformation")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_reference_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder),
                                      "-t0",    os.path.abspath(transformation)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.i_similarity()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["i_registered_sub_folder"] + image_data[anatomy + "i_similarity_transf_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy + "i_similarity_transf_name"])
            os.rename(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["i_registered_sub_folder"] + image_data[anatomy + "i_similarity_transf_name"])


    def i_spline(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        # parameters
        params                           = image_data["i_param_file_spline"]
        # tranformation
        transformation                   = image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"]
        # output folder
        output_folder                    = image_data["i_registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Inverting spline transformation")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_reference_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder),
                                      "-t0",    os.path.abspath(transformation)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.i_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"])
            os.rename(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"])


    def t_rigid(self, image_data):
        # anatomy
        anatomy                   = image_data["current_anatomy"]
        # input mask name
        if image_data["registration_type"] == "newsubject":
            mask_to_warp          = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"]
        elif image_data["registration_type"] == "multimodal":
            mask_to_warp          = image_data["reference_folder"]        + image_data[anatomy+"levelset_mask_file_name"]
        elif image_data["registration_type"] == "longitudinal":
            mask_to_warp          = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"]

        # tranformation
        transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Rigid warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        elastix_path              = image_data["elastix_folder"]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.t_rigid()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists( image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_name"])
            os.rename(image_data["i_registered_sub_folder"] + "result.mha",
                      image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_name"])


    def t_similarity(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        # input mask name
        mask_to_warp              = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"]
        # tranformation
        transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Similarity warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.t_similarity()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"])
            os.rename(image_data["i_registered_sub_folder"] + "result.mha",
                      image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"])


    def t_spline(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        # input mask name
        mask_to_warp              = image_data["reference_folder"] + image_data[anatomy + "levelset_mask_file_name"]
        # tranformation
        transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_spline_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Spline warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output name
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.t_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"])
            os.rename(image_data["i_registered_sub_folder"] + "result.mha",
                      image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"])


    def vf_spline(self, image_data):

        # transformation
        bone                      = image_data["bone"]
        transformation            = image_data["registered_sub_folder"] + image_data[bone + "similarity_transf_name"]
        # output folder for  "deformationField.mha" (different from folder for "vector_field_name")
        # (needing 2 folders for //isation. "deformationField.mha" gets overwritten when more produced by parallel processes)
        output_folder             = image_data["registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # get vector field
        cmd = [complete_transformix_path, "-def", "all",
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output name
        if not os.path.exists(image_data["registered_sub_folder"] + "deformationField.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.vf_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            os.rename(image_data["registered_sub_folder"] + "deformationField.mha",
                      image_data["registered_folder"]    + image_data["vector_field_name"])
    
    
    def tibia_rigid(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        complete_reference_mask_dil_name = complete_reference_mask_dil_name[:-8] + 't' + complete_reference_mask_dil_name[-7:]
        complete_moving_name             = image_data["moving_folder"]    + image_data["moving_name"]
        # parameters
        params                           = image_data["param_file_rigid"]
        # output folder
        output_folder                    = image_data["registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        # print ("     Rigid registration")
        # os.path.abspath because working directory is where elastix is in pykneer, so images need whole path, not only relative
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_moving_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder)]

        subprocess.run(cmd, cwd=elastix_path)

        # check if the registration worked
        # if the registration did not work
        if not os.path.exists(image_data["registered_sub_folder"] + "result.0.mha"):
            print("-------------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.tibia_rigid()")
            print("-------------------------------------------------------------------------------------------")
            
            # print out possible errors
            # elastix not in system
            output = test_elastix()
            if output == 0:
                print ("-> elastix correctly installed")
                print ("complete_elastix_path           : " + complete_elastix_path)
                print ("complete_reference_name         : " + complete_reference_name)
                print ("complete_reference_mask_dil_name: " + complete_reference_mask_dil_name)
                print ("complete_moving_name            : " + complete_moving_name)
                print ("params                          : " + params)
                print ("output_folder                   : " + output_folder)
            else: 
                print ("-> elastix not installed. Error message is: " + output)
                print ("   You might need to set Elastix environmental variables separately. To do so, go to pyKNEEr documentation here: https://sbonaretti.github.io/pyKNEEr/faq.html#elastix")
            
            
            
            
            return
        # change output names
        else:
            tibia_name = image_data["registered_sub_folder"] + image_data[anatomy + "rigid_transf_name"]
            tibia_name = tibia_name[:-11] + 't' + tibia_name[-10:]
            if os.path.exists(image_data["registered_sub_folder"] + 't' + image_data[anatomy + "rigid_name"][1:]):
                os.remove(image_data["registered_sub_folder"] + 't' + image_data[anatomy + "rigid_name"][1:])
                os.remove(tibia_name)
            os.rename(image_data["registered_sub_folder"] + "result.0.mha",
                      image_data["registered_sub_folder"] + 't' + image_data[anatomy + "rigid_name"][1:])
            os.rename(image_data["registered_sub_folder"] + "TransformParameters.0.txt", tibia_name)


    def tibia_similarity(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"]      + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"]      + image_data[anatomy + "dil_mask_file_name"]
        complete_reference_mask_dil_name = complete_reference_mask_dil_name[:-8] + 't' + complete_reference_mask_dil_name[-7:]
        complete_moving_name             = image_data["registered_sub_folder"] + 't' + image_data[anatomy + "rigid_name"][1:]
        # parameters
        params                           = image_data["param_file_similarity"]
        # output folder
        output_folder                    = image_data["registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Similarity registration")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_moving_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["registered_sub_folder"] + "result.0.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.tibia_similarity()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            tibia_name = image_data["registered_sub_folder"] + image_data[anatomy + "similarity_transf_name"]
            tibia_name = tibia_name[:-16] + 't' + tibia_name[-15:]
            if os.path.exists(image_data["registered_sub_folder"] + 't' + image_data[anatomy + "similarity_name"][1:]):
                os.remove(image_data["registered_sub_folder"] + 't' + image_data[anatomy + "similarity_name"][1:])
                os.remove(tibia_name)
            os.rename(image_data["registered_sub_folder"] + "result.0.mha",
                      image_data["registered_sub_folder"] + 't' + image_data[anatomy + "similarity_name"][1:])            
            os.rename(image_data["registered_sub_folder"] + "TransformParameters.0.txt", tibia_name)


    def tibia_spline(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        complete_reference_mask_dil_name = complete_reference_mask_dil_name[:-8] + 't' + complete_reference_mask_dil_name[-7:]
        if image_data["registration_type"] == "newsubject" or image_data["registration_type"] == "multimodal":
            complete_moving_name         = image_data["registered_sub_folder"] + 't' + image_data[anatomy + "similarity_name"][1:]
        elif image_data["registration_type"] == "longitudinal":
            complete_moving_name         = image_data["registered_sub_folder"] + 't' + image_data[anatomy + "rigid_name"][1:]
        # parameters
        params                           = image_data["param_file_spline"]
        # output folder
        output_folder                    = image_data["registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Spline registration")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_moving_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["registered_sub_folder"] + "result.0.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.tibia_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            tibia_name = image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"]
            tibia_name = tibia_name[:-12] + 't' + tibia_name[-11:]
            if os.path.exists(image_data["registered_sub_folder"] + 't' + image_data[anatomy + "spline_name"][1:]):
                os.remove(image_data["registered_sub_folder"] + 't' + image_data[anatomy + "spline_name"][1:])
                os.remove(tibia_name)
            os.rename(image_data["registered_sub_folder"] + "result.0.mha",
                      image_data["registered_sub_folder"] + 't' + image_data[anatomy + "spline_name"][1:])            
            os.rename(image_data["registered_sub_folder"] + "TransformParameters.0.txt", tibia_name)
    
    
    def tibia_i_rigid(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        complete_reference_mask_dil_name = complete_reference_mask_dil_name[:-8] + 't' + complete_reference_mask_dil_name[-7:]
        # parameters
        params                           = image_data["i_param_file_rigid"]
        # tranformation
        transformation                   = image_data["registered_sub_folder"] + image_data[anatomy + "rigid_transf_name"]
        transformation = transformation[:-11] + 't' + transformation[-10:]
        # output folder   
        output_folder                    = image_data["i_registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Inverting rigid transformation")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_reference_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder),
                                      "-t0",    os.path.abspath(transformation)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.tibia_i_rigid()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            tibia_name = image_data["i_registered_sub_folder"] + image_data[anatomy + "i_rigid_transf_name"]
            tibia_name = tibia_name[:-11] + 't' + tibia_name[-10:]
            if os.path.exists(tibia_name):
                os.remove(tibia_name)
            os.rename(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt", tibia_name)


    def tibia_i_similarity(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        complete_reference_mask_dil_name = complete_reference_mask_dil_name[:-8] + 't' + complete_reference_mask_dil_name[-7:]
        # parameters
        params                           = image_data["i_param_file_similarity"]
         # tranformation
        transformation                   = image_data["registered_sub_folder"] + image_data[anatomy + "similarity_transf_name"]
        transformation = transformation[:-16] + 't' + transformation[-15:]
        # output folder
        output_folder                    = image_data["i_registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Inverting similarity transformation")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_reference_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder),
                                      "-t0",    os.path.abspath(transformation)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.tibia_i_similarity()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            tibia_name = image_data["i_registered_sub_folder"] + image_data[anatomy + "i_similarity_transf_name"]
            tibia_name = tibia_name[:-16] + 't' + tibia_name[-15:]
            if os.path.exists(tibia_name):
                os.remove(tibia_name)
            os.rename(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt", tibia_name)


    def tibia_i_spline(self, image_data):
        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        complete_reference_mask_dil_name = complete_reference_mask_dil_name[:-8] + 't' + complete_reference_mask_dil_name[-7:]
        # parameters
        params                           = image_data["i_param_file_spline"]
        # tranformation
        transformation                   = image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"]
        transformation = transformation[:-12] + 't' + transformation[-11:]
        # output folder
        output_folder                    = image_data["i_registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Inverting spline transformation")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_reference_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder),
                                      "-t0",    os.path.abspath(transformation)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.tibia_i_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            tibia_name = image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"]
            tibia_name = tibia_name[:-12] + 't' + tibia_name[-11:]
            if os.path.exists(tibia_name):
                os.remove(tibia_name)
            os.rename(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt", tibia_name)
    
    
    def tibia_t_rigid(self, image_data):
        # anatomy
        anatomy                   = image_data["current_anatomy"]
        # input mask name
        if image_data["registration_type"] == "newsubject":
            mask_to_warp          = image_data["i_registered_sub_folder"] + 't' + image_data[anatomy+"m_similarity_name"][1:]
        elif image_data["registration_type"] == "multimodal":
            mask_to_warp          =  image_data[anatomy+"levelset_mask_file_name"]
            mask_to_warp = mask_to_warp[:-14] + 't' + mask_to_warp[-13:]
            mask_to_warp          = image_data["reference_folder"]        + mask_to_warp
        elif image_data["registration_type"] == "longitudinal":
            mask_to_warp          = image_data["i_registered_sub_folder"] + 't' + image_data[anatomy+"m_spline_name"][1:]
        
        if image_data["registration_type"] == "newsubject":
            transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_transf_name"]
            transformation = transformation[:-11] + 't' + transformation[-10:]
        else:
            # tranformation
            transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Rigid warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        elastix_path              = image_data["elastix_folder"]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.tibia_t_rigid()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            tibia_name = image_data["i_registered_sub_folder"] + 't' + image_data[anatomy + "m_rigid_name"][1:]
            if os.path.exists(tibia_name):
                os.remove(tibia_name)
            os.rename(image_data["i_registered_sub_folder"] + "result.mha", tibia_name)


    def tibia_t_similarity(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        # input mask name
        mask_to_warp              = image_data["i_registered_sub_folder"] + 't' + image_data[anatomy+"m_spline_name"][1:]
        if image_data["registration_type"] == "newsubject":
            transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_similarity_transf_name"]
            transformation = transformation[:-16] + 't' + transformation[-15:]
        else:
            # tranformation
            transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Similarity warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.t_similarity()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            tibia_name = image_data["i_registered_sub_folder"] + 't' + image_data[anatomy+"m_similarity_name"][1:]
            if os.path.exists(tibia_name):
                os.remove(tibia_name)
            os.rename(image_data["i_registered_sub_folder"] + "result.mha", tibia_name)


    def tibia_t_spline(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        # input mask name
        mask_to_warp              = image_data[anatomy + "levelset_mask_file_name"]
        mask_to_warp = mask_to_warp[:-14] + 't' + mask_to_warp[-13:]
        mask_to_warp              = image_data["reference_folder"] + mask_to_warp
        if image_data["registration_type"] == "newsubject":
            transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_spline_transf_name"]
            transformation = transformation[:-12] + 't' + transformation[-11:]
        else:
            # tranformation
            transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy + "m_spline_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Spline warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output name
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.t_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            tibia_name = image_data["i_registered_sub_folder"] + 't' + image_data[anatomy+"m_spline_name"][1:]
            if os.path.exists(tibia_name):
                os.remove(tibia_name)
            os.rename(image_data["i_registered_sub_folder"] + "result.mha", tibia_name)

##############################################################
# centerline
################################################################
    def centerline_t_rigid(self, image_data):
        # anatomy
        anatomy                   = image_data["current_anatomy"]
        # input mask name
        if image_data["registration_type"] != "longitudinal":
            return
        centerline_to_warp = image_data["reference_folder"][:-13] + "centerline\\" + image_data['reference_name'][:-8] + "line.txt"
#         centerline_to_warp = image_data["reference_folder"][:-13] + "centerline\\" + image_data['fmask'][:-10] + "line.txt"
        # tranformation
        transformation            = image_data["registered_sub_folder"] + image_data[anatomy + "rigid_transf_name"]
        # output folder
        
        output_folder             = image_data["registered_sub_folder"] + "centerline\\"
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        # transformix path
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print (" Rigid warping")
        cmd = [complete_transformix_path, "-def",  os.path.abspath(centerline_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        elastix_path              = image_data["elastix_folder"]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(output_folder + "outputpoints.txt"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in bone.centerline_t_rigid()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
#             os.rename(output_folder + "outputpoints.txt", output_folder + "registered_line.txt")
            center_folder             = image_data["segmented_folder"][:-10] + "centerline\\"
            modify_output_line(output_folder + "outputpoints.txt", center_folder+image_data[anatomy+'mask'][:-10] + "line.txt")
     
            
# ---------------------------------------------------------------------------------------------------------------------------
# CARTILAGE INSTANCE --------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

class cartilage (registration):

    def rigid(self, image_data):
        pass

    def similarity(self, image_data):
        pass

    def spline(self, image_data):

        # anatomy
        anatomy                        = image_data["current_anatomy"]
        bone                           = image_data["bone"]
        # input image names
        complete_reference_name        = image_data["reference_folder"]     + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"]     + image_data[anatomy + "dil_mask_file_name"]
        if image_data["registration_type"] == "newsubject" or image_data["registration_type"] == "multimodal":
            complete_moving_name       = image_data["registered_sub_folder"] + image_data[bone + "similarity_name"]
        elif image_data["registration_type"] == "longitudinal":
            complete_moving_name       = image_data["registered_sub_folder"] + image_data[bone + "rigid_name"]
        # parameters
        params                         = image_data["param_file_spline"]
        # output folder
        output_folder                  = image_data["registered_sub_folder"]
        # elastix path
        elastix_path                   = image_data["elastix_folder"]
        complete_elastix_path          = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Spline registration")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_moving_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["registered_sub_folder"] + "result.0.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in cartilage.spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["registered_sub_folder"] + image_data[anatomy + "spline_name"]):
                os.remove(image_data["registered_sub_folder"] + image_data[anatomy + "spline_name"])
                os.remove(image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"])
            os.rename(image_data["registered_sub_folder"] + "result.0.mha",
                      image_data["registered_sub_folder"] + image_data[anatomy + "spline_name"])
            os.rename(image_data["registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"])


    def i_rigid(self, image_data):
        pass

    def i_similarity(self, image_data):
        pass

    def i_spline(self, image_data):

        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        # parameters
        params                           = image_data["i_param_file_spline"]
        # tranformation
        transformation                   = image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"]
        # output folder
        output_folder                    = image_data["i_registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Inverting spline transformation")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_reference_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder),
                                      "-t0",    os.path.abspath(transformation)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in cartilage.i_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"])
            os.rename(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"])


    def t_rigid(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        bone                      = image_data["bone"]
        # input mask name
        if image_data["registration_type"] == "newsubject":
            mask_to_warp          = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"]
        elif image_data["registration_type"] == "multimodal":
            mask_to_warp          = image_data["reference_folder"]        + image_data[anatomy+"levelset_mask_file_name"]
        elif image_data["registration_type"] == "longitudinal":
            mask_to_warp          = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"]
        # tranformation
        transformation            = image_data["i_registered_sub_folder"] + image_data[bone+"m_rigid_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Rigid warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in cartilage.t_rigid()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_name"])
            os.rename(image_data["i_registered_sub_folder"] + "result.mha",
                      image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_name"])


    def t_similarity(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        bone                      = image_data["bone"]
        # input mask name
        mask_to_warp              = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"]
        # tranformation
        transformation            = image_data["i_registered_sub_folder"] + image_data[bone+"m_similarity_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Similarity warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in cartilage.t_similarity()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"])
            os.rename(image_data["i_registered_sub_folder"] + "result.mha",
                      image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"])


    def t_spline(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        # input mask name
        mask_to_warp              = image_data["reference_folder"]        + image_data[anatomy+"levelset_mask_file_name"]
        # tranformation
        transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Spline warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in cartilage.t_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            if os.path.exists(image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"]):
                os.remove(image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"])
            os.rename(image_data["i_registered_sub_folder"] + "result.mha",
                      image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"])


    def vf_spline(self):
        pass



# ---------------------------------------------------------------------------------------------------------------------------
# VESSEL INSTANCE --------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------

class vessel (registration):

    def rigid(self, image_data):
        pass

    def similarity(self, image_data):
        pass

    def spline(self, image_data):

        # anatomy
        anatomy                        = image_data["current_anatomy"]
        bone                           = image_data["bone"]
        # input image names
        complete_reference_name        = image_data["reference_folder"]     + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"]     + image_data[anatomy + "dil_mask_file_name"]
        if image_data["registration_type"] == "newsubject" or image_data["registration_type"] == "multimodal":
            complete_moving_name       = image_data["registered_sub_folder"] + image_data[bone + "similarity_name"]
        elif image_data["registration_type"] == "longitudinal":
            complete_moving_name       = image_data["registered_sub_folder"] + image_data[bone + "rigid_name"]
        # parameters
        params                         = image_data["param_file_spline"]
        # output folder
        output_folder                  = image_data["registered_sub_folder"]
        # elastix path
        elastix_path                   = image_data["elastix_folder"]
        complete_elastix_path          = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Spline registration")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_moving_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["registered_sub_folder"] + "result.0.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in vessel.spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            os.rename(image_data["registered_sub_folder"] + "result.0.mha",
                      image_data["registered_sub_folder"] + image_data[anatomy + "spline_name"])
            os.rename(image_data["registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"])


    def i_rigid(self, image_data):
        pass

    def i_similarity(self, image_data):
        pass

    def i_spline(self, image_data):

        # anatomy
        anatomy                          = image_data["current_anatomy"]
        # input image names
        complete_reference_name          = image_data["reference_folder"] + image_data["reference_name"]
        complete_reference_mask_dil_name = image_data["reference_folder"] + image_data[anatomy + "dil_mask_file_name"]
        # parameters
        params                           = image_data["i_param_file_spline"]
        # tranformation
        transformation                   = image_data["registered_sub_folder"] + image_data[anatomy + "spline_transf_name"]
        # output folder
        output_folder                    = image_data["i_registered_sub_folder"]
        # elastix path
        elastix_path                     = image_data["elastix_folder"]
        complete_elastix_path            = image_data["complete_elastix_path"]

        # execute registration
        #print ("     Inverting spline transformation")
        cmd = [complete_elastix_path, "-f",     os.path.abspath(complete_reference_name),
                                      "-fMask", os.path.abspath(complete_reference_mask_dil_name),
                                      "-m",     os.path.abspath(complete_reference_name),
                                      "-p",     os.path.abspath(params),
                                      "-out",   os.path.abspath(output_folder),
                                      "-t0",    os.path.abspath(transformation)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in cartilage.i_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            os.rename(image_data["i_registered_sub_folder"] + "TransformParameters.0.txt",
                      image_data["i_registered_sub_folder"] + image_data[anatomy + "i_spline_transf_name"])


    def t_rigid(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        bone                      = image_data["bone"]
        # input mask name
        if image_data["registration_type"] == "newsubject":
            mask_to_warp          = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"]
        elif image_data["registration_type"] == "multimodal":
            mask_to_warp          = image_data["reference_folder"]        + image_data[anatomy+"levelset_mask_file_name"]
        elif image_data["registration_type"] == "longitudinal":
            mask_to_warp          = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"]
        # tranformation
        transformation            = image_data["i_registered_sub_folder"] + image_data[bone+"m_rigid_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Rigid warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in cartilage.t_rigid()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            os.rename(image_data["i_registered_sub_folder"] + "result.mha",
                      image_data["i_registered_sub_folder"] + image_data[anatomy + "m_rigid_name"])


    def t_similarity(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        bone                      = image_data["bone"]
        # input mask name
        mask_to_warp              = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"]
        # tranformation
        transformation            = image_data["i_registered_sub_folder"] + image_data[bone+"m_similarity_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Similarity warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in cartilage.t_similarity()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            os.rename(image_data["i_registered_sub_folder"] + "result.mha",
                      image_data["i_registered_sub_folder"] + image_data[anatomy+"m_similarity_name"])


    def t_spline(self, image_data):

        # anatomy
        anatomy                   = image_data["current_anatomy"]
        # input mask name
        mask_to_warp              = image_data["reference_folder"]        + image_data[anatomy+"levelset_mask_file_name"]
        # tranformation
        transformation            = image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_transf_name"]
        # output folder
        output_folder             = image_data["i_registered_sub_folder"]
        # transformix path
        elastix_path              = image_data["elastix_folder"]
        complete_transformix_path = image_data["complete_transformix_path"]

        # execute transformation
        #print ("     Spline warping")
        cmd = [complete_transformix_path, "-in",  os.path.abspath(mask_to_warp),
                                          "-tp",  os.path.abspath(transformation),
                                          "-out", os.path.abspath(output_folder)]
        subprocess.run(cmd, cwd=elastix_path)

        # change output names
        if not os.path.exists(image_data["i_registered_sub_folder"] + "result.mha"):
            print("----------------------------------------------------------------------------------------")
            print("ERROR: No output created in cartilage.t_spline()")
            print("----------------------------------------------------------------------------------------")
            return
        else:
            os.rename(image_data["i_registered_sub_folder"] + "result.mha",
                      image_data["i_registered_sub_folder"] + image_data[anatomy+"m_spline_name"])


    def vf_spline(self):
        pass




# ---------------------------------------------------------------------------------------------------------------------------
# TESTING POSSIBLE ERRORS ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------
def test_elastix():

    sys = platform.system()
    
    # get the folder depending on the OS
    if sys == "Darwin":
        dirpath = pkg_resources.resource_filename('pykneer','elastix/Darwin/')
    elif sys == "Linux":
        dirpath = pkg_resources.resource_filename('pykneer','elastix/Linux/')
    elif sys == "Windows":
        dirpath = pkg_resources.resource_filename('pykneer','elastix\\Windows\\')
    
    # create the full path
    if sys == "Darwin" or sys == "Linux": 
        completeElastixPath     = dirpath + "elastix"
    elif sys == "Windows": 
        completeElastixPath     = dirpath + "elastix.exe"
    
    # call elastix to see if it reponds
    cmd = [completeElastixPath]
    output = subprocess.call(cmd, cwd=dirpath)
    
    return output
    

    
