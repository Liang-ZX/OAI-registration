import pydicom
import os,re
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import math
import sys

import registration_utilities as ru
import registration_callbacks as rc
import AssistVisualization as vis
import cv2


# rigid registration
def rigid_registration(fixed_image, moving_image, initial_transform=None):
    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetMetricAsMeanSquares()

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1000, convergenceMinimumValue=1e-10, convergenceWindowSize=50)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, vis.start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, vis.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, vis.update_multires_iterations) 
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: vis.plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                   sitk.Cast(moving_image, sitk.sitkFloat32))
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return final_transform


# affine registration
def affine_registration(fixed_image, moving_image, initial_transform=None):
    dimension = fixed_image.GetDimension()
    if initial_transform is None:
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.AffineTransform(dimension), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
    else:
        initial_transform.AddTransform(sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.AffineTransform(dimension), 
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY))
    
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=500)
    # registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    # registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetMetricAsMeanSquares()

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1000, convergenceMinimumValue=1e-10, convergenceWindowSize=50)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    registration_method.AddCommand(sitk.sitkStartEvent, vis.start_plot)
    registration_method.AddCommand(sitk.sitkEndEvent, vis.end_plot)
    registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, vis.update_multires_iterations) 
    registration_method.AddCommand(sitk.sitkIterationEvent, lambda: vis.plot_values(registration_method))

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                   sitk.Cast(moving_image, sitk.sitkFloat32))
    print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    return final_transform