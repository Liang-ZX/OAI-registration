# Manually annotate the tibial mask of the sample case

pyKNEEr demo doesn’t contain tibial mask, so we draw it by ourself.

## Method
- Draw Regions Of Interest (ROI) of some sagittal slices of the reference at certain interval
  - using MATLAB *roipoly()* function
- Utilize the *nearest neighbor interpolation* to make the slices continuous, then we will obtain the tibial mask with the *same spacing and direction* as reference.

## Code
- `generate_reference_t.ipynb`

## Result
![sample case](https://github.com/Liang-ZX/OAI-registration/blob/master/annotated_tibia/img/ref_img.png)

![tibial mask](https://github.com/Liang-ZX/OAI-registration/blob/master/annotated_tibia/img/tibia_mask.png)


