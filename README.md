# Longitudinal Registration of Knee MRI Based on Femoral and Tibial Alignment

## Introduction

We develop an automatic and robust algorithm for longitudinal registration of knee Magnetic Resonance Imaging (MRI) across the time span of eight years. This algorithm firstly achieves rigid registration based on femoral segmentation, and then makes evaluations based on tibial alignment.

It's the code of this article. Codebase is from pyKNEEr research [1].

## File Structure
`pykneer-yg/`  
pykneer codebase

`pykneer-yg/pykneer/notebooks/main.ipynb`  
application entry point

`annotated_tibia/`  
code and result of generating tibial mask of sample case

## Reference
[1] Bonaretti, Serena, Garry E. Gold, and Gary S. Beaupre. "pyKNEEr: An image analysis workflow for open and reproducible research on femoral knee cartilage." Plos one15.1 (2020): e0226501.
