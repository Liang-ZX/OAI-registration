# Longitudinal Registration of Knee MRI Based on Femoral and Tibial Alignment

This is the official code for **ISMRM 2021** abstract paper 

[Longitudinal Registration of Knee MRI Based on Femoral and Tibial Alignment](https://archive.ismrm.org/2021/3734.html) by 

[Zhixuan Liang](https://liang-zx.github.io/), Yin Guo, and [Chun Yuan](https://scholar.google.com/citations?user=ujKJ-w4AAAAJ&hl=en&oi=ao).

For full details, please check out our [paper link](https://cds.ismrm.org/protected/21MProceedings/PDFfiles/3734.html).

## Introduction

We develop an automatic and robust algorithm for longitudinal registration of knee Magnetic Resonance Imaging (MRI) across the time span of eight years. This algorithm firstly achieves rigid registration based on femoral segmentation, and then makes evaluations based on tibial alignment.

Here's the code of this article. Codebase refers to the *pyKNEEr* research [[1]](#refer-anchor-1).

![rigid registration](https://user-images.githubusercontent.com/42173433/112028657-5f38f400-8b73-11eb-9104-a7680bd02e52.png)

## File Structure
```
└── OAI-registration
    ├── pykneer-yg          # pykneer codebase
    │   └── pykneer
    │         └── notebooks     
    │             ├── main.ipynb    # application entry point
    │             └── ...
    └── annotated_tibia     # code and result of generating tibial mask of sample case
```

## Acknowledgement
<div id="refer-anchor-1"></div>

- Bonaretti, Serena, Garry E. Gold, and Gary S. Beaupre. "pyKNEEr: An image analysis workflow for open and reproducible research on femoral knee cartilage." Plos one15.1 (2020): e0226501.

## Citation
To cite our work, you can use the following bibtex
```
@article{lianglongitudinal,
  title={Longitudinal Registration of Knee MRI Based on Femoral and Tibial Alignment},
  author={Liang, Zhixuan and Guo, Yin and Yuan, Chun},
}
```
