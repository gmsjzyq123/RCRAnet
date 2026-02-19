# Diversity-Driven Domain Generalization for Hyperspectral Image via Rank-Increased Attention Fusion.

```
## Datasets

The following datasets are used for experiments:

- **Houston**: Contains hyperspectral images from the Houston area, including various land cover types. Files:
  - `Houston13.mat`: Dataset for Houston13.
  - `Houston13_7gt.mat`: Ground truth data for Houston13.
  - `Houston18.mat`: Dataset for Houston18.
  - `Houston18_7gt.mat`: Ground truth data for Houston18.

- **Pavia**: Includes hyperspectral images from the Pavia region, commonly used for classification. Files:
  - `paviaC.mat`: Dataset for Pavia Center.
  - `paviaC_7gt.mat`: Ground truth data for Pavia Center.
  - `paviaU.mat`: Dataset for Pavia University.
  - `paviaU_7gt.mat`: Ground truth data for Pavia University.

- HyRANK: A cross-scene hyperspectral dataset designed for domain generalization and land-cover classification tasks, consisting of two geographically distinct scenes:
  - `Dioni.mat: Hyperspectral image of the Dioni scene.
  - `Dioni_gt_out68.mat: Corresponding ground truth labels for Dioni.
 - `Loukia.mat: Hyperspectral image of the Loukia scene.
 - `Loukia_gt_out68.mat: Corresponding ground truth labels for Loukia.

File Structure:
```

## Requirements
- **Python**: 3.8.19  
- **PyTorch**: 1.11.0  
- **CUDA**: 11.3  


```
datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
└── Pavia
│   ├── paviaC.mat
│   └── paviaC_7gt.mat
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
└── HyRANK
│   ├── Dioni.mat
│   └── Dioni_gt_out68.mat
│   ├── Loukia.mat
└── └── Loukia_gt_out68.mat
```
   
   
