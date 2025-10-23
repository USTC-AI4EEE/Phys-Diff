## [ICASSP 2026] Phys-Diff: A Physics-Inspired Latent Diffusion Model for Tropical Cyclone Forecasting

> **Authors:**

Lei Liu, Xiaoning Yu, Kang Chen, Jiahui Huang, Tengyuan Liu, Hongwei Zhao, Bin Li.

## 1. Abstract

Tropical Cyclone (TC) forecasting is crucial for disaster warning and emergency response. Recently, Deep Learning (DL) methods have been widely explored to address computational challenges in this domain. However, existing methods often model cyclone attributes independently, ignoring their physical relationships. This leads to predictions lacking physical consistency, particularly in long-term forecasts. To address this, we propose Phys-Diff, a physics-inspired latent diffusion model for TC forecasting. Specifically, Phys-Diff firstly proposes to disentangle the latent features into three parts for different TC attributes (i.e., trajectory, pressure, and wind speed). Then, Phys-Diff introduces a cross-task attention mechanism, enabling features from different attributes to attend and capture their inherent relationships. By incorporating these dependencies, Phys-Diff effectively enhances the physical consistency of learned features. Moreover, Phys-Diff integrates multimodal data—including historical cyclone attributes, ERA5 reanalysis data, and FengWu forecast fields—into the Transformer encoder-decoder architecture, thus generating the features with comprehensive environmental information to further improve the effectiveness for TC forecasting. Experiments show that Phys-Diff outperforms existing methods, especially in long-term forecasting, reducing the 120-hour wind speed forecast error by up to 65.7% against state-of-the-art models on the global dataset.

## 2. Requirements

```bash
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.21.0
numpy==1.24.3
scipy>=1.8.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
timm>=0.6.0
einops>=0.4.0
h5py>=3.6.0
netCDF4>=1.5.7
xarray>=0.19.0
pillow>=8.3.0
onnxruntime>=1.12.0
pyyaml>=6.0
tqdm>=4.62.0
tensorboard>=2.8.0
umap-learn>=0.5.3
scikit-image>=0.18.0
cdsapi>=0.5.1
metpy>=1.3.0
```

## 3. Datasets

The datasets and models required for this experiment can be accessed through the following links:

ERA5 Dataset: The fifth-generation reanalysis data provided by the European Centre for Medium-Range Weather Forecasts (ECMWF). It can be downloaded from [this page](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5).

FengWu Model: A meteorological forecasting model open-sourced by OpenEarthLab. The code and related documentation are available on [GitHub](https://github.com/OpenEarthLab/FengWu). Download the **fengwu.onnx** model file and place it in the `./scripts` directory.

BTrACS Dataset (Best Track): The International Best Track Archive for Climate Stewardship (IBTrACS) provided by the National Centers for Environmental Information (NCEI). It can be accessed through [this link](https://www.ncei.noaa.gov/products/international-best-track-archive).

The preprocessing tool for generating FengWu model forecast fields is located in `./utils/precompute_fengwu_forecasts.py`. Before running this script, approximately 17.16 TB of ERA5 data should be prepared. Model parameters can be adjusted by modifying the corresponding `.yaml` configuration files, and default settings are provided in the code for reference.


## 4. Usage

- an example for train and evaluate a new model：

```bash
# Train full Phys-Diff model
python scripts/train.py --config configs/config.yaml

# Train Phys-Diff without FengWu
python scripts/train.py --config configs/config_wo_fengwu.yaml
```

- You can get the following output:

```bash
(Phys-Diff) yuxiaoning@user-SYS-420GP-TNR:~/workspace/workspace/Phys-Diff$ python scripts/train.py --config configs/config_wo_fengwu.yaml
2025-09-10 19:57:51,681 - __main__ - INFO - Starting training with relative coordinate normalization
2025-09-10 19:57:51,870 - __main__ - INFO - Using GPU 4: NVIDIA GeForce RTX 4090 D
2025-09-10 19:58:12,834 - dataset.dataset - INFO - Loaded 153973 valid IBTrACS records
2025-09-10 19:58:12,852 - dataset.dataset - INFO - Applied 6-hour time resolution filter
2025-09-10 19:58:12,852 - dataset.dataset - INFO - Filtered to 56391 records for years [1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017] at 6h resolution
2025-09-10 19:58:13,991 - dataset.dataset - INFO - Processing storm 100/2002: 1989238N22253
2025-09-10 19:58:15,409 - dataset.dataset - INFO - Processing storm 200/2002: 1992281N11263
2025-09-10 19:58:16,553 - dataset.dataset - INFO - Processing storm 300/2002: 1996237N14339
2025-09-10 19:58:17,775 - dataset.dataset - INFO - Processing storm 400/2002: 2000222N28286
2025-09-10 19:58:18,761 - dataset.dataset - INFO - Processing storm 500/2002: 2001329S06095
2025-09-10 19:58:19,694 - dataset.dataset - INFO - Processing storm 600/2002: 2002356S07070
2025-09-10 19:58:21,106 - dataset.dataset - INFO - Processing storm 700/2002: 2003359S15177
2025-09-10 19:58:22,259 - dataset.dataset - INFO - Processing storm 800/2002: 2005013N05153
2025-09-10 19:58:23,585 - dataset.dataset - INFO - Processing storm 900/2002: 2006007S15182
2025-09-10 19:58:24,679 - dataset.dataset - INFO - Processing storm 1000/2002: 2007052S10093
2025-09-10 19:58:25,718 - dataset.dataset - INFO - Processing storm 1100/2002: 2008083S12102
2025-09-10 19:58:27,069 - dataset.dataset - INFO - Processing storm 1200/2002: 2009104N13088
2025-09-10 19:58:28,251 - dataset.dataset - INFO - Processing storm 1300/2002: 2010167N15265
2025-09-10 19:58:29,260 - dataset.dataset - INFO - Processing storm 1400/2002: 2011231N15278
2025-09-10 19:58:30,784 - dataset.dataset - INFO - Processing storm 1500/2002: 2012256N15257
2025-09-10 19:58:31,742 - dataset.dataset - INFO - Processing storm 1600/2002: 2013274N18152
2025-09-10 19:58:32,882 - dataset.dataset - INFO - Processing storm 1700/2002: 2014290N14261
2025-09-10 19:58:34,021 - dataset.dataset - INFO - Processing storm 1800/2002: 2015286S10175
2025-09-10 19:58:35,576 - dataset.dataset - INFO - Processing storm 1900/2002: 2016323N13279
2025-09-10 19:58:36,450 - dataset.dataset - INFO - Processing storm 2000/2002: 2017360S15124
2025-09-10 19:58:36,454 - dataset.dataset - INFO - Loaded coordinate statistics from: dataset/coord_stats.json
2025-09-10 19:58:36,454 - dataset.dataset - INFO - Loaded intensity statistics from: dataset/intensity_stats.json
2025-09-10 19:58:36,454 - dataset.dataset - INFO - Loaded environment channel statistics from: dataset/env_channel_stats.json
2025-09-10 19:58:36,454 - dataset.dataset - INFO - Created train dataset with 40216 sequences
2025-09-10 19:58:36,454 - dataset.dataset - INFO - Coord stats - lat_std: 0.4745, lon_std: 0.7570
2025-09-10 19:58:36,454 - dataset.dataset - INFO - Intensity stats - wind: 49.89±28.09 kt, pres: 989.73±21.46 hPa
2025-09-10 19:58:36,455 - dataset.dataset - INFO - FengWu integration: disabled
2025-09-10 19:58:54,398 - dataset.dataset - INFO - Loaded 153973 valid IBTrACS records
2025-09-10 19:58:54,404 - dataset.dataset - INFO - Applied 6-hour time resolution filter
2025-09-10 19:58:54,404 - dataset.dataset - INFO - Filtered to 3563 records for years [2018] at 6h resolution
2025-09-10 19:58:55,463 - dataset.dataset - INFO - Processing storm 100/113: 2018309S06068
2025-09-10 19:58:55,576 - dataset.dataset - INFO - Loaded coordinate statistics from: dataset/coord_stats.json
2025-09-10 19:58:55,576 - dataset.dataset - INFO - Loaded intensity statistics from: dataset/intensity_stats.json
2025-09-10 19:58:55,577 - dataset.dataset - INFO - Loaded environment channel statistics from: dataset/env_channel_stats.json
2025-09-10 19:58:55,577 - dataset.dataset - INFO - Created val dataset with 2519 sequences
2025-09-10 19:58:55,577 - dataset.dataset - INFO - Coord stats - lat_std: 0.4745, lon_std: 0.7570
2025-09-10 19:58:55,577 - dataset.dataset - INFO - Intensity stats - wind: 49.89±28.09 kt, pres: 989.73±21.46 hPa
2025-09-10 19:58:55,577 - dataset.dataset - INFO - FengWu integration: disabled
2025-09-10 19:59:14,043 - dataset.dataset - INFO - Loaded 153973 valid IBTrACS records
2025-09-10 19:59:14,051 - dataset.dataset - INFO - Applied 6-hour time resolution filter
2025-09-10 19:59:14,051 - dataset.dataset - INFO - Filtered to 8757 records for years [2019, 2020, 2021] at 6h resolution
2025-09-10 19:59:15,053 - dataset.dataset - INFO - Processing storm 100/316: 2019335N05058
2025-09-10 19:59:15,881 - dataset.dataset - INFO - Processing storm 200/316: 2020303N05149
2025-09-10 19:59:17,436 - dataset.dataset - INFO - Processing storm 300/316: 2021267N31298
2025-09-10 19:59:17,577 - dataset.dataset - INFO - Loaded coordinate statistics from: dataset/coord_stats.json
2025-09-10 19:59:17,577 - dataset.dataset - INFO - Loaded intensity statistics from: dataset/intensity_stats.json
2025-09-10 19:59:17,577 - dataset.dataset - INFO - Loaded environment channel statistics from: dataset/env_channel_stats.json
2025-09-10 19:59:17,577 - dataset.dataset - INFO - Created test dataset with 6193 sequences
2025-09-10 19:59:17,577 - dataset.dataset - INFO - Coord stats - lat_std: 0.4745, lon_std: 0.7570
2025-09-10 19:59:17,577 - dataset.dataset - INFO - Intensity stats - wind: 49.89±28.09 kt, pres: 989.73±21.46 hPa
2025-09-10 19:59:17,577 - dataset.dataset - INFO - FengWu integration: disabled
2025-09-10 19:59:17,577 - dataset.dataset - INFO - === NORMALIZATION STATISTICS (from precomputed files) ===
2025-09-10 19:59:17,577 - dataset.dataset - INFO -   Wind: 49.89±28.09 kt
2025-09-10 19:59:17,577 - dataset.dataset - INFO -   Pressure: 989.73±21.46 hPa
2025-09-10 19:59:17,577 - dataset.dataset - INFO -   Latitude std: 0.4745
2025-09-10 19:59:17,577 - dataset.dataset - INFO -   Longitude std: 0.7570
2025-09-10 19:59:17,578 - __main__ - INFO - Train samples: 40216, Validation samples: 2519, Test samples: 6193
/home/yuxiaoning/miniconda3/envs/Phys-Diff/lib/python3.10/site-packages/torch/nn/modules/transformer.py:392: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
2025-09-10 19:59:17,741 - __main__ - INFO - Using uncertainty-weighted multi-task learning
2025-09-10 19:59:17,741 - __main__ - INFO - Uncertainty parameters: 4
2025-09-10 19:59:18,345 - __main__ - INFO - Early stopping enabled: patience=2, min_delta=0.001
2025-09-10 19:59:18,345 - __main__ - INFO - Starting training...
2025-09-10 19:59:18,345 - __main__ - INFO - Epoch 0/30
Epoch 0 Training:   8%|██████▊                                                                            | 52/629 [00:52<07:01,  4.37it/s, Loss=4.2774, Diff=1.0241, Coord=5.6435, MSW=0.4577, MLSP=0.4778]
```

## 5. Acknowledgments

We appreciate the following open-sourced repositories for their valuable code base:

- MSCAR (2024): https://github.com/1457756434/MSCAR
- VQLTI (2025): https://github.com/1457756434/VQLTI
- MMSTN (2022): https://github.com/Zjut-MultimediaPlus/MMSTN
- MGTCF (2023): https://github.com/Zjut-MultimediaPlus/MGTCF
- TC-Diffuser (2025): https://github.com/Zjut-MultimediaPlus/TC-Diffuser



## 6. Citation

If you find our work useful in your research, please consider citing:

```latex

```

If you have any problems, contact me via liulei13@ustc.edu.cn.
