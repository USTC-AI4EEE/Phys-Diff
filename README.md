## [ICASSP 2026] Phys-Diff: A Physics-Inspired Latent Diffusion Model for Tropical Cyclone Forecasting

English | [中文](README_zh.md)

> **Authors:**

Lei Liu, Xiaoning Yu, Kang Chen, Jiahui Huang, Tengyuan Liu, Hongwei Zhao, Bin Li.

Our paper has been accepted by ICASSP 2026.

Website: https://arxiv.org/abs/2603.00521 .

> **Phys-Diff v2** (2026.11)

## 1. Abstract

Tropical cyclone (TC) forecasting is critical for disaster warning and emergency response. Deep learning methods address computational challenges but often neglect physical relationships between TC attributes, resulting in predictions lacking physical consistency. To address this, we propose Phys-Diff, a physics-inspired latent diffusion model that disentangles latent features into task-specific components (trajectory, pressure, wind speed) and employs cross-task attention to introduce prior physics-inspired inductive biases, thereby embedding physically consistent dependencies among TC attributes. Phys-Diff integrates multimodal data including historical cyclone attributes, ERA5 reanalysis data, and FengWu forecast fields via a Transformer encoder-decoder architecture, further enhancing forecasting performance. Experiments demonstrate state-of-the-art performance on global and regional datasets.

## 2. Requirements

```bash
numpy>=1.24
torch>=2.5
```

```bash
pip install -r requirements.txt
pip install -e .        # optional: makes `import physdiff` available everywhere
```

## 3. Datasets

The datasets and models required for this experiment can be accessed through the following links:

ERA5 Dataset: The fifth-generation reanalysis data provided by the European Centre for Medium-Range Weather Forecasts (ECMWF). It can be downloaded from [this page](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5).

FengWu Model: A meteorological forecasting model open-sourced by OpenEarthLab. The code and related documentation are available on [GitHub](https://github.com/OpenEarthLab/FengWu).

IBTrACS Dataset (Best Track): The International Best Track Archive for Climate Stewardship (IBTrACS) provided by the National Centers for Environmental Information (NCEI). It can be accessed through [this link](https://www.ncei.noaa.gov/products/international-best-track-archive).

The data is split chronologically: train 1980–2017 (32,721 windows), validation 2018 (2,307) and test 2019–2022 (6,699). The prepared tensors that the loader expects (`PREP`, `GEOM`, `FINE` below) are specified in [docs/DATA.md](docs/DATA.md).

Released models (`configs/`, `checkpoints/`, and the 50-member test ensembles in `members/`):

| Variant     | Inputs                            | Config                   | Checkpoint                  |
|-------------|-----------------------------------|--------------------------|-----------------------------|
| `main`      | IBTrACS + ERA5 + FengWu           | `configs/main.json`      | `checkpoints/main.pth`      |
| `wo_fengwu` | IBTrACS + ERA5 (no FengWu fields) | `configs/wo_fengwu.json` | `checkpoints/wo_fengwu.pth` |

## 4. Usage

- build `PREP` / `FINE` from IBTrACS and (69, 80, 80) ERA5 crops (see [docs/DATA.md](docs/DATA.md#building-prep-and-fine)):

```bash
python tools/prepare.py --ibtracs ibtracs.ALL.list.v04r00.csv --crops TC_ERA5 --splits test --data PREP --fine-root FINE
```

- inference with a released checkpoint and evaluation:

```bash
python tools/infer.py --variant main --data PREP --geometry GEOM --fine-root FINE --output outputs/main_test.npz
python tools/evaluate.py outputs/main_test.npz --reference members/main.npz

python tools/infer.py --variant wo_fengwu --data PREP --fine-root FINE --output outputs/wo_fengwu_test.npz
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
@inproceedings{liu2026phys,
  title={Phys-Diff: A physics-inspired latent diffusion model for tropical cyclone forecasting},
  author={Liu, Lei and Yu, Xiaoning and Chen, Kang and Huang, Jiahui and Liu, Tengyuan and Zhao, Hongwei and Li, Bin},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3536--3540},
  year={2026},
  organization={IEEE}
}
```

If you have any problems, contact me via liulei13@ustc.edu.cn.
