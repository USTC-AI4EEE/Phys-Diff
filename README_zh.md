## [ICASSP 2026] Phys-Diff：一种物理启发的热带气旋预报潜空间扩散模型

[English](README.md) | 中文

> **作者：**

Lei Liu, Xiaoning Yu, Kang Chen, Jiahui Huang, Tengyuan Liu, Hongwei Zhao, Bin Li.

本文已被 ICASSP 2026 录用。

论文地址：https://arxiv.org/abs/2603.00521 。

> **Phys-Diff v2**（2026.11）

## 1. 摘要

热带气旋（TC）预报对灾害预警和应急响应至关重要。深度学习方法缓解了计算上的挑战，但往往忽略热带气旋各属性之间的物理联系，导致预测缺乏物理一致性。为此，我们提出 Phys-Diff，一种物理启发的潜空间扩散模型：它将潜在特征解耦为分别对应路径、气压和风速的任务特定分量，并通过跨任务注意力引入物理启发的先验归纳偏置，从而在热带气旋各属性之间嵌入物理一致的依赖关系。Phys-Diff 通过 Transformer 编码器-解码器结构融合历史气旋属性、ERA5 再分析数据和 FengWu 预报场等多模态数据，进一步提升预报性能。实验表明，Phys-Diff 在全球和区域数据集上均达到了最先进的性能。

## 2. 环境依赖

```bash
numpy>=1.24
torch>=2.5
```

```bash
pip install -r requirements.txt
pip install -e .        # 可选：在任意位置都能 `import physdiff`
```

## 3. 数据集

本实验所需的数据集和模型可通过以下链接获取：

ERA5 数据集：欧洲中期天气预报中心（ECMWF）提供的第五代再分析数据，可从[该页面](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)下载。

FengWu 模型：OpenEarthLab 开源的气象预报模型，代码及相关文档见 [GitHub](https://github.com/OpenEarthLab/FengWu)。

IBTrACS 数据集（最佳路径）：美国国家环境信息中心（NCEI）提供的国际气候管理最佳路径档案（IBTrACS），可通过[该链接](https://www.ncei.noaa.gov/products/international-best-track-archive)获取。

数据按时间划分：训练集 1980–2017（32,721 个窗口），验证集 2018（2,307 个），测试集 2019–2022（6,699 个）。数据加载器所需的预处理张量（下文的 `PREP`、`GEOM`、`FINE`）格式见 [docs/DATA.md](docs/DATA.md)。

发布的模型（`configs/`、`checkpoints/`，以及 `members/` 中的 50 成员测试集集合预报）：

| 变体        | 输入                              | 配置                     | Checkpoint                  |
|-------------|-----------------------------------|--------------------------|-----------------------------|
| `main`      | IBTrACS + ERA5 + FengWu           | `configs/main.json`      | `checkpoints/main.pth`      |
| `wo_fengwu` | IBTrACS + ERA5（不使用 FengWu 场）| `configs/wo_fengwu.json` | `checkpoints/wo_fengwu.pth` |

## 4. 使用方法

- 由 IBTrACS 和 (69, 80, 80) ERA5 裁剪场生成 `PREP` / `FINE`（见 [docs/DATA.md](docs/DATA.md#building-prep-and-fine)）：

```bash
python tools/prepare.py --ibtracs ibtracs.ALL.list.v04r00.csv --crops TC_ERA5 --splits test --data PREP --fine-root FINE
# 仅 main 模型：FengWu 未来步与 GEOM
python tools/prepare_fengwu.py --model fengwu_v1.onnx --era5 ERA5 --ibtracs ibtracs.ALL.list.v04r00.csv --splits test --data PREP --fine-root FINE --geometry GEOM
```

- 使用发布的 checkpoint 推理并评估：

```bash
python tools/infer.py --variant main --data PREP --geometry GEOM --fine-root FINE --output outputs/main_test.npz
python tools/evaluate.py outputs/main_test.npz --reference members/main.npz

python tools/infer.py --variant wo_fengwu --data PREP --fine-root FINE --output outputs/wo_fengwu_test.npz
```

## 5. 致谢

感谢以下开源仓库提供的宝贵代码：

- MSCAR (2024): https://github.com/1457756434/MSCAR
- VQLTI (2025): https://github.com/1457756434/VQLTI
- MMSTN (2022): https://github.com/Zjut-MultimediaPlus/MMSTN
- MGTCF (2023): https://github.com/Zjut-MultimediaPlus/MGTCF
- TC-Diffuser (2025): https://github.com/Zjut-MultimediaPlus/TC-Diffuser

## 6. 引用

如果我们的工作对您的研究有帮助，欢迎引用：

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

如有任何问题，请通过 liulei13@ustc.edu.cn 联系我。
