# Prepared data format

`physdiff.data.PreparedData` reads pre-processed forecast windows. Each window has 4 observed 6-hourly steps
(t−18 h … t) and 4 target steps (t+6 h … t+24 h). The raw sources are IBTrACS, ERA5 and FengWu forecasts initialised
from ERA5. Three directories are passed on the command line:

```
--data PREP                     --fine-root FINE                --geometry GEOM   (main model only)
PREP/                           FINE/                           GEOM/
├── manifest.json               ├── train/                      ├── train.npz
├── train/                      │   ├── fine_core.npy           ├── val.npz
│   ├── states.npz              │   └── packed_index.npy        └── test.npz
│   ├── weather.npy             ├── val/ ...
│   └── extrema.npy             └── test/ ...
├── val/ ...
└── test/ ...
```

## Splits

| Split | Years     | Windows | Use                                                           |
|-------|-----------|--------:|---------------------------------------------------------------|
| train | 1980–2017 | 32,721  | fitting                                                       |
| val   | 2018      | 2,307   | model (epoch) selection                                       |
| test  | 2019–2022 | 6,699   | evaluation only                                               |

Windows are assigned by forecast time, so a few test windows belong to storms whose IBTrACS SID begins with 2018.

## `states.npz`

| key            | shape      | dtype   | description |
|----------------|------------|---------|-------------|
| `states`       | (N, 8, 4)  | float32 | normalised `[lat, lon, MSW, MSLP]`; steps 0–3 observed, 4–7 targets |
| `references`   | (N, 2)     | float32 | reference `[lat, lon]` (deg); coordinates in `states` are `(x − ref) / coord_std` |
| `truth_coords` | (N, 4, 2)  | float32 | target lat/lon (deg) |
| `truth_winds`  | (N, 4)     | float32 | target MSW (kt) |
| `truth_pres`   | (N, 4)     | float32 | target MSLP (hPa) |
| `basin`        | (N,)       | str     | IBTrACS basin (`WP`, `EP`, `NA`, `NI`, `SI`, `SP`) |
| `sid`          | (N,)       | str     | IBTrACS storm id |
| `packed_index` | (N,)       | int64   | window id; must match `FINE` and `GEOM` |

Intensities are normalised with `config['normalization']`: `(MSW − wind_mean) / wind_std`, `(MSLP − pressure_mean) / pressure_std`.

## Environmental fields

69 channels at 0.25°: `u10, v10, t2m, msl` followed by `z, q, u, v, t` on 13 levels
(50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa), z-scored with
`config['weather_normalization']` (statistics computed from the training years).
Steps 0–3 come from ERA5 and steps 4–7 from the FengWu forecast. The model without FengWu never reads steps 4–7.

| file               | shape                    | description |
|--------------------|--------------------------|-------------|
| `weather.npy`      | (N, 8, 2, 69, 10, 10)    | storm-centred crops at two spatial scales, pooled to 10×10 |
| `extrema.npy`      | (N, 8, 69, 2)            | per-channel extrema of the crops (index 0 of channel 3 is the min MSLP) |
| `fine_core.npy`    | (N, 8, 11, 40, 40)       | high-resolution core crop of channels `[0, 1, 2, 3, 24, 26, 33, 40, 46, 53, 66]` |

## `GEOM/<split>.npz` (main model)

| key                | shape     | description |
|--------------------|-----------|-------------|
| `forecast_centers` | (N, 4, 2) | lat/lon (deg) of the FengWu vortex tracked at t+6 h … t+24 h |
| `vortex_detected`  | (N, 4)    | whether the tracker found a vortex |
| `packed_index`     | (N,)      | must match `states.npz` |

## Building `PREP` and `FINE`

`windows/<split>.csv` lists every window in release order (`row, packed_index, sid, basin, t0`, with `t0` the last
observed step). `tools/prepare.py` builds `PREP` and `FINE` from it:

1. IBTrACS v04r00 (`ibtracs.ALL.list.v04r00.csv`) gives `LAT, LON, USA_WIND, USA_PRES` at t0−18 h … t0+24 h.
2. ERA5 crops for t0−18 h … t0 are read from `<crops>/<year>/<sid>/<YYYY-mm-dd HH_MM_SS>/ERA5_data.npy`:
   (69, 80, 80) float32, 0.25°, centred on the IBTrACS position, rows north to south, channels as below.
3. Each crop is z-scored with `(x − mean) / (std + 1e-8)` (`weather_normalization`), then
   `weather` = 8×8 block means of the 80×80 crop and 4×4 block means of its central 40×40 (pixels 20:60),
   `extrema` = per-channel min/max over the 80×80 crop, `fine_core` = central 40×40 of 11 channels.

```bash
python tools/prepare.py --ibtracs ibtracs.ALL.list.v04r00.csv --crops TC_ERA5 --splits test --data PREP --fine-root FINE
```

Steps 4–7 are left as zeros (the `wo_fengwu` model never reads them). For the main model, `tools/prepare_fengwu.py`
then fills them and writes `GEOM`:

1. FengWu v1 (`fengwu_v1.onnx` with `data_mean.npy`, `data_std.npy`, from [FengWu](https://github.com/OpenEarthLab/FengWu))
   is initialised from global ERA5 at t0−6 h and t0 and rolled out to t0+24 h. ERA5 files are `<era5>/<YYYYmmdd_HH>.npy`,
   (69, 721, 1440), 90°N→90°S, 0°E→359.75°E, in the official FengWu channel order (same as below).
2. At each lead the vortex is the 850 hPa relative-vorticity maximum within ±5° of the previous centre (IBTrACS at t0
   for the first lead); if it is below 5×10⁻⁵ s⁻¹ the previous centre is kept and `vortex_detected` is false.
3. An 80×80 crop around that centre is reduced as in step 3 above.

```bash
python tools/prepare_fengwu.py --model fengwu_v1.onnx --era5 ERA5 --ibtracs ibtracs.ALL.list.v04r00.csv \
    --splits test --data PREP --fine-root FINE --geometry GEOM
```

This needs `onnxruntime-gpu` and takes ~7 s per initialisation on one GPU (test: 3,649 initialisations).

## `manifest.json`

Training requires `{"phase": "complete", "stats": {"env_channel_stats": {...}}}`. The channel statistics overwrite
`weather_normalization` in the config. For the released checkpoints they are identical to the copy in `configs/*.json`.
