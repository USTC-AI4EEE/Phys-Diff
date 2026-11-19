"""Add FengWu steps 4-7 to PREP / FINE and write GEOM (main model). Run after tools/prepare.py.

    python tools/prepare_fengwu.py --model fengwu_v1.onnx --era5 ERA5 --ibtracs ibtracs.ALL.list.v04r00.csv \
        --splits test --data PREP --fine-root FINE --geometry GEOM
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import ROOT, reduce, read_ibtracs  # noqa: E402

LEVEL_TO_VARIABLE_MAJOR = [0, 1, 2, 3] + [4 + 5 * level + v for v in range(5) for level in range(13)]
U850, V850 = 4 + 2 * 13 + 10, 4 + 3 * 13 + 10  # official FengWu order
RES, CROP, SEARCH, THRESHOLD = 0.25, 80, 20, 5.0  # 0.25 deg grid, 80x80 crop, +-5 deg search, 5e-5 s^-1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT / 'configs' / 'main.json')
    p.add_argument('--model', type=Path, required=True, help='fengwu_v1.onnx (data_mean.npy / data_std.npy beside it)')
    p.add_argument('--era5', type=Path, required=True, help='global ERA5 <YYYYmmdd_HH>.npy, (69, 721, 1440)')
    p.add_argument('--level-major', action='store_true', help='ERA5 files interleave z,q,u,v,t per level')
    p.add_argument('--ibtracs', type=Path, required=True)
    p.add_argument('--windows', type=Path, default=ROOT / 'windows')
    p.add_argument('--splits', default='test')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--fine-root', type=Path, required=True)
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    return p.parse_args()


class FengWu:
    def __init__(self, path):
        import onnxruntime as ort
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()  # CUDA / cuDNN from the torch or nvidia-* wheels
        self.session = ort.InferenceSession(str(path), providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.mean = np.load(path.parent / 'data_mean.npy').reshape(69, 1, 1).astype(np.float32)
        self.std = np.load(path.parent / 'data_std.npy').reshape(69, 1, 1).astype(np.float32)

    def rollout(self, frames, steps=4):
        # returns normalised fields; de-normalise only what is used (u/v 850 and the crops)
        x = np.concatenate([(f - self.mean) / self.std for f in frames])[None].astype(np.float32)
        out = []
        for _ in range(steps):
            y = self.session.run(None, {self.session.get_inputs()[0].name: x})[0]
            out.append(y[0, :69])
            x = np.concatenate([x[:, 69:], y[:, :69]], 1)
        return out

    def physical(self, field, channels=slice(None)):
        return field[channels] * self.std[channels] + self.mean[channels]


def grid_index(lat, lon):
    return int(np.clip(round((90 - lat) / RES), 0, 720)), int(round((lon % 360) / RES)) % 1440


def vorticity_850(u, v):
    u, v = u.astype(np.float64), v.astype(np.float64)
    d = np.deg2rad(RES)
    cos = np.cos(np.deg2rad(90 - np.arange(721) * RES))[:, None]
    dv = (np.roll(v, -1, 1) - np.roll(v, 1, 1)) / (2 * d)
    du = np.gradient(u * cos, -d, axis=0, edge_order=2)
    out = np.zeros_like(u)
    np.divide(dv - du, 6371000. * cos, out=out, where=np.abs(cos) > 1e-6)
    return out * 1e5


def track(vorticity, lat, lon):
    # 850 hPa vorticity peak within +-5 deg of the previous centre; None if below threshold
    r, c = grid_index(lat, lon)
    rows = np.arange(max(0, r - SEARCH), min(721, r + SEARCH + 1))
    cols = np.arange(c - SEARCH, c + SEARCH + 1) % 1440
    region = vorticity[np.ix_(rows, cols)] * (1 if lat >= 0 else -1)
    i, j = np.unravel_index(np.argmax(region), region.shape)
    if region[i, j] < THRESHOLD:
        return None
    return 90 - rows[i] * RES, (cols[j] * RES) % 360


def crop(field, lat, lon):
    r, c = grid_index(lat, lon)
    return field[:, r - CROP // 2:r + CROP // 2][:, :, np.arange(c - CROP // 2, c + CROP // 2) % 1440]


def crop_physical(model, field, lat, lon):
    return crop(field, lat, lon) * model.std + model.mean


def build_split(split, args, model, mean, std):
    with open(args.windows / f'{split}.csv', newline='', encoding='utf-8') as f:
        windows = list(csv.DictReader(f))
    if args.limit:
        windows = windows[:args.limit]
    states = np.load(args.data / split / 'states.npz')
    if not (states['sid'][:len(windows)] == [w['sid'] for w in windows]).all():
        raise ValueError('Run tools/prepare.py on the same windows first')
    track_table = read_ibtracs(args.ibtracs, {w['sid'] for w in windows})
    weather = np.load(args.data / split / 'weather.npy', mmap_mode='r+')
    extrema = np.load(args.data / split / 'extrema.npy', mmap_mode='r+')
    fine = np.load(args.fine_root / split / 'fine_core.npy', mmap_mode='r+')
    centers = np.zeros((len(windows), 4, 2), np.float32)
    detected = np.zeros((len(windows), 4), bool)

    cache = {}

    def frame(time):
        key = f'{time:%Y%m%d_%H}'
        if key not in cache:
            x = np.load(args.era5 / f'{key}.npy')
            if len(cache) > 2:
                cache.clear()
            cache[key] = np.ascontiguousarray(x[LEVEL_TO_VARIABLE_MAJOR] if args.level_major else x, np.float32)
        return cache[key]

    by_init = defaultdict(list)
    for i, w in enumerate(windows):
        by_init[w['t0']].append(i)
    for done, (t0, rows) in enumerate(sorted(by_init.items()), 1):
        t = datetime.fromisoformat(t0)
        fields = model.rollout([frame(t - timedelta(hours=6)), frame(t)])  # inputs: ERA5 at t0-6h and t0
        vorticity = [vorticity_850(*model.physical(f, [U850, V850])) for f in fields]
        for i in rows:
            lat, lon = track_table[windows[i]['sid'], t][:2]
            crops = []
            for k, field in enumerate(fields):
                center = track(vorticity[k], lat, lon)
                detected[i, k] = center is not None
                if center is not None:
                    lat, lon = center
                crops.append(crop_physical(model, field, lat, lon))
                centers[i, k] = lat, lon
            wt, ex, fc = reduce(np.stack(crops), mean, std)
            weather[i, 4:], extrema[i, 4:], fine[i, 4:] = wt, ex, fc
        print(f'\r[{split}] {done}/{len(by_init)} initialisations', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    for a in (weather, extrema, fine):
        a.flush()
    args.geometry.mkdir(parents=True, exist_ok=True)
    np.savez(args.geometry / f'{split}.npz', forecast_centers=centers, vortex_detected=detected,
             packed_index=states['packed_index'][:len(windows)])


def main():
    args = parse_args()
    stats = json.loads(args.config.read_text(encoding='utf-8'))['weather_normalization']
    mean = np.array(stats['channel_means'], np.float32)[:, None, None]
    std = np.array(stats['channel_stds'], np.float32)[:, None, None]
    model = FengWu(args.model)
    for split in args.splits.split(','):
        build_split(split, args, model, mean, std)


if __name__ == '__main__':
    main()
