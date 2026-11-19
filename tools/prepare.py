"""Build PREP / FINE (docs/DATA.md) from IBTrACS and storm-centred (69, 80, 80) ERA5 crops.

    python tools/prepare.py --ibtracs ibtracs.ALL.list.v04r00.csv --crops TC_ERA5 --splits test --data PREP --fine-root FINE
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FINE_CHANNELS = [0, 1, 2, 3, 24, 26, 33, 40, 46, 53, 66]  # u10 v10 t2m msl q500 q700 u200 u850 v200 v850 t850
STEPS, HISTORY = 8, 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=ROOT / 'configs' / 'wo_fengwu.json')
    p.add_argument('--ibtracs', type=Path, required=True)
    p.add_argument('--crops', type=Path, required=True, help='root of <year>/<sid>/<YYYY-mm-dd HH_MM_SS>/ERA5_data.npy')
    p.add_argument('--windows', type=Path, default=ROOT / 'windows')
    p.add_argument('--splits', default='test')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--fine-root', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--check', type=Path, help='released PREP root to compare with')
    p.add_argument('--check-fine', type=Path, help='released FINE root to compare with')
    return p.parse_args()


def step_times(t0):
    t0 = datetime.fromisoformat(t0)  # t0 = last observed step
    return [t0 + timedelta(hours=6 * (k - HISTORY + 1)) for k in range(STEPS)]


def read_ibtracs(path, sids):
    table = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['SID'] not in sids:
                continue
            try:
                table[r['SID'], datetime.strptime(r['ISO_TIME'], '%Y-%m-%d %H:%M:%S')] = tuple(
                    float(r[k]) for k in ('LAT', 'LON', 'USA_WIND', 'USA_PRES'))
            except ValueError:
                continue
    return table


def crop_path(root, sid, time):
    name = time.strftime('%Y-%m-%d %H_%M_%S')
    for year in (time.year, int(sid[:4]), time.year - 1, time.year + 1):  # storms crossing New Year
        path = root / str(year) / sid / name / 'ERA5_data.npy'
        if path.exists():
            return path
    raise FileNotFoundError(f'No ERA5 crop for {sid} at {name}')


def reduce(fields, mean, std):
    # z-score -> [8x8 block means of 80x80, 4x4 block means of central 40x40], channel min/max, central 40x40 core
    x = ((fields.astype(np.float32) - mean) / (std + 1e-8)).astype(np.float32)
    s = len(x)
    full = x.astype(np.float64).reshape(s, 69, 10, 8, 10, 8).mean((3, 5))
    core = x[..., 20:60, 20:60].astype(np.float64).reshape(s, 69, 10, 4, 10, 4).mean((3, 5))
    return (np.stack([full, core], 1).astype(np.float32), np.stack([x.min((2, 3)), x.max((2, 3))], -1),
            x[:, FINE_CHANNELS, 20:60, 20:60])


def build_split(split, args, config, mean, std):
    with open(args.windows / f'{split}.csv', newline='', encoding='utf-8') as f:
        windows = list(csv.DictReader(f))
    if args.limit:
        windows = windows[:args.limit]
    n, norm = len(windows), config['normalization']
    track = read_ibtracs(args.ibtracs, {w['sid'] for w in windows})
    raw = np.array([[track[w['sid'], t] for t in step_times(w['t0'])] for w in windows], np.float64)
    coords, ref = raw[..., :2], raw[:, 0, :2]
    states = np.concatenate([(coords - ref[:, None]) / np.array(norm['coord_std']),
                             ((raw[..., 2:3] - norm['wind_mean']) / norm['wind_std']),
                             ((raw[..., 3:4] - norm['pressure_mean']) / norm['pressure_std'])], -1)
    packed_index = np.array([int(w['packed_index']) for w in windows], np.int64)

    out, fine_out = args.data / split, args.fine_root / split
    out.mkdir(parents=True, exist_ok=True)
    fine_out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'states.npz', states=states.astype(np.float32), references=ref.astype(np.float32),
             truth_coords=coords[:, HISTORY:].astype(np.float32), truth_winds=raw[:, HISTORY:, 2].astype(np.float32),
             truth_pres=raw[:, HISTORY:, 3].astype(np.float32), basin=np.array([w['basin'] for w in windows]),
             sid=np.array([w['sid'] for w in windows]), packed_index=packed_index)
    np.save(fine_out / 'packed_index.npy', packed_index)
    weather = np.lib.format.open_memmap(out / 'weather.npy', 'w+', np.float32, (n, STEPS, 2, 69, 10, 10))
    extrema = np.lib.format.open_memmap(out / 'extrema.npy', 'w+', np.float32, (n, STEPS, 69, 2))
    fine = np.lib.format.open_memmap(fine_out / 'fine_core.npy', 'w+', np.float32, (n, STEPS, 11, 40, 40))
    # steps 4-7 stay zero here; tools/prepare_fengwu.py fills them for the main model

    def window(i):
        w = windows[i]
        fields = [np.load(crop_path(args.crops, w['sid'], t)) for t in step_times(w['t0'])[:HISTORY]]
        return i, reduce(np.stack(fields), mean, std)

    with ThreadPoolExecutor(args.workers) as pool:
        for done, (i, (wt, ex, fc)) in enumerate(pool.map(window, range(n)), 1):
            weather[i, :len(wt)], extrema[i, :len(wt)], fine[i, :len(wt)] = wt, ex, fc
            if done % 500 == 0 or done == n:
                print(f'\r[{split}] {done}/{n}', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    for a in (weather, extrema, fine):
        a.flush()
    return n


def check(split, args, n, steps):
    new, ref = np.load(args.data / split / 'states.npz'), np.load(args.check / split / 'states.npz')
    diffs = {k: float(np.abs(new[k] - ref[k][:n]).max())
             for k in ('states', 'references', 'truth_coords', 'truth_winds', 'truth_pres')}
    diffs['ids_equal'] = bool((new['sid'] == ref['sid'][:n]).all() and (new['packed_index'] == ref['packed_index'][:n]).all())
    for name, a, b in (('weather', args.data, args.check), ('extrema', args.data, args.check),
                       ('fine_core', args.fine_root, args.check_fine)):
        x, y = np.load(a / split / f'{name}.npy', mmap_mode='r'), np.load(b / split / f'{name}.npy', mmap_mode='r')
        diffs[name] = max(float(np.abs(np.asarray(x[i, :steps]) - np.asarray(y[i, :steps])).max()) for i in range(n))
    print(f'[{split}] max |new - released|:', json.dumps(diffs))


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    stats = config['weather_normalization']
    mean = np.array(stats['channel_means'], np.float32)[:, None, None]
    std = np.array(stats['channel_stds'], np.float32)[:, None, None]
    splits = args.splits.split(',')
    sizes = {s: build_split(s, args, config, mean, std) for s in splits}
    args.data.mkdir(parents=True, exist_ok=True)
    (args.data / 'manifest.json').write_text(json.dumps(
        dict(phase='complete', stats=dict(env_channel_stats=stats), splits=sizes)), encoding='utf-8')
    if args.check:
        for s in splits:
            check(s, args, sizes[s], HISTORY)


if __name__ == '__main__':
    main()
