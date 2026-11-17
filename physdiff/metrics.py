"""De-normalisation and deterministic forecast metrics (track / pressure / wind MAE per lead time).

反归一化与确定性预报指标（各时效的路径 / 气压 / 风速 MAE）。
"""
from typing import Dict, List

import numpy as np

__all__ = ['physical_values', 'haversine_km', 'metric_rows', 'clean_basin']

EARTH_DIAMETER_KM = 12742.0
KNOT_TO_MS = 1852 / 3600
LEAD_HOURS = (6, 12, 18, 24)


def clean_basin(basin: np.ndarray) -> np.ndarray:
    """IBTrACS uses ``'NA'`` for the North Atlantic, which pandas reads as NaN; restore it.

    IBTrACS 用 ``'NA'`` 表示北大西洋，pandas 会把它读成 NaN，这里将其还原。
    """
    basin = np.asarray(basin).astype(str)
    return np.where(np.isin(basin, ('nan', 'NaN', '')), 'NA', basin)


def physical_values(states: np.ndarray, references: np.ndarray, config: dict) -> Dict[str, np.ndarray]:
    """Map normalised ``(..., lead, [lat, lon, wind, pressure])`` to degrees, knots and hPa.

    将归一化的 ``(..., lead, [lat, lon, wind, pressure])`` 转换为度、节和 hPa。
    """
    n = config['normalization']
    coords = states[..., :2] * np.array(n['coord_std']) + references[:, None]
    coords[..., 1] = (coords[..., 1] + 180) % 360 - 180
    return dict(coords=coords,
                winds=states[..., 2] * n['wind_std'] + n['wind_mean'],
                pres=states[..., 3] * n['pressure_std'] + n['pressure_mean'])


def haversine_km(a_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    a, b = np.deg2rad(a_deg), np.deg2rad(b_deg)
    h = np.sin((a[..., 0] - b[..., 0]) / 2) ** 2
    h += np.cos(a[..., 0]) * np.cos(b[..., 0]) * np.sin((a[..., 1] - b[..., 1]) / 2) ** 2
    return EARTH_DIAMETER_KM * np.arcsin(np.sqrt(h.clip(0, 1)))


def metric_rows(mean: np.ndarray, meta: Dict[str, np.ndarray], config: dict,
                basins=('Global', 'WP')) -> List[dict]:
    """Ensemble-mean MAE: track (km, great-circle), MSLP (hPa) and MSW (m/s) for every lead time.

    ``mean``: normalised ensemble mean ``(N, 4, 4)``; ``meta`` needs ``references``, ``basin``, ``truth_coords``,
    ``truth_winds`` (kt) and ``truth_pres`` (hPa).

    集合平均 MAE：各时效的路径（km，大圆距离）、MSLP（hPa）和 MSW（m/s）。

    ``mean`` 为归一化集合平均 ``(N, 4, 4)``；``meta`` 需包含 references、basin、
    truth_coords、truth_winds（kt）和 truth_pres（hPa）。
    """
    p = physical_values(mean.astype(np.float64), meta['references'], config)
    track = haversine_km(p['coords'], meta['truth_coords'].astype(np.float64))
    pressure = np.abs(p['pres'] - meta['truth_pres'])
    wind = np.abs(p['winds'] - meta['truth_winds']) * KNOT_TO_MS
    basin = clean_basin(meta['basin'])
    rows = []
    for name in basins:
        mask = np.ones(len(mean), bool) if name == 'Global' else basin == name
        if not mask.any():
            continue
        for lead, hours in enumerate(LEAD_HOURS):
            rows.append(dict(basin=name, lead_hours=hours, samples=int(mask.sum()),
                             track_mae_km=float(track[mask, lead].mean()),
                             pressure_mae_hpa=float(pressure[mask, lead].mean()),
                             wind_mae_ms=float(wind[mask, lead].mean())))
    return rows


def format_table(rows: List[dict]) -> str:
    """Markdown table: one line per basin, columns grouped by metric and lead time.

    Markdown 表格：每个海盆一行，列按指标和预报时效分组。
    """
    leads = sorted({r['lead_hours'] for r in rows})
    header = '| Basin | N | ' + ' | '.join(f'{m} {h}h' for m in ('Track (km)', 'MSLP (hPa)', 'MSW (m/s)') for h in leads)
    lines = [header + ' |', '|' + '---|' * (2 + 3 * len(leads))]
    for basin in dict.fromkeys(r['basin'] for r in rows):
        sub = {r['lead_hours']: r for r in rows if r['basin'] == basin}
        cells = [f"{sub[h][k]:.2f}" for k in ('track_mae_km', 'pressure_mae_hpa', 'wind_mae_ms') for h in leads]
        lines.append(f"| {basin} | {sub[leads[0]]['samples']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
