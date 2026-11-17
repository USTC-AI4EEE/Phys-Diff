import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physdiff.checkpoint import VARIANTS, load_config  # noqa: E402
from physdiff.metrics import format_table, metric_rows  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('files', nargs='*', type=Path, help='ensemble .npz files (default: members/*.npz)')
    p.add_argument('--config', default='main', help='config providing the normalisation statistics')
    p.add_argument('--reference', type=Path, help='compare normalized_members against this file')
    p.add_argument('--json', action='store_true', help='print rows as JSON instead of a table')
    args = p.parse_args()
    config = load_config(args.config)
    files = args.files or [v['members'] for v in VARIANTS.values()]
    for path in files:
        with np.load(path) as z:
            bundle = {k: z[k] for k in z.files}
        mean = bundle['normalized_members'].mean(1)
        rows = metric_rows(mean, bundle, config)
        print(f'## {path}  (windows={len(mean)}, members={bundle["normalized_members"].shape[1]})')
        print(json.dumps(rows, indent=2) if args.json else format_table(rows))
        if args.reference:
            with np.load(args.reference) as ref:
                np.testing.assert_array_equal(ref['sid'][:len(mean)], bundle['sid'])
                diff = np.abs(ref['normalized_members'][:len(mean)] - bundle['normalized_members'])
            print(f'max |members - reference| = {diff.max():.3e}  (mean {diff.mean():.3e})')
        print()


if __name__ == '__main__':
    main()
