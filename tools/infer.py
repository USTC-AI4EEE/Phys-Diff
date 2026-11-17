"""Generate the 50-member Phys-Diff test-set ensemble by replaying the initial noise stored in members/<variant>.npz.

    python tools/infer.py --variant main --data PREP --geometry GEOM --fine-root FINE --output outputs/main_test.npz
    python tools/infer.py --variant wo_fengwu --data PREP --fine-root FINE --output outputs/wo_fengwu_test.npz

The output ``.npz`` has the same schema as ``members/*.npz``; a ``.json`` with the metric table is written next to it.

回放 members/<variant>.npz 中保存的初始噪声，生成 50 成员 Phys-Diff 测试集集合预报，并评估集合平均。
输出 ``.npz`` 与 ``members/*.npz`` 格式相同，并在旁边写出含指标表的 ``.json``。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physdiff.checkpoint import VARIANTS, load_model  # noqa: E402
from physdiff.data import PreparedData  # noqa: E402
from physdiff.metrics import format_table, metric_rows  # noqa: E402

META_KEYS = ('sid', 'basin', 'references', 'truth_coords', 'truth_winds', 'truth_pres')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', choices=sorted(VARIANTS), default='main')
    p.add_argument('--checkpoint', type=Path, help='override the released checkpoint of --variant')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--geometry', type=Path, help='FengWu vortex-tracker centres (main model only)')
    p.add_argument('--fine-root', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--limit', type=int, default=0, help='only the first N windows (debugging)')
    p.add_argument('--output', type=Path)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--num-threads', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    model, config = load_model(args.checkpoint or args.variant, args.device)
    if config['story']['use_fengwu'] != (args.variant == 'main'):
        raise ValueError(f'Checkpoint use_fengwu={config["story"]["use_fengwu"]} does not match --variant {args.variant}')
    torch.set_num_threads(args.num_threads)

    data = PreparedData(args.data, args.geometry, args.fine_root, config, parts=['test'])
    n = min(data.n, args.limit) if args.limit else data.n
    ids = np.arange(n)
    meta = {k: data.meta[k][:n] for k in META_KEYS}

    with np.load(VARIANTS[args.variant]['members']) as bundle:
        np.testing.assert_array_equal(bundle['sid'][:n], meta['sid'])
        np.testing.assert_allclose(bundle['references'][:n], meta['references'])
        replay = bundle['initial_noise'][:n]

    outputs, noises = [], []
    for start in range(0, n, args.batch_size):
        batch_ids = ids[start:start + args.batch_size]
        noise = torch.as_tensor(replay[batch_ids], device=args.device)
        prediction = model.sample(noise=noise, **data.inputs(batch_ids, args.device))
        outputs.append(prediction.cpu().numpy())
        noises.append(noise.cpu().numpy())
        print(f'\r{min(start + args.batch_size, n)}/{n}', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    members = np.concatenate(outputs)
    mean = members.mean(1)

    rows = metric_rows(mean, meta, config)
    result = dict(variant=args.variant, split='test', samples=int(n), members=int(members.shape[1]),
                  ddim_steps=config['model']['ddpm']['sampling_steps'], eta=config['model']['ddpm']['ddim_eta'], rows=rows)
    print(format_table(rows))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **meta, initial_noise=np.concatenate(noises), normalized_members=members,
                            normalized_mean=mean, member_selection=np.array('random_initialization'))
        args.output.with_suffix('.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(f'saved {args.output}')


if __name__ == '__main__':
    main()
