"""Train Phys-Diff from scratch (anchor -> base_diffusion -> alignment) with validation-based epoch selection.

    python tools/train.py --config configs/main.json      --data PREP --geometry GEOM --fine-root FINE --output runs/main.pth
    python tools/train.py --config configs/wo_fengwu.json --data PREP                 --fine-root FINE --output runs/wo_fengwu.pth

The model is fitted on the train split (1980-2017). After every epoch it is scored on the validation split (2018):
ensemble-mean MAE of track + MSW + MSLP, each divided by the phase's task scale and averaged over the 4 lead times
(the anchor forecast in the ``anchor`` phase, a ``--val-members`` DDIM ensemble with fixed seeded noise in the
diffusion phases, using the EMA weights). The best epoch of each phase is restored before the next phase starts, and
the best alignment epoch is saved. Validation scores are only printed, not stored in the checkpoint.
The test split (2019-2022) is never read.

从头训练 Phys-Diff（anchor -> base_diffusion -> alignment），按验证集选择 epoch。

模型在训练集（1980-2017）上拟合，每个 epoch 后在验证集（2018）上评估：路径 + MSW + MSLP 的集合平均 MAE，
各自除以该阶段的任务尺度并在 4 个时效上平均（``anchor`` 阶段用 anchor 预报，扩散阶段用 EMA 权重和固定种子噪声生成 ``--val-members`` 个成员）
。每个阶段的最佳 epoch 在下一阶段开始前恢复，最终保存 alignment 的最佳 epoch。验证得分只打印，不写入 checkpoint。训练过程从不读取测试集（2019-2022）
。
"""
import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physdiff.checkpoint import load_config, save_checkpoint  # noqa: E402
from physdiff.data import PreparedData  # noqa: E402
from physdiff.losses import UncertaintyLoss, routed_backward  # noqa: E402
from physdiff.models import PhysDiff  # noqa: E402

PHASES = ('anchor', 'base_diffusion', 'alignment')
PHASE_SEED_OFFSET = dict(anchor=0, base_diffusion=100, alignment=200)


@torch.no_grad()
def cache_context(model, data, memory=False, batch_size=64):
    """Pre-compute anchors (and optionally encoder memories) for every training window.

    为每个训练窗口预先计算 anchor（以及可选的编码器 memory）。
    """
    model.eval()
    device = next(model.parameters()).device
    anchor = torch.empty(data.n, 4, 4, device=device)
    memories = None
    for start in range(0, data.n, batch_size):
        ids = np.arange(start, min(start + batch_size, data.n))
        a, m = model.anchor(**data.inputs(ids, device))
        anchor[ids] = a
        if memory:
            if memories is None:
                memories = torch.empty((data.n,) + m.shape[1:], device=device)
            memories[ids] = m
    result = dict(anchor=anchor, target=torch.as_tensor(data.meta['states'][:, 4:], device=device),
                  references=torch.as_tensor(data.meta['references'], device=device))
    if memory:
        result['memory'] = memories
    return result


def objective(model, criterion, data, ids, phase, cached):
    device = next(model.parameters()).device
    if phase == 'anchor':
        batch = data.inputs(ids, device)
        target = torch.as_tensor(data.targets(ids), device=device)
        prediction, _ = model.anchor(**batch)
        diffusion = {'value': prediction.sum() * 0}
        references = batch['references']
    else:
        target, references = cached['target'][ids], cached['references'][ids]
        # base_diffusion: frozen cached memory; alignment: condition encoder is trained, so recompute it
        # base_diffusion：使用冻结的缓存 memory；alignment：条件编码器参与训练，需重新计算
        memory = cached['memory'][ids] if phase == 'base_diffusion' else model.diffusion_memory(**data.inputs(ids, device))
        prediction, diffusion = model.training_prediction(target, cached['anchor'][ids], memory,
                                                          alignment=phase == 'alignment')
    return criterion(prediction, target, references, diffusion, phase)


def optimizer_for(model, criterion, config, phase):
    settings = config['full_data'][phase]
    if phase == 'alignment':
        groups = [dict(params=model.denoiser.parameters(), lr=settings['learning_rate']),
                  dict(params=model.condition_context.parameters(), lr=settings['condition_learning_rate']),
                  dict(params=criterion.parameters(), lr=settings['criterion_learning_rate'])]
    else:
        trainable = [p for p in model.parameters() if p.requires_grad] + \
                    [p for p in criterion.parameters() if p.requires_grad]
        groups = [dict(params=trainable, lr=settings['learning_rate'])]
    optimizer = torch.optim.Adam(groups, weight_decay=config['training']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings['schedule_horizon'], eta_min=1e-5)
    return optimizer, scheduler


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith('cuda'):
        torch.cuda.manual_seed(seed)


def update_ema(ema, model, decay):
    with torch.no_grad():
        for name in ('denoiser', 'condition_context'):
            for old, new in zip(getattr(ema, name).parameters(), getattr(model, name).parameters()):
                old.lerp_(new.detach(), 1 - decay)
            for old, new in zip(getattr(ema, name).buffers(), getattr(model, name).buffers()):
                old.copy_(new)


@torch.no_grad()
def validation_score(model, criterion, val, phase, members, seed, batch_size):
    """Normalised validation MAE (track + MSW + MSLP, averaged over leads); lower is better.

    归一化验证 MAE（路径 + MSW + MSLP，在各时效上平均），越低越好。
    """
    model.eval()
    device = next(model.parameters()).device
    total = torch.zeros(3, device=device)
    for start in range(0, val.n, batch_size):
        ids = np.arange(start, min(start + batch_size, val.n))
        batch = val.inputs(ids, device)
        target = torch.as_tensor(val.targets(ids), device=device)
        if phase == 'anchor':
            prediction, _ = model.anchor(**batch)
        else:
            noise = model.seeded_noise(val.meta['noise_index'][ids], members, seed, device)
            prediction = model.sample(noise=noise, **batch).mean(1)
        errors = criterion.per_lead(prediction, target, batch['references']) / criterion.task_scales.clamp_min(1e-3)
        total += errors.mean(1).sum(0)
    per_task = (total / val.n).tolist()
    return sum(per_task), dict(coord=per_task[0], msw=per_task[1], mslp=per_task[2])


def cpu_state(module):
    return {k: v.detach().to('cpu', copy=True) for k, v in module.state_dict().items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='main', help="config path or variant name ('main' / 'wo_fengwu')")
    p.add_argument('--data', type=Path, required=True, help='prepared windows root (contains manifest.json)')
    p.add_argument('--geometry', type=Path, help='FengWu vortex-tracker centres (main model only)')
    p.add_argument('--fine-root', type=Path, required=True, help='high-resolution storm-core crops')
    p.add_argument('--output', type=Path, required=True, help='where to write the final checkpoint')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--val-members', type=int, default=10, help='ensemble size for validating diffusion phases')
    p.add_argument('--val-batch-size', type=int, default=32)
    p.add_argument('--num-threads', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; choose a new output path.')
    config = load_config(args.config)
    manifest = json.loads((args.data / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('phase') != 'complete':
        raise RuntimeError('Data preparation is incomplete (manifest.json phase != "complete")')
    config['weather_normalization'] = manifest['stats']['env_channel_stats']
    assert config['training']['task_gradient_routing']

    data = PreparedData(args.data, args.geometry, args.fine_root, config, parts=('train',))
    val = PreparedData(args.data, args.geometry, args.fine_root, config, parts=('val',))
    torch.set_num_threads(args.num_threads)
    torch.backends.cudnn.benchmark = True
    seed = config['training']['seed']
    seed_everything(seed, args.device)
    model = PhysDiff(config).to(args.device)
    criterion = UncertaintyLoss(config).to(args.device)

    initial = cache_context(model, data)
    criterion.task_scales.copy_(
        criterion.per_lead(initial['anchor'], initial['target'], initial['references']).mean(0).clamp_min(1e-3))
    del initial

    history = []
    batch_size = config['training']['batch_size']
    for phase in PHASES:
        settings = config['full_data'][phase]
        cached = None if phase == 'anchor' else cache_context(model, data, memory=phase == 'base_diffusion')
        if phase == 'base_diffusion':
            # initialise the diffusion condition encoder from the trained anchor encoder and fit the latent codec
            # 用训练好的 anchor 编码器初始化扩散条件编码器，并拟合潜空间编解码器
            model.condition_context.load_state_dict(model.context.state_dict(), strict=True)
            residual = cached['target'] - cached['anchor']
            model.residual_scale.copy_(residual.square().mean(0).sqrt().clamp_min(1e-3))
            criterion.task_scales.copy_(
                criterion.per_lead(cached['anchor'], cached['target'], cached['references']).mean(0).clamp_min(1e-3))
            model.codec.fit(residual / model.residual_scale)
            del residual
        model.set_phase(phase)
        criterion.requires_grad_(phase != 'anchor')
        optimizer, scheduler = optimizer_for(model, criterion, config, phase)
        params = [parameter for group in optimizer.param_groups for parameter in group['params']]
        ema = copy.deepcopy(model).eval().requires_grad_(False) if phase != 'anchor' else None
        best = dict(score=float('inf'), epoch=None, state=None)
        for epoch in range(1, settings['epochs'] + 1):
            seed_everything(seed + 100003 * (epoch + PHASE_SEED_OFFSET[phase]), args.device)
            order = torch.randperm(data.n, generator=torch.Generator().manual_seed(seed + epoch)).numpy()
            model.set_phase(phase)
            criterion.requires_grad_(phase != 'anchor')
            totals, began = {}, time.monotonic()
            for start in range(0, data.n, batch_size):
                ids = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                losses = objective(model, criterion, data, ids, phase, cached)
                if not torch.isfinite(losses['total_loss']):
                    raise FloatingPointError('Nonfinite training loss')
                routed_backward(model, losses)
                torch.nn.utils.clip_grad_norm_(params, config['training']['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                if ema is not None:
                    update_ema(ema, model, config['training']['ema_decay'])
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.) + float(value.detach()) * len(ids)
            seconds = time.monotonic() - began
            evaluated = ema if ema is not None else model
            score, per_task = validation_score(evaluated, criterion, val, phase, args.val_members, seed,
                                               args.val_batch_size)
            if score < best['score']:
                best = dict(score=score, epoch=epoch, state=cpu_state(evaluated), criterion=cpu_state(criterion))
            history.append(dict(stage=phase, epoch=epoch, seconds=seconds,
                                training_losses={k: v / data.n for k, v in totals.items()},
                                learning_rates=scheduler.get_last_lr()))
            print(json.dumps(dict(history[-1], validation=dict(score=score, **per_task), best_epoch=best['epoch'])),
                  flush=True)
            scheduler.step()
        model.load_state_dict(best['state'], strict=True)
        criterion.load_state_dict(best['criterion'], strict=True)
        print(json.dumps(dict(stage=phase, selected_epoch=best['epoch'], validation_score=best['score'])), flush=True)
        selected = dict(stage=phase, epoch=best['epoch'])
        del cached, ema, optimizer, scheduler, best
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()

    save_checkpoint(args.output, model, criterion, history, stage=selected['stage'], epoch=selected['epoch'])
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
