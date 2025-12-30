# This is an exact copy of tools/test.py from open-mmlab/mmdetection3d.
import argparse
import os
import os.path as osp
import re

from mmengine.config import Config, ConfigDict, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--ceph', action='store_true', help='Use ceph as data storage backend')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    parser.add_argument(
        '--task',
        type=str,
        choices=[
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ],
        help='Determine the visualization method depending on the task.')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--tta', action='store_true', help='Test time augmentation')
    # 新增：是否使用类无关实例评测
    parser.add_argument(
        '--cat-agnostic', action='store_true',
        help='If set, use category-agnostic instance evaluation (AP).')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/test.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def trigger_visualization_hook(cfg, args):
    default_hooks = cfg.default_hooks
    if 'visualization' in default_hooks:
        visualization_hook = default_hooks['visualization']
        # Turn on visualization
        visualization_hook['draw'] = True
        if args.show:
            visualization_hook['show'] = True
            visualization_hook['wait_time'] = args.wait_time
        if args.show_dir:
            visualization_hook['test_out_dir'] = args.show_dir
        all_task_choices = [
            'mono_det', 'multi-view_det', 'lidar_det', 'lidar_seg',
            'multi-modality_det'
        ]
        assert args.task in all_task_choices, 'You must set '\
            f"'--task' in {all_task_choices} in the command " \
            'if you want to use visualization hook'
        visualization_hook['vis_task'] = args.task
        visualization_hook['score_thr'] = args.score_thr
    else:
        raise RuntimeError(
            'VisualizationHook must be included in default_hooks.'
            'refer to usage '
            '"visualization=dict(type=\'VisualizationHook\')"')

    return cfg


def _rewrite_diagnostics_out_dir(cfg):
    """Make diagnostics output dir unique per run.

    UnifiedSegMetric resolves relative `diagnostics.out_dir` against logger
    fields, which may fall back to CWD. Here we force it to live under the
    runner `work_dir` (absolute) to avoid cross-run overwrites.
    """

    def _rewrite_one(evaluator):
        if not isinstance(evaluator, dict):
            return
        diag = evaluator.get('diagnostics', None)
        if not isinstance(diag, dict):
            return
        out_dir = diag.get('out_dir', 'instance_diagnostics')
        out_dir = str(out_dir)
        if not osp.isabs(out_dir):
            diag['out_dir'] = osp.abspath(osp.join(cfg.work_dir, out_dir))

    for key in ('val_evaluator', 'test_evaluator'):
        if key not in cfg:
            continue
        if isinstance(cfg[key], dict):
            _rewrite_one(cfg[key])
        elif isinstance(cfg[key], (list, tuple)):
            for e in cfg[key]:
                _rewrite_one(e)

def _rewrite_online_monitor_out_dir(cfg):
    """Resolve `online_monitor.out_dir` under runner `work_dir`.

    Online monitor dumps are frequently configured as relative paths like
    `online_monitorcd`. If resolved against CWD, multiple experiments will
    overwrite each other. Here we force it to live under `cfg.work_dir`.
    """

    def _rewrite_one(evaluator):
        if not isinstance(evaluator, dict):
            return
        mon = evaluator.get('online_monitor', None)
        if not isinstance(mon, dict):
            return
        out_dir = mon.get('out_dir', 'online_monitor')
        out_dir = str(out_dir)
        if not osp.isabs(out_dir):
            mon['out_dir'] = osp.abspath(osp.join(cfg.work_dir, out_dir))

    for key in ('val_evaluator', 'test_evaluator'):
        if key not in cfg:
            continue
        if isinstance(cfg[key], dict):
            _rewrite_one(cfg[key])
        elif isinstance(cfg[key], (list, tuple)):
            for e in cfg[key]:
                _rewrite_one(e)


def _warn_checkpoint_cfg_mismatch(cfg, checkpoint_path: str) -> None:
    """Warn when checkpoint meta['cfg'] disagrees with current config.

    This repo has multiple experiment branches (baseline / online merge / DINO),
    and a silent mismatch can produce *valid but meaningless* metrics.
    """
    try:
        import torch
    except Exception:
        return

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    except Exception:
        return

    meta = ckpt.get('meta', {}) if isinstance(ckpt, dict) else {}
    ck_cfg = meta.get('cfg', None) if isinstance(meta, dict) else None
    if not isinstance(ck_cfg, str):
        return

    def _extract(pattern: str):
        m = re.search(pattern, ck_cfg)
        return m.group(1) if m else None

    ck_merge_norm = _extract(r"merge_head=dict\([^)]*norm='([^']+)'")
    ck_data_root = _extract(r"data_root\s*=\s*'([^']+)'")

    cur_merge_norm = None
    try:
        mh = cfg.get('model', {}).get('merge_head', None)
        if isinstance(mh, dict):
            cur_merge_norm = mh.get('norm', None)
    except Exception:
        cur_merge_norm = None

    cur_data_root = cfg.get('data_root', None)

    warned = False
    if ck_merge_norm is not None and (cur_merge_norm or None) != ck_merge_norm:
        print(
            f"[CKPT][cfg-mismatch] merge_head.norm: checkpoint='{ck_merge_norm}' vs config='{cur_merge_norm}'. "
            "This commonly causes AP collapse."
        )
        warned = True

    if ck_data_root is not None and cur_data_root is not None and str(cur_data_root) != ck_data_root:
        print(
            f"[CKPT][cfg-mismatch] data_root: checkpoint='{ck_data_root}' vs config='{cur_data_root}'. "
            "Check you are evaluating on the intended dataset variant."
        )
        warned = True

    if warned:
        print("[CKPT][cfg-mismatch] If this is intended, ignore; otherwise rerun with the matching config.")


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # TODO: We will unify the ceph support approach with other OpenMMLab repos
    if args.ceph:
        cfg = replace_ceph_backend(cfg)

    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 根据 --cat-agnostic 调整评测模式
    eval_mode = 'cat_agnostic' if args.cat_agnostic else 'multi_class'
    for key in ['val_evaluator', 'test_evaluator']:
        if key in cfg:
            # 支持 evaluator 为字典或列表
            if isinstance(cfg[key], dict):
                cfg[key]['eval_mode'] = eval_mode
            elif isinstance(cfg[key], (list, tuple)):
                for _e in cfg[key]:
                    if isinstance(_e, dict):
                        _e['eval_mode'] = eval_mode

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # Ensure diagnostics output does not overwrite across experiments.
    _rewrite_diagnostics_out_dir(cfg)
    # Ensure online monitor output does not overwrite across experiments.
    _rewrite_online_monitor_out_dir(cfg)

    # Guardrail: warn about common silent config/ckpt mismatches.
    _warn_checkpoint_cfg_mismatch(cfg, args.checkpoint)

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        # Currently, we only support tta for 3D segmentation
        # TODO: Support tta for 3D detection
        assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.'
        assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` in config.'
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start testing
    runner.test()


if __name__ == '__main__':
    # 记录某个函数的运行时间
    main()
