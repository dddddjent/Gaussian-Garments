"""Coordinate ContourCraft rollout and Gaussian-Garments comparison videos."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

# Public template command (gaugar environment, from Gaussian-Garments):
# python run.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage evaluate --simulation-python /home/ljl/miniforge3/envs/ccraft/bin/python --evaluate-from-start


def select_checkpoint(output: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        assert path.is_file(), f'Missing fitted ContourCraft checkpoint: {path}'
        return path
    candidates = [path for path in (output / 'stage4/checkpoints').glob('step_*.pth')
                  if path.stem.removeprefix('step_').isdigit()]
    assert candidates, f'Run --stage simulation-fit first: no fitted checkpoint in {output / "stage4/checkpoints"}'
    return max(candidates, key=lambda path: int(path.stem.removeprefix('step_')))


def select_appearance(output: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        assert path.is_file(), f'Missing appearance checkpoint: {path}'
        return path
    candidates = [path for path in (output / 'stage3').glob('epoch*/net.pt')
                  if path.parent.name.removeprefix('epoch').isdigit()]
    assert candidates, f'Complete an appearance epoch first: {output / "stage3"}'
    return max(candidates, key=lambda path: int(path.parent.name.removeprefix('epoch')))


def evaluate(args: argparse.Namespace, manifest: dict[str, Any], data: Path, output: Path) -> None:
    assert manifest.get('body_model', 'smplx') in ('smplx', 'raw'), 'Unsupported ContourCraft body model.'
    assert args.initialization == 'colmap', '--initialization applies only to initialization.'
    assert args.simulation_python is not None, 'Set --simulation-python to the ContourCraft interpreter.'
    simulation_python = args.simulation_python.expanduser().absolute()
    assert simulation_python.is_file(), simulation_python
    checkpoint = select_checkpoint(output, args.checkpoint)
    appearance = select_appearance(output, args.appearance_checkpoint)
    config = json.loads((data / 'config.json').read_text())
    render_python = Path(config['render_python'])
    assert render_python.is_file(), render_python
    scope = 'full_sequence' if args.evaluate_from_start else 'held_out'
    evaluation = output / 'evaluation' / scope
    assert not evaluation.exists(), f'Evaluation destination already exists: {evaluation}'
    contourcraft = args.contourcraft_root.resolve()
    rollout_script = contourcraft / 'evaluate_gaussian_garments.py'
    renderer = Path(__file__).resolve().parent / 'render_evaluation.py'
    for path in (rollout_script, renderer, output / 'stage4/finetune.yaml',
                 output / 'stage4/preparation.json', data / 'preparation/sequence.npz',
                 data / 'capture/manifest.json', data / 'capture/cam_info.json'):
        assert path.is_file(), path
    rollout = [str(simulation_python), str(rollout_script), '--data', str(data), '--output', str(output),
               '--evaluation', str(evaluation), '--checkpoint', str(checkpoint)]
    if args.evaluate_from_start:
        rollout.append('--evaluate-from-start')
    rendering = [str(render_python), str(renderer), '--data', str(data), '--output', str(output),
                 '--evaluation', str(evaluation), '--appearance-checkpoint', str(appearance)]
    commands = [(rollout, contourcraft)] + [
        ([str(render_python) if branch == 'gt-lighting' else sys.executable] + rendering[1:] + ['--branch', branch], renderer.parent)
        for branch in ('gt-lighting', 'textures', 'fitted-appearance')]
    env = dict(os.environ, GAUGAR_DATA_ROOT=str(data / 'input'), GAUGAR_OUTPUT_ROOT=str(output.parent),
               GAUGAR_AUX_ROOT=str(args.aux_root.resolve()))
    print(f'Evaluation: {scope}; continuous rollout from frame 0; videos at {manifest["fps"]} FPS', flush=True)
    for command, cwd in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=cwd, env=env, check=True)
    if not args.dry_run:
        (evaluation / 'manifest.json').write_text(json.dumps({
            'status': 'complete', 'data': str(data), 'checkpoint': str(checkpoint),
            'appearance_checkpoint': str(appearance), 'evaluation_scope': scope,
            'frame_ids': manifest['frame_ids'] if args.evaluate_from_start else manifest['evaluation_frame_ids'],
            'branches': ['gt_lighting', 'fitted_appearance'],
        }, indent=2) + '\n')
