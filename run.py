"""Run native reconstruction stages on a generated Gaussian-Garments export."""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from types import FrameType
from utils.body_input import body_directory, body_mesh_paths

# Template command (gaugar environment, from Gaussian-Garments):
# python run.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage initialize --template-frame 0 --aux-root ../data/GaussianGarments/auxiliary --dry-run
# Remove --dry-run to execute; stages: initialize, template, register, appearance, simulation-fit, evaluate.
# Optional GT frame-0 initialization (ClothTransformer exports):
# python run.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000_gt_init --stage initialize --template-frame 0 --initialization gt-mesh
# Simulation fitting uses the separate ContourCraft environment:
# python run.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage simulation-fit --template-frame 0 --simulation-python /path/to/contourcraft/bin/python --contourcraft-root ../ContourCraft --ccraft-data ../data/ContourCraft --cmu-root /path/to/CMU --body-model-root ../data/ContourCraft/aux_data/body_models --checkpoint ../data/ContourCraft/trained_models/contourcraft.pth --steps 1000 --save-every 10 --dry-run
# Use --prepare-only to convert training registrations without fitting.
# Use --resume [PATH] instead of --checkpoint to resume a fitted checkpoint; omitted PATH selects latest.
# Raw-body fitting uses stage1/template_uv.obj as the garment rest reference.
# Evaluation (gaugar environment; omit --evaluate-from-start for held-out frames):
# python run.py --data ../data/GaussianGarments/ClothTransformer/sim_00000 --output output/ClothTransformer/sim_00000 --stage evaluate --simulation-python /home/ljl/miniforge3/envs/ccraft/bin/python --evaluate-from-start


def run_simulation(command: list[str], cwd: Path) -> None:
    """Forward stop requests and wait for the trainer to save its current sequence."""
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)

    def forward_signal(signum: int, frame: FrameType | None) -> None:
        process.send_signal(signum)

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        for signum in previous:
            signal.signal(signum, forward_signal)
        returncode = process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def simulation_fit(args: argparse.Namespace, manifest: dict, root: Path, output: Path) -> None:
    """Delegate the author's behavior fitting to its own environment."""
    assert manifest.get('body_model', 'smplx') in ('smplx', 'raw')
    assert args.initialization == 'colmap', '--initialization applies only to --stage initialize.'
    assert args.simulation_python, 'Set --simulation-python to the ContourCraft environment interpreter.'
    assert args.prepare_only or args.cmu_root, 'Set --cmu-root for the author\'s CMU motion regularization.'
    assert args.steps is None or (args.steps >= 2 and args.steps % 2 == 0), (
        '--steps must be an even integer >= 2 for alternating fitting and CMU batches.')
    assert args.save_every is None or args.save_every >= 1, '--save-every must be >= 1.'
    assert not (args.resume is not None and args.prepare_only), '--resume cannot be used with --prepare-only.'
    assert not (args.resume is not None and args.checkpoint), '--resume cannot be used with --checkpoint.'
    contourcraft = args.contourcraft_root.resolve()
    script = contourcraft / 'fit_gaussian_garments.py'
    interpreter = args.simulation_python.expanduser().absolute()
    command = [str(interpreter), str(script), '--data', str(root), '--output', str(output),
               '--template-frame', str(args.template_frame), '--ccraft-data', str(args.ccraft_data.resolve())]
    for option, value in (('--cmu-root', args.cmu_root), ('--body-model-root', args.body_model_root),
                          ('--checkpoint', args.checkpoint)):
        if value:
            command += [option, str(value.resolve())]
    if args.steps:
        command += ['--steps', str(args.steps)]
    if args.resume is not None:
        resume = 'latest' if args.resume == 'latest' else str(Path(args.resume).expanduser().resolve())
        command += ['--resume', resume]
    if args.save_every is not None:
        command += ['--save-every', str(args.save_every)]
    if args.prepare_only:
        command += ['--prepare-only']
    print(f'Training input: {root / manifest["splits"]["train"]["directory"]}', flush=True)
    print(f'Experiment: {output}', flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    assert script.is_file(), f'Missing ContourCraft fitting entry point: {script}'
    assert interpreter.is_file(), f'Missing ContourCraft environment interpreter: {interpreter}'
    template = output / 'stage2/Template/template.obj'
    assert template.is_file(), f'Run --stage template first: {template}'
    for frame in manifest['splits']['train']['local_frame_ids']:
        mesh = output / 'stage2/train/meshes' / f'frame_{frame:05d}.obj'
        assert mesh.is_file(), f'Complete --stage register before simulation fitting; missing {mesh}'
    run_simulation(command, cwd=contourcraft)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='This experiment directory, including the subject name.')
    parser.add_argument('--stage', choices=('initialize', 'template', 'register', 'appearance', 'simulation-fit', 'evaluate'),
                        required=True)
    parser.add_argument('--template-frame', type=int, default=0)
    parser.add_argument('--initialization', choices=('colmap', 'gt-mesh'), default='colmap',
                        help='Initialization only: native COLMAP (default), or supplied ClothTransformer frame-0 mesh.')
    parser.add_argument('--aux-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/GaussianGarments/auxiliary')
    parser.add_argument('--simulation-python', type=Path,
                        help='Simulation fitting/evaluation: interpreter from the separate ContourCraft environment.')
    parser.add_argument('--contourcraft-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'ContourCraft')
    parser.add_argument('--ccraft-data', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/ContourCraft')
    parser.add_argument('--cmu-root', type=Path, help='Simulation fitting: CMU motion dataset used by the author.')
    parser.add_argument('--body-model-root', type=Path,
                        help='Simulation fitting: licensed body models; defaults to ccraft-data/aux_data/body_models.')
    parser.add_argument('--checkpoint', type=Path,
                        help='Fitting: pretrained ContourCraft checkpoint. Evaluation: fitted checkpoint (default: latest saved step).')
    parser.add_argument('--appearance-checkpoint', type=Path,
                        help='Evaluation: stage3 net.pt (default: highest completed epoch).')
    parser.add_argument('--evaluate-from-start', action='store_true',
                        help='Evaluation: render all frames, including frame 0; default is the held-out suffix.')
    parser.add_argument('--steps', type=int,
                        help='Simulation fitting: total combined batches since step 45000, even and >= 2; '
                             'fresh default 1000, resume default is the saved target.')
    parser.add_argument('--resume', nargs='?', const='latest',
                        help='Simulation fitting: resume PATH, or latest saved fitting checkpoint when PATH is omitted.')
    parser.add_argument('--save-every', type=int,
                        help='Simulation fitting: save every N combined batches (default: 10).')
    parser.add_argument('--prepare-only', action='store_true', help='Simulation fitting: convert data without fitting.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert args.stage == 'evaluate' or not (args.evaluate_from_start or args.appearance_checkpoint), (
        '--evaluate-from-start and --appearance-checkpoint require --stage evaluate.')
    assert args.stage == 'simulation-fit' or not args.prepare_only, '--prepare-only requires --stage simulation-fit.'
    assert args.stage == 'simulation-fit' or (args.resume is None and args.save_every is None), (
        '--resume and --save-every require --stage simulation-fit.')
    root, output, aux = args.data.resolve(), args.output.resolve(), args.aux_root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    assert manifest['component'] in ('clothtransformer_gaussian_garments_export',
                                     'dgarments_gaussian_garments_export')
    subject = manifest['subject']
    train = root / manifest['splits']['train']['directory']
    assert train == root / 'input' / subject / 'train', train
    assert train.is_dir(), train
    body_model = manifest.get('body_model', 'smplx')
    assert body_model in ('raw', 'smplx'), body_model
    assert 0 <= args.template_frame < len(manifest['splits']['train']['local_frame_ids'])
    if args.stage == 'simulation-fit':
        simulation_fit(args, manifest, root, output)
        return
    if args.stage == 'evaluate':
        from evaluation_runner import evaluate
        evaluate(args, manifest, root, output)
        return
    expected_body_dir = 'body_mesh' if body_model == 'raw' else 'smplx'
    assert body_directory(train).name == expected_body_dir
    body_mesh_paths(train, len(manifest['splits']['train']['local_frame_ids']))
    if body_model == 'smplx':
        assert (aux / 'smplx/smplx_vert_segmentation.json').is_file(), aux
    assert sorted(p.name for p in train.parent.iterdir() if p.is_dir()) == ['train']
    if args.initialization == 'gt-mesh':
        assert args.stage == 'initialize', '--initialization applies only to --stage initialize.'
        assert manifest['component'] == 'clothtransformer_gaussian_garments_export', \
            'GT mesh initialization currently supports ClothTransformer exports only.'
        assert args.template_frame == 0, 'GT mesh initialization currently supplies frame 0 only.'
    env = dict(os.environ, GAUGAR_DATA_ROOT=str(root / 'input'),
               GAUGAR_OUTPUT_ROOT=str(output.parent), GAUGAR_AUX_ROOT=str(aux))
    scripts = {'initialize': 's1_initialisation.py', 'template': 's2_registration.py',
               'register': 's2_registration.py', 'appearance': 's3_appearance.py'}
    command = [sys.executable, scripts[args.stage], '--subject', subject,
               '--subject_out', output.name]
    if args.stage != 'appearance':
        command += ['--sequence', 'train']
    if args.stage in ('initialize', 'template'):
        command += ['--template_frame', str(args.template_frame)]
    if args.initialization == 'gt-mesh':
        command = [sys.executable, 's1_gt_initialisation.py', '--data', str(root),
                   '--output', str(output), '--template-frame', str(args.template_frame)]
    destinations = {'initialize': output / 'stage1', 'template': output / 'stage2/Template',
                    'register': output / 'stage2/train', 'appearance': output / 'stage3'}
    print(f'Training input: {train}', flush=True)
    print(f'Experiment: {output}', flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if args.stage == 'register':
        # Template fitting already writes camera/config/input metadata here.
        for name in ('meshes', 'point_cloud', 'renders', 'colmap'):
            path = destinations[args.stage] / name
            assert not path.exists(), f'Registration output already exists: {path}'
    else:
        assert not destinations[args.stage].exists(), destinations[args.stage]
    if args.stage != 'initialize':
        for name in ('template_uv.obj', 'point_cloud.ply'):
            assert (output / 'stage1' / name).is_file(), (
                f'Missing stage1/{name}. Complete initialization first; native COLMAP '
                'also requires manual UV unwrapping.')
    if args.stage in ('register', 'appearance'):
        assert (output / 'stage2/Template').is_dir(), 'Run --stage template first.'
    if args.stage == 'appearance':
        meshes = output / 'stage2/train/meshes'
        for frame in manifest['splits']['train']['local_frame_ids']:
            assert (meshes / f'frame_{frame:05d}.obj').is_file(), meshes / f'frame_{frame:05d}.obj'
    output.mkdir(parents=True, exist_ok=True)
    record = output / 'experiment.json'
    config = {'data': str(root), 'subject': subject, 'template_frame': args.template_frame,
              'aux_root': str(aux)}
    if record.exists():
        assert json.loads(record.read_text()) == config, 'Use the same dataset and template frame.'
    else:
        record.write_text(json.dumps(config, indent=2) + '\n')
    subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env, check=True)


if __name__ == '__main__':
    main()
