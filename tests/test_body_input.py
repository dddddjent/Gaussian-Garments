"""Input dispatch checks without reconstruction or model assets."""

import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run
from utils.body_input import body_directory, body_mesh_paths, camera_directories

# Template command (activate gaugar; run from Gaussian-Garments):
# python -m unittest discover -s tests -p test_body_input.py -v


class BodyInputTests(unittest.TestCase):
    def test_discovery_and_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Cam001').mkdir()
            (root / 'unrelated').mkdir()
            (root / 'cameras.json').write_text(json.dumps({'Cam001': {}}))
            (root / 'body_mesh').mkdir()
            for frame in range(2):
                (root / 'body_mesh' / f'{frame:05d}.ply').touch()
            self.assertEqual(camera_directories(root), [root / 'Cam001'])
            self.assertEqual(body_directory(root).name, 'body_mesh')
            self.assertEqual(len(body_mesh_paths(root, 2)), 2)
            with self.assertRaisesRegex(AssertionError, 'frame count'):
                body_mesh_paths(root, 3)
            (root / 'smplx').mkdir()
            with self.assertRaisesRegex(AssertionError, 'exactly one'):
                body_directory(root)
            (root / 'smplx').rmdir()
            (root / 'body_mesh/00001.ply').rename(root / 'body_mesh/00002.ply')
            with self.assertRaises(AssertionError):
                body_mesh_paths(root, 2)

    def test_raw_launcher_all_stages_without_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'data'
            output = Path(directory) / 'output'
            train = root / 'input/subject/train'
            (train / 'body_mesh').mkdir(parents=True)
            (train / 'body_mesh/00000.ply').touch()
            manifest = {'status': 'complete', 'body_model': 'raw', 'subject': 'subject',
                        'consumer_status': 'raw_mesh_loader_required',
                        'component': 'clothtransformer_gaussian_garments_export',
                        'splits': {'train': {'directory': 'input/subject/train', 'local_frame_ids': [0]}}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            argv = ['run.py', '--data', str(root), '--output', str(output),
                    '--aux-root', str(root / 'absent_auxiliary')]
            for stage in ('initialize', 'template', 'register', 'appearance'):
                if stage == 'template':
                    (output / 'stage1').mkdir()
                    for name in ('template_uv.obj', 'point_cloud.ply'):
                        (output / 'stage1' / name).touch()
                if stage == 'register':
                    (output / 'stage2/Template').mkdir(parents=True)
                    (output / 'stage2/train').mkdir()
                    for name in ('cameras.json', 'cfg_args', 'input.ply'):
                        (output / 'stage2/train' / name).write_text('template metadata')
                if stage == 'appearance':
                    (output / 'stage2/train/meshes').mkdir(parents=True)
                    (output / 'stage2/train/meshes/frame_00000.obj').touch()
                with self.subTest(stage=stage), patch('sys.argv', [*argv, '--stage', stage]), \
                        patch('run.subprocess.run') as launch:
                    run.main()
                    launch.assert_called_once()
                    if stage == 'initialize':
                        self.assertEqual(launch.call_args.args[0][1], 's1_initialisation.py')
                    self.assertEqual(launch.call_args.kwargs['env']['GAUGAR_DATA_ROOT'], str(root / 'input'))
                if stage == 'register':
                    for name in ('cameras.json', 'cfg_args', 'input.ply'):
                        self.assertEqual((output / 'stage2/train' / name).read_text(), 'template metadata')
                    for name in ('meshes', 'point_cloud', 'renders', 'colmap'):
                        artifact = output / 'stage2/train' / name
                        artifact.mkdir()
                        with self.subTest(artifact=name), patch('sys.argv', [*argv, '--stage', stage]), \
                                patch('run.subprocess.run') as launch, \
                                self.assertRaisesRegex(AssertionError, 'Registration output already exists'):
                            run.main()
                        launch.assert_not_called()
                        artifact.rmdir()
            manifest['body_model'] = 'smplx'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch('sys.argv', [*argv, '--stage', 'initialize', '--dry-run']), self.assertRaises(AssertionError):
                run.main()

    def test_gt_initialization_is_explicit_and_frame_zero_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'data'
            output = Path(directory) / 'output'
            train = root / 'input/subject/train'
            (train / 'body_mesh').mkdir(parents=True)
            for frame in range(2):
                (train / 'body_mesh' / f'{frame:05d}.ply').touch()
            manifest = {'status': 'complete', 'body_model': 'raw', 'subject': 'subject',
                        'component': 'clothtransformer_gaussian_garments_export',
                        'splits': {'train': {'directory': 'input/subject/train', 'local_frame_ids': [0, 1]}}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            argv = ['run.py', '--data', str(root), '--output', str(output),
                    '--initialization', 'gt-mesh']
            with patch('sys.argv', [*argv, '--stage', 'initialize']), patch('run.subprocess.run') as launch:
                run.main()
                command = launch.call_args.args[0]
                self.assertEqual(command[1], 's1_gt_initialisation.py')
                self.assertEqual(command[command.index('--data') + 1], str(root))
            for extra in (['--stage', 'initialize', '--template-frame', '1'],
                          ['--stage', 'template']):
                with self.subTest(extra=extra), patch('sys.argv', [*argv, *extra]), \
                        patch('run.subprocess.run') as launch, self.assertRaises(AssertionError):
                    run.main()
                launch.assert_not_called()
            manifest['component'] = 'dgarments_gaussian_garments_export'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch('sys.argv', [*argv, '--stage', 'initialize']), self.assertRaises(AssertionError):
                run.main()

    def test_simulation_launcher_requires_complete_registration_and_own_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, output, contourcraft = base / 'data', base / 'output', base / 'ContourCraft'
            (root / 'input/subject/train').mkdir(parents=True)
            manifest = {'status': 'complete', 'body_model': 'smplx', 'subject': 'subject',
                        'component': 'clothtransformer_gaussian_garments_export',
                        'splits': {'train': {'directory': 'input/subject/train', 'local_frame_ids': [0, 1]}}}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            interpreter = base / 'ccraft/bin/python'
            argv = ['run.py', '--data', str(root), '--output', str(output), '--stage', 'simulation-fit',
                    '--simulation-python', str(interpreter), '--contourcraft-root', str(contourcraft),
                    '--ccraft-data', str(base / 'ccraft-data'), '--body-model-root', str(base / 'models'),
                    '--cmu-root', str(base / 'CMU'), '--checkpoint', str(base / 'checkpoint.pth'), '--steps', '2']
            # Planning must work before the separate environment or assets have been installed.
            with patch('sys.argv', [*argv, '--dry-run']), patch('run.run_simulation') as launch:
                run.main()
            launch.assert_not_called()
            self.assertFalse(output.exists())
            contourcraft.mkdir()
            (contourcraft / 'fit_gaussian_garments.py').touch()
            interpreter.parent.mkdir(parents=True)
            interpreter.touch()
            (output / 'stage2/Template').mkdir(parents=True)
            (output / 'stage2/Template/template.obj').touch()
            (output / 'stage2/train/meshes').mkdir(parents=True)
            (output / 'stage2/train/meshes/frame_00000.obj').touch()
            with patch('sys.argv', argv), patch('run.run_simulation') as launch, \
                    self.assertRaisesRegex(AssertionError, 'Complete --stage register'):
                run.main()
            launch.assert_not_called()
            (output / 'stage2/train/meshes/frame_00001.obj').touch()
            with patch('sys.argv', argv), patch('run.run_simulation') as launch:
                run.main()
            command = launch.call_args.args[0]
            self.assertEqual(command[:2], [str(interpreter), str(contourcraft / 'fit_gaussian_garments.py')])
            self.assertEqual(launch.call_args.kwargs['cwd'], contourcraft)
            for option, value in (('--data', root), ('--output', output), ('--ccraft-data', base / 'ccraft-data'),
                                  ('--body-model-root', base / 'models'), ('--cmu-root', base / 'CMU'),
                                  ('--checkpoint', base / 'checkpoint.pth'), ('--steps', 2)):
                self.assertEqual(command[command.index(option) + 1], str(value))
            # Conversion needs no CMU motions; the fitting script validates its own model inputs.
            cmu_option = argv.index('--cmu-root')
            prepare_argv = argv[:cmu_option] + argv[cmu_option + 2:]
            with patch('sys.argv', [*prepare_argv, '--prepare-only']), patch('run.run_simulation') as launch:
                run.main()
            self.assertIn('--prepare-only', launch.call_args.args[0])
            with patch('sys.argv', prepare_argv), patch('run.run_simulation') as launch, \
                    self.assertRaisesRegex(AssertionError, '--cmu-root'):
                run.main()
            launch.assert_not_called()
            python_option = argv.index('--simulation-python')
            missing_python_argv = argv[:python_option] + argv[python_option + 2:]
            with patch('sys.argv', missing_python_argv), self.assertRaisesRegex(AssertionError, '--simulation-python'):
                run.main()
            for steps in ('0', '1', '3'):
                with self.subTest(steps=steps), patch('sys.argv', [*argv[:-1], steps]), \
                        self.assertRaisesRegex(AssertionError, 'even integer'):
                    run.main()
            manifest['body_model'] = 'raw'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            with patch('sys.argv', argv), patch('run.run_simulation') as launch:
                run.main()
            command = launch.call_args.args[0]
            self.assertNotIn('--rest-mesh', command)
            checkpoint_option = argv.index('--checkpoint')
            resume_argv = argv[:checkpoint_option] + argv[checkpoint_option + 2:]
            resume_argv = resume_argv[:-2]  # Omit --steps to retain the saved fitting target.
            for resume in (['--resume'], ['--resume', str(base / 'step_45002.pth')]):
                with self.subTest(resume=resume), patch('sys.argv', [*resume_argv, *resume, '--save-every', '4']), \
                        patch('run.run_simulation') as launch:
                    run.main()
                command = launch.call_args.args[0]
                self.assertEqual(command[command.index('--resume') + 1],
                                 'latest' if len(resume) == 1 else str(base / 'step_45002.pth'))
                self.assertEqual(command[command.index('--save-every') + 1], '4')
                self.assertNotIn('--steps', command)
                self.assertNotIn('--checkpoint', command)
            for extra in (['--resume', '--prepare-only'],
                          ['--resume', '--checkpoint', str(base / 'checkpoint.pth')],
                          ['--save-every', '0']):
                with self.subTest(extra=extra), patch('sys.argv', [*resume_argv, *extra]), \
                        patch('run.run_simulation') as launch, self.assertRaises(AssertionError):
                    run.main()
                launch.assert_not_called()

    def test_simulation_stop_signals_are_forwarded_and_handlers_restored(self) -> None:
        previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
        with patch('run.subprocess.Popen') as popen:
            process = popen.return_value

            def complete_sequence() -> int:
                for signum in previous:
                    handler = signal.getsignal(signum)
                    self.assertTrue(callable(handler))
                    handler(signum, None)
                return 0

            process.wait.side_effect = complete_sequence
            run.run_simulation(['fitting-python', 'fit.py'], Path('/tmp'))
            popen.assert_called_once_with(['fitting-python', 'fit.py'], cwd=Path('/tmp'), start_new_session=True)
            self.assertEqual([call.args[0] for call in process.send_signal.call_args_list],
                             [signal.SIGINT, signal.SIGTERM])
            process.kill.assert_not_called()
            process.terminate.assert_not_called()
        for signum, handler in previous.items():
            self.assertEqual(signal.getsignal(signum), handler)
        with patch('run.subprocess.Popen') as popen:
            popen.return_value.wait.return_value = 3
            with self.assertRaises(subprocess.CalledProcessError):
                run.run_simulation(['fitting-python', 'fit.py'], Path('/tmp'))
        for signum, handler in previous.items():
            self.assertEqual(signal.getsignal(signum), handler)


if __name__ == '__main__':
    unittest.main()
