"""Evaluation checkpoint selection and split-command integration checks."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation_runner import evaluate, select_appearance, select_checkpoint
from render_evaluation import finish_gt_lighting_images

# Template command (gaugar environment, from Gaussian-Garments):
# python -m unittest discover -s tests -p test_evaluation.py -v


class EvaluationTests(unittest.TestCase):
    def test_numeric_checkpoints_and_explicit_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / 'stage4/checkpoints'
            checkpoints.mkdir(parents=True)
            for name in ('step_9.pth', 'step_100.pth', 'step_bad.pth'):
                (checkpoints / name).touch()
            self.assertEqual(select_checkpoint(root, None).name, 'step_100.pth')
            self.assertEqual(select_checkpoint(root, checkpoints / 'step_9.pth').name, 'step_9.pth')
            for epoch in (2, 11):
                destination = root / f'stage3/epoch{epoch}'
                destination.mkdir(parents=True)
                (destination / 'net.pt').touch()
            self.assertEqual(select_appearance(root, None).parent.name, 'epoch11')

    def test_missing_fitted_checkpoint_is_not_pretrained_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AssertionError, 'simulation-fit first'):
                select_checkpoint(Path(directory), None)

    def test_both_ranges_share_rollout_and_both_renderers(self) -> None:
        self.check_both_ranges('smplx')

    def test_raw_body_uses_same_continuous_rollout_and_renderers(self) -> None:
        self.check_both_ranges('raw')

    def check_both_ranges(self, body_model: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for file in ('simulation/bin/python', 'render/bin/python', 'render/bin/ffmpeg', 'config.json',
                         'render/lib/libstdc++.so.6',
                         'stage4/finetune.yaml', 'stage4/preparation.json',
                         'stage4/checkpoints/step_2.pth', 'stage3/epoch5/net.pt',
                         'preparation/sequence.npz', 'capture/manifest.json', 'capture/cam_info.json'):
                path = root / file
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / 'config.json').write_text(json.dumps({'render_python': str(root / 'python')}))
            args = argparse.Namespace(initialization='colmap', simulation_python=root / 'simulation/bin/python',
                                      render_python=root / 'render/bin/python',
                                      evaluate_from_images=False,
                                      checkpoint=None, appearance_checkpoint=None, evaluate_from_start=False,
                                      contourcraft_root=Path(__file__).resolve().parents[2] / 'ContourCraft',
                                      aux_root=root, dry_run=False)
            manifest = {'body_model': body_model, 'fps': 25, 'frame_ids': list(range(10)),
                        'evaluation_frame_ids': [8, 9]}
            for full in (False, True):
                args.evaluate_from_start = full
                scope = 'full_sequence' if full else 'held_out'
                # Emulate only the subprocess outputs; inspect the real orchestration.
                def execute(command: list[str], **kwargs: object) -> None:
                    destination = Path(command[command.index('--evaluation') + 1])
                    destination.mkdir(parents=True, exist_ok=True)
                with patch.dict('evaluation_runner.os.environ', {'LD_PRELOAD': '/simulation/lib/libstdc++.so.6'}), \
                        patch('evaluation_runner.subprocess.run', side_effect=execute) as run:
                    evaluate(args, manifest, root, root)
                commands = [call.args[0] for call in run.call_args_list]
                self.assertEqual(len(commands), 4)
                self.assertEqual([call.kwargs['env']['PATH'].split(':')[0] for call in run.call_args_list],
                                 [str(root / 'simulation/bin')] + [str(root / 'render/bin')] * 3)
                self.assertEqual([call.kwargs['env']['LD_PRELOAD'] for call in run.call_args_list],
                                 ['/simulation/lib/libstdc++.so.6'] + [str(root / 'render/lib/libstdc++.so.6')] * 3)
                self.assertEqual(commands[0][0], str(root / 'simulation/bin/python'))
                self.assertEqual([command[0] for command in commands[1:]],
                                 [str(root / 'render/bin/python')] * 3)
                self.assertEqual('--evaluate-from-start' in commands[0], full)
                self.assertEqual([command[-1] for command in commands[1:]],
                                 ['gt-lighting', 'textures', 'fitted-appearance'])
                result = json.loads((root / 'evaluation' / scope / 'manifest.json').read_text())
                self.assertEqual(result['frame_ids'], list(range(10)) if full else [8, 9])
                destination = root / 'evaluation' / scope
                (destination / 'predictions.npz').touch()
                (destination / 'rollout.json').write_text(json.dumps({
                    'status': 'complete', 'data': str(root),
                    'checkpoint': str(root / 'stage4/checkpoints/step_2.pth'),
                    'evaluation_scope': scope, 'frame_ids': result['frame_ids'],
                }))
                args.evaluate_from_images = True
                with patch('evaluation_runner.subprocess.run', side_effect=execute) as run:
                    evaluate(args, manifest, root, root)
                self.assertEqual([call.args[0][-1] for call in run.call_args_list],
                                 ['gt-lighting-videos', 'textures', 'fitted-appearance'])
                self.assertTrue(all(call.kwargs['env']['LD_PRELOAD'] == str(root / 'render/lib/libstdc++.so.6')
                                    for call in run.call_args_list))
                args.evaluate_from_images = False

            (root / 'render/bin/ffmpeg').unlink()
            with patch('evaluation_runner.subprocess.run') as run:
                with self.assertRaisesRegex(AssertionError, 'encoder'):
                    evaluate(args, manifest, root, root)
                run.assert_not_called()

    def test_finish_images_checks_body_plates_and_reuses_partial_gt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / 'evaluation'
            destination = evaluation / 'gt_lighting/Cam001'
            (destination / 'pred').mkdir(parents=True)
            (destination / 'gt').mkdir()
            (destination / 'pred/0080.png').touch()
            reference = root / 'capture/rgbs/Cam001/Cam001_rgb000080.png'
            reference.parent.mkdir(parents=True)
            reference.write_bytes(b'saved reference')
            manifest = {'camera_ids': ['Cam001'], 'fps': 25}
            with patch('render_evaluation.encode_comparison') as encode:
                with self.assertRaisesRegex(AssertionError, 'Incomplete saved render'):
                    finish_gt_lighting_images(root, evaluation, manifest, [80])
                encode.assert_not_called()
                plate = evaluation / 'body_plate/Cam001/0080.png'
                plate.parent.mkdir(parents=True)
                plate.touch()
                finish_gt_lighting_images(root, evaluation, manifest, [80])
                encode.assert_called_once_with(destination, [80], 25, atomic=True)
            self.assertEqual((destination / 'gt/0080.png').read_bytes(), b'saved reference')
            self.assertEqual(json.loads((destination.parent / 'manifest.json').read_text())['status'], 'complete')

    def test_unsupported_body_stops_before_launch(self) -> None:
        with patch('evaluation_runner.subprocess.run') as run:
            with self.assertRaisesRegex(AssertionError, 'Unsupported'):
                evaluate(argparse.Namespace(), {'body_model': 'unknown'}, Path('/unused'), Path('/unused'))
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
